import json
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings("ignore")

class SyllogismDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create input text
        text = f"Syllogism: {item['syllogism']}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(int(item['validity']), dtype=torch.long)
        }

def calculate_content_effect_score(predictions, labels, plausibility):
    """
    Calculate content effect as: Accuracy(Plausible) - Accuracy(Implausible)
    As specified in the methodology
    """
    plausible_mask = np.array(plausibility) == True
    implausible_mask = np.array(plausibility) == False
    
    # Calculate accuracy for plausible examples
    plausible_preds = np.array(predictions)[plausible_mask]
    plausible_labels = np.array(labels)[plausible_mask]
    plausible_acc = np.mean(plausible_preds == plausible_labels) if len(plausible_preds) > 0 else 0
    
    # Calculate accuracy for implausible examples
    implausible_preds = np.array(predictions)[implausible_mask]
    implausible_labels = np.array(labels)[implausible_mask]
    implausible_acc = np.mean(implausible_preds == implausible_labels) if len(implausible_preds) > 0 else 0
    
    content_effect_score = plausible_acc - implausible_acc
    
    return {
        'plausible_accuracy': plausible_acc,
        'implausible_accuracy': implausible_acc,
        'content_effect_score': content_effect_score
    }

def compute_metrics(eval_pred):
    """Compute metrics for evaluation during training"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = np.mean(predictions == labels)
    
    return {
        'accuracy': accuracy,
    }

def main():
    print("Loading balanced split data...")
    
    # Load the balanced splits
    with open('data_splits/train_split.json', 'r') as f:
        train_data = json.load(f)
    
    with open('data_splits/val_split.json', 'r') as f:
        val_data = json.load(f)
    
    with open('data_splits/test_split.json', 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded balanced splits:")
    print(f"- Training: {len(train_data)} examples")
    print(f"- Validation: {len(val_data)} examples") 
    print(f"- Test: {len(test_data)} examples")
    
    # Initialize tokenizer and model
    model_name = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Create datasets
    train_dataset = SyllogismDataset(train_data, tokenizer)
    val_dataset = SyllogismDataset(val_data, tokenizer)
    
    # Training arguments optimized for RTX 4060
    training_args = TrainingArguments(
        output_dir='./roberta_balanced_baseline',
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir='./logs',
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=None,  # Disable wandb
        seed=42,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    print("Starting training...")
    trainer.train()
    
    print("\nEvaluating on test set...")
    
    # Create test dataset and get predictions
    test_dataset = SyllogismDataset(test_data, tokenizer)
    test_predictions = trainer.predict(test_dataset)
    
    # Extract predictions and ground truth
    pred_labels = np.argmax(test_predictions.predictions, axis=1)
    true_labels = test_predictions.label_ids
    plausibility = [item['plausibility'] for item in test_data]
    
    # Calculate baseline metrics
    overall_accuracy = np.mean(pred_labels == true_labels)
    content_metrics = calculate_content_effect_score(pred_labels, true_labels, plausibility)
    
    # Print results in the format specified
    print("\n" + "="*60)
    print("BASELINE MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Plausible Accuracy (VP + IP): {content_metrics['plausible_accuracy']:.4f}")
    print(f"Implausible Accuracy (VI + II): {content_metrics['implausible_accuracy']:.4f}")
    print(f"Content Effect Score: {content_metrics['content_effect_score']:.4f}")
    print("="*60)
    
    # Additional category-wise breakdown
    print(f"\nDetailed Category Breakdown:")
    print("-" * 40)
    
    categories = {
        'VP': [],  # Valid, Plausible
        'IP': [],  # Invalid, Plausible
        'VI': [],  # Valid, Implausible
        'II': []   # Invalid, Implausible
    }
    
    for i, item in enumerate(test_data):
        pred = bool(pred_labels[i])
        true_val = item['validity']
        plaus = item['plausibility']
        
        if true_val and plaus:
            categories['VP'].append(pred == true_val)
        elif not true_val and plaus:
            categories['IP'].append(pred == true_val)
        elif true_val and not plaus:
            categories['VI'].append(pred == true_val)
        else:
            categories['II'].append(pred == true_val)
    
    for cat_name, results in categories.items():
        if results:
            acc = np.mean(results)
            count = len(results)
            valid_str = "Valid" if cat_name[0] == 'V' else "Invalid"
            plaus_str = "Plausible" if cat_name[1] == 'P' else "Implausible"
            print(f"{cat_name} ({valid_str}, {plaus_str}): {acc:.3f} ({count:2d} examples)")
    
    # Save the model
    trainer.save_model('./roberta_balanced_final')
    tokenizer.save_pretrained('./roberta_balanced_final')
    
    # Save detailed results
    results = {
        'overall_accuracy': overall_accuracy,
        'content_effect_metrics': content_metrics,
        'category_breakdown': {cat: np.mean(results) if results else 0.0 
                             for cat, results in categories.items()},
        'test_predictions': pred_labels.tolist(),
        'test_labels': true_labels.tolist(),
        'test_plausibility': plausibility
    }
    
    with open('balanced_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel saved to: ./roberta_balanced_final")
    print(f"Results saved to: balanced_baseline_results.json")
    print("\nBaseline training complete! This model is ready for teacher-student distillation.")

if __name__ == "__main__":
    main()