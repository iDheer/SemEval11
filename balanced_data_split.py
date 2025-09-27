import json
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter, defaultdict
import os

def categorize_data(data):
    """
    Categorize syllogisms into four groups based on validity and plausibility:
    VP (Valid, Plausible), IP (Invalid, Plausible), 
    VI (Valid, Implausible), II (Invalid, Implausible)
    """
    categories = {
        'VP': [],  # Valid, Plausible
        'IP': [],  # Invalid, Plausible  
        'VI': [],  # Valid, Implausible
        'II': []   # Invalid, Implausible
    }
    
    for item in data:
        validity = item['validity']
        plausibility = item['plausibility']
        
        if validity and plausibility:
            categories['VP'].append(item)
        elif not validity and plausibility:
            categories['IP'].append(item)
        elif validity and not plausibility:
            categories['VI'].append(item)
        else:  # not validity and not plausibility
            categories['II'].append(item)
    
    return categories

def print_category_stats(categories, title="Data Distribution"):
    """Print statistics for each category"""
    print(f"\n{title}")
    print("=" * 50)
    total = sum(len(cat) for cat in categories.values())
    
    for cat_name, items in categories.items():
        count = len(items)
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"{cat_name} (Valid={cat_name[0]=='V'}, Plausible={cat_name[1]=='P'}): "
              f"{count:3d} examples ({percentage:5.1f}%)")
    
    print(f"Total: {total} examples")
    print("=" * 50)

def create_balanced_splits(categories, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create balanced train/val/test splits ensuring each category is represented
    proportionally in each split
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    train_data = []
    val_data = []
    test_data = []
    
    print(f"\nCreating splits: {train_ratio:.0%} train, {val_ratio:.0%} val, {test_ratio:.0%} test")
    print("-" * 50)
    
    for cat_name, items in categories.items():
        if len(items) == 0:
            print(f"Warning: Category {cat_name} has no examples!")
            continue
            
        # Calculate split sizes
        n_total = len(items)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val  # Remainder goes to test
        
        # Shuffle items for random split
        np.random.shuffle(items)
        
        # Split the data
        cat_train = items[:n_train]
        cat_val = items[n_train:n_train + n_val]
        cat_test = items[n_train + n_val:]
        
        train_data.extend(cat_train)
        val_data.extend(cat_val)
        test_data.extend(cat_test)
        
        print(f"{cat_name}: {len(cat_train):2d} train, {len(cat_val):2d} val, {len(cat_test):2d} test")
    
    # Shuffle the final splits
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)
    
    return train_data, val_data, test_data

def calculate_content_effect_baseline(predictions, labels, plausibility):
    """
    Calculate content effect as described: Accuracy(Plausible) - Accuracy(Implausible)
    """
    plausible_correct = sum(1 for p, l, pl in zip(predictions, labels, plausibility) 
                           if p == l and pl == True)
    plausible_total = sum(1 for pl in plausibility if pl == True)
    
    implausible_correct = sum(1 for p, l, pl in zip(predictions, labels, plausibility) 
                             if p == l and pl == False)
    implausible_total = sum(1 for pl in plausibility if pl == False)
    
    plausible_acc = plausible_correct / plausible_total if plausible_total > 0 else 0
    implausible_acc = implausible_correct / implausible_total if implausible_total > 0 else 0
    
    content_effect = plausible_acc - implausible_acc
    
    return {
        'plausible_accuracy': plausible_acc,
        'implausible_accuracy': implausible_acc,
        'content_effect_score': content_effect
    }

def main():
    print("Loading and categorizing training data...")
    
    # Load the original data
    with open('train_data.json', 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} total examples")
    
    # Set random seed for reproducible splits
    np.random.seed(42)
    
    # Categorize data
    categories = categorize_data(data)
    print_category_stats(categories, "Original Data Distribution")
    
    # Check if we have examples in all categories
    empty_categories = [cat for cat, items in categories.items() if len(items) == 0]
    if empty_categories:
        print(f"\nWarning: Empty categories found: {empty_categories}")
        print("This may indicate data imbalance issues.")
    
    # Create balanced splits
    train_data, val_data, test_data = create_balanced_splits(
        categories, 
        train_ratio=0.7, 
        val_ratio=0.15, 
        test_ratio=0.15
    )
    
    # Verify splits maintain category balance
    print(f"\nFinal split sizes:")
    print(f"Train: {len(train_data)} examples")
    print(f"Val:   {len(val_data)} examples") 
    print(f"Test:  {len(test_data)} examples")
    
    # Check category distribution in each split
    for split_name, split_data in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
        split_categories = categorize_data(split_data)
        print_category_stats(split_categories, f"{split_name} Split Distribution")
    
    # Save splits to separate files
    os.makedirs('data_splits', exist_ok=True)
    
    with open('data_splits/train_split.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open('data_splits/val_split.json', 'w') as f:
        json.dump(val_data, f, indent=2)
        
    with open('data_splits/test_split.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\nSaved splits to:")
    print(f"- data_splits/train_split.json ({len(train_data)} examples)")
    print(f"- data_splits/val_split.json ({len(val_data)} examples)")
    print(f"- data_splits/test_split.json ({len(test_data)} examples)")
    
    # Create summary statistics
    summary = {
        'total_examples': len(data),
        'original_distribution': {cat: len(items) for cat, items in categories.items()},
        'split_sizes': {
            'train': len(train_data),
            'validation': len(val_data), 
            'test': len(test_data)
        },
        'train_distribution': {cat: len(items) for cat, items in categorize_data(train_data).items()},
        'val_distribution': {cat: len(items) for cat, items in categorize_data(val_data).items()},
        'test_distribution': {cat: len(items) for cat, items in categorize_data(test_data).items()}
    }
    
    with open('data_splits/split_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nData splitting complete! Use these files for training your baseline model.")

if __name__ == "__main__":
    main()