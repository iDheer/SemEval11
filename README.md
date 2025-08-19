# SemEval-2026 Task 11: Ollama-Based Knowledge Distillation Framework

This project provides a complete, end-to-end framework to compete in **SemEval-2026 Task 11: Disentangling Content and Formal Reasoning in Large Language Models** using locally hosted Ollama models.

## 🎯 Project Overview

Our approach uses **Knowledge Distillation** with **Symbolic Chain-of-Thought (S-CoT)** reasoning:

- **Teacher Model:** `deepseek-r1:8b` - Generates high-quality symbolic reasoning data
- **Student Model:** `gemma3:4b-it-q4_K_M` - Fine-tuned specialist for syllogistic reasoning

### Key Features
- ✅ **Subtask 1:** Validity classification with content effect metrics
- ✅ **Subtask 2:** Validity + premise selection with irrelevant premise handling
- ✅ **Local Inference:** Uses Ollama for all model interactions (no HuggingFace needed)
- ✅ **Knowledge Distillation:** Teacher creates training data, Student specializes
- ✅ **SemEval Compliance:** Implements exact metrics from competition requirements

## 🚀 Quick Start

### Prerequisites
1. **Python 3.9+** with pip
2. **Ollama** installed and running
3. **NVIDIA GPU** (recommended for training)

Important note about automation
- The scripts in this repository now include an automated helper that will (best-effort) restart the local Ollama server and pull the requested model before running. This means you can run `python prepare_dataset.py` or `python train.py ...` and the code will try to stop a running Ollama process, start it cleanly, and pull the specified model so only that model will be resident in GPU memory.
- For the automation to work the `ollama` CLI must be available on your PATH and you should be on Windows PowerShell (the helper uses PowerShell / taskkill for stop/start). If you prefer to manage Ollama yourself, you can pre-pull models with `ollama pull <model>` and run `ollama serve` manually.

### Setup
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd SemEval11

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Ensure Ollama CLI is available (optional)
# The scripts will attempt to restart/pull models automatically if `ollama` is on PATH. You can also pre-pull models:
ollama pull deepseek-r1:8b
ollama pull gemma3:4b-it-q4_K_M
```

### Complete Workflow

#### Step 1: Generate Training Data (Teacher's Job)
```bash
# Use deepseek-r1:8b to create S-CoT training data
python prepare_dataset.py
```
Note: `prepare_dataset.py` will try to restart Ollama and pull `deepseek-r1:8b` automatically if needed.
**Output:** Creates `training_data_st1_scot.jsonl` and `training_data_st2_scot.jsonl`

#### Step 2: Fine-tune Student Models
```bash
# Train Gemma for Subtask 1 (clean reasoning)
python train.py \
    --model_name "gemma3:4b-it-q4_K_M" \
    --dataset_path "training_data_st1_scot.jsonl" \
    --output_dir "./models/gemma-st1" \
    --epochs 3

# Train Gemma for Subtask 2 (noisy premises)
python train.py \
    --model_name "gemma3:4b-it-q4_K_M" \
    --dataset_path "training_data_st2_scot.jsonl" \
    --output_dir "./models/gemma-st2" \
    --epochs 3
```

#### Step 3: Evaluate Performance
```bash
# Evaluate Subtask 1 model
python evaluate_subtask1.py \
    --model_name "gemma3:4b-it-q4_K_M" \
    --test_data_path "pilot_dataset.json"

# Evaluate Subtask 2 model
python evaluate_subtask2.py \
    --model_name "gemma3:4b-it-q4_K_M" \
    --test_data_path "training_data_st2_scot.jsonl"
```
Note: evaluation scripts will restart Ollama and ensure the chosen `--model_name` is available before running.

#### Step 4: Analyze Errors
```bash
# Analyze evaluation results for insights
python analyze_errors.py
```

#### Step 5: Generate Final Submission
```bash
# Create competition submission file
python generate_submission.py \
    --model_name "gemma3:4b-it-q4_K_M" \
    --test_data_path "official_test_set.jsonl" \
    --output_file "submission.jsonl"
```

## 📊 Understanding the Metrics

### Subtask 1 Metrics
- **Accuracy:** Basic correctness of validity predictions
- **Intra-Plausibility Content Effect:** Bias within plausible/implausible groups
- **Cross-Plausibility Content Effect:** Bias between plausible/implausible groups
- **Total Content Effect:** Average of intra and cross effects (lower is better)
- **Ranking Ratio:** Accuracy / Total Content Effect (higher is better)

### Subtask 2 Metrics
- **Premise Selection F1:** How well the model identifies relevant premises
- **Subtask 2 Ranking Ratio:** (Accuracy + F1) / 2 / Content Effect

## 🔧 Project Structure

```
SemEval11/
├── prepare_dataset.py      # Data generation (Teacher)
├── train.py               # Model fine-tuning (Student)
├── evaluate_subtask1.py   # Subtask 1 evaluation
├── evaluate_subtask2.py   # Subtask 2 evaluation
├── analyze_errors.py      # Error analysis
├── generate_submission.py # Final submission
├── run_experiments.sh     # Hyperparameter sweeping
├── pilot_dataset.json     # Sample test data
└── requirements.txt       # Python dependencies
```

## 🎓 The Knowledge Distillation Approach

### Why This Works
1. **Teacher Expertise:** `deepseek-r1:8b` generates high-quality symbolic reasoning
2. **Student Specialization:** `gemma3:4b-it-q4_K_M` learns focused syllogistic skills
3. **Symbolic Reasoning:** Forces models to use logical structure over content
4. **Task-Specific Training:** Separate models for clean vs. noisy reasoning

### The S-CoT Format
Our Teacher generates structured reasoning:
```
Terms: [Dogs, Animals, Pets]
Symbols: [A = Dogs, B = Animals, C = Pets]
Premise 1: All A are B (All Dogs are Animals)
Premise 2: Some B are C (Some Animals are Pets)
Conclusion: Some A are C (Some Dogs are Pets)
Analysis: [Symbolic logical analysis]
Validity: True
```

## 🔄 Iterative Improvement

1. **Run Evaluation:** Get baseline metrics
2. **Analyze Errors:** Understand failure patterns
3. **Refine Prompts:** Improve Teacher's S-CoT generation
4. **Retrain Student:** Use improved data
5. **Repeat:** Until performance plateaus

## ⚡ Advanced Usage

### Hyperparameter Sweeping
```bash
# Run automated experiments with different configurations
bash run_experiments.sh
```

### Custom Model Fine-tuning
```bash
# Train with custom parameters
python train.py \
    --model_name "deepseek-r1:8b" \
    --dataset_path "training_data_st1_scot.jsonl" \
    --output_dir "./models/deepseek-st1" \
    --epochs 5 \
    --learning_rate 1e-4
```

## 🐛 Troubleshooting

### Common Issues
1. **Ollama models not found:** Run `ollama pull deepseek-r1:8b` and `ollama pull gemma3:4b-it-q4_K_M`
2. **Automatic Ollama restart/pull fails:** The repository's automation uses the `ollama` CLI, `taskkill`, and PowerShell Start-Process. If the helper cannot restart or pull (permission issues or missing CLI), you can manually do the following in PowerShell:

```powershell
Get-Process -Name ollama -ErrorAction SilentlyContinue | Stop-Process
Start-Process -FilePath 'ollama' -ArgumentList 'serve' -NoNewWindow
ollama pull deepseek-r1:8b
ollama pull gemma3:4b-it-q4_K_M
```

If you prefer to avoid the automatic behavior, you can edit `ollama_utils.py` and change the helpers or call scripts after you start `ollama serve` manually.
2. **CUDA out of memory:** Reduce batch size in training scripts
3. **Training fails:** Ensure Ollama server is running (`ollama serve`)

### Performance Tips
- **Caching:** First run of `prepare_dataset.py` is slow; subsequent runs use cache
- **GPU Memory:** Monitor with `nvidia-smi` during training
- **Batch Size:** Adjust based on your GPU memory

## 📚 Key Files Explained

- **`prepare_dataset.py`:** Teacher creates S-CoT training data using `deepseek-r1:8b`
- **`train.py`:** Student fine-tuning using Ollama's training capabilities
- **`evaluate_subtask*.py`:** Competition-compliant evaluation with exact SemEval metrics
- **`analyze_errors.py`:** Diagnostic tool for understanding model failures

## 🏆 Competition Submission

When official test data is released:
1. Run your best Student model on test data using `generate_submission.py`
2. Submit the generated `submission.jsonl` file
3. Rankings based on the Ranking Ratio metrics

## 📄 License

This project follows the SemEval-2026 Task 11 guidelines and is intended for academic competition use.

---

**Quick Start Summary:**
```bash
pip install -r requirements.txt
python prepare_dataset.py
python train.py --model_name gemma3:4b-it-q4_K_M --dataset_path training_data_st1_scot.jsonl --output_dir ./models/st1
python evaluate_subtask1.py --model_name gemma3:4b-it-q4_K_M --test_data_path pilot_dataset.json
```