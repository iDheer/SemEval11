# Comprehensive Framework for SemEval-2026 Task 11

This project provides a complete, end-to-end framework to compete in **SemEval-2026 Task 11** using the `deepseek-r1:8b` and `gemma3:4b-it-q4_K_M` models. Our goal is to achieve a top leaderboard rank by employing a novel fine-tuning strategy called **Symbolic Chain-of-Thought (S-CoT)**, along with a robust suite of tools for training, evaluation, and analysis.

## Project Philosophy

Our approach is built on three core principles:
1.  **Specialization**: We create a separate, fine-tuned model for each subtask. A model trained to handle noisy data (Subtask 2) is fundamentally different from one trained on clean data (Subtask 1).
2.  **Data-Centric Strategy**: The quality and design of our training data are paramount. Our scripts focus on creating the perfect "lesson plan" for each specialized model.
3.  **Iterative Improvement**: Winning is not a single shot. This framework is designed for a cycle of training, evaluating, analyzing failures, and then improving the data or training process.

## Core Strategy: Symbolic Chain-of-Thought (S-CoT)

Instead of just predicting `validity: true/false`, we train the model to first perform a more fundamental task: translating the natural language syllogism into its abstract, symbolic form (e.g., "All A are B," "No B are C").

**Why this is a winning strategy:**
*   **It forces content-independence**: By focusing on the symbolic structure, the model learns to ignore the distracting real-world plausibility of the content. This directly combats the "content effect" at the heart of the competition.
*   **It mimics formal reasoning**: This process encourages the model to build a robust, internal representation of the logical problem before arriving at a conclusion, leading to more reliable deductions.

## The Workflow at a Glance

1.  **Prepare Data** (`prepare_dataset.py`)
2.  **Train Specialized Models** (`train.py`)
3.  **Evaluate Performance** (`evaluate_subtask1.py`, `evaluate_subtask2.py`)
4.  **Analyze Weaknesses** (`analyze_errors.py`)
5.  **Iterate & Improve** (Repeat steps 1-4)
6.  **Generate Final Submission** (`generate_submission.py`)

## The Scripts: A Deeper Dive

(The descriptions of each script's purpose and reasoning remain the same as the previous README).

---
*   `prepare_dataset.py`: The data engine.
*   `train.py`: The general-purpose model trainer.
*   `evaluate_subtask1.py` & `evaluate_subtask2.py`: Specialized evaluators for each subtask.
*   `analyze_errors.py`: The diagnostics and error analysis tool.
*   `generate_submission.py`: The script for creating the final competition entry.
---

## Step-by-Step Execution Guide

### Step 1: Setup

1.  **Prerequisites**: Python 3.9+, NVIDIA GPU, and CUDA 12.1.
2.  **Install Dependencies**: `pip install -r requirements.txt`
3.  **Setup Ollama**: Install [Ollama](https://ollama.com/) and pull all required models. We use `deepseek-r1:8b` as our powerful "teacher" model for data generation.
    ```bash
    # Pull the models you will fine-tune
    ollama pull deepseek-r1:8b
    ollama pull gemma3:4b-it-q4_K_M

    # Ensure the "teacher" model for prepare_dataset.py is also pulled
    # (In this case, it's the same as one of our training models)
    ```

### Step 2: Data Generation

This command creates both `training_data_st1_scot.jsonl` and `training_data_st2_scot.jsonl`. The `TEACHER_MODEL` inside `prepare_dataset.py` is set to `deepseek-r1:8b`.
```bash
python prepare_dataset.py
```

### Step 3: Training

Train a separate model for each subtask. It is recommended to create a separate experiment for each base model.

#### Training DeepSeek
```bash
# Train DeepSeek for Subtask 1
python train.py \
    --model_name "deepseek-r1:8b" \
    --dataset_path "training_data_st1_scot.jsonl" \
    --output_dir "./deepseek-finetuned-st1"

# Train DeepSeek for Subtask 2
python train.py \
    --model_name "deepseek-r1:8b" \
    --dataset_path "training_data_st2_scot.jsonl" \
    --output_dir "./deepseek-finetuned-st2"
```

#### Training Gemma
```bash
# Train Gemma for Subtask 1
python train.py \
    --model_name "gemma3:4b-it-q4_K_M" \
    --dataset_path "training_data_st1_scot.jsonl" \
    --output_dir "./gemma-finetuned-st1"

# Train Gemma for Subtask 2
python train.py \
    --model_name "gemma3:4b-it-q4_K_M" \
    --dataset_path "training_data_st2_scot.jsonl" \
    --output_dir "./gemma-finetuned-st2"
```

### Step 4: Evaluation

Use the correct evaluation script and model paths.

#### Evaluating DeepSeek
```bash
# Evaluate the DeepSeek Subtask 1 model
python evaluate_subtask1.py \
    --base_model_name "deepseek-r1:8b" \
    --adapter_path "./deepseek-finetuned-st1" \
    --test_data_path "pilot_dataset.json"

# Evaluate the DeepSeek Subtask 2 model
python evaluate_subtask2.py \
    --base_model_name "deepseek-r1:8b" \
    --adapter_path "./deepseek-finetuned-st2" \
    --test_data_path "training_data_st2_scot.jsonl"
```

#### Evaluating Gemma
```bash
# Evaluate the Gemma Subtask 1 model
python evaluate_subtask1.py \
    --base_model_name "gemma3:4b-it-q4_K_M" \
    --adapter_path "./gemma-finetuned-st1" \
    --test_data_path "pilot_dataset.json"

# Evaluate the Gemma Subtask 2 model
python evaluate_subtask2.py \
    --base_model_name "gemma3:4b-it-q4_K_M" \
    --adapter_path "./gemma-finetuned-st2" \
    --test_data_path "training_data_st2_scot.jsonl"
```

### Step 5: The Improvement Loop

Analyze the `evaluation_results_st1.json` or `evaluation_results_st2.json` to find patterns in your model's mistakes.
```bash
python analyze_errors.py --results_file evaluation_results_st1.json
```
Use these insights to guide your next steps. Compare the performance of DeepSeek and Gemma. Which one has a lower content effect? Which is more accurate? Double down on the more promising model.

### Step 6: Final Submission

When you have identified your best-performing model and the official test data is released, generate your submission file.

#### Example using your best DeepSeek model for Subtask 1
```bash
python generate_submission.py \
    --base_model_name "deepseek-r1:8b" \
    --adapter_path "./deepseek-finetuned-st1" \
    --test_data_path "official_unlabeled_test_data.jsonl" \
    --output_file "submission.jsonl"
```

#### Example using your best Gemma model for Subtask 1
```bash
python generate_submission.py \
    --base_model_name "gemma3:4b-it-q4_K_M" \
    --adapter_path "./gemma-finetuned-st1" \
    --test_data_path "official_unlabeled_test_data.jsonl" \
    --output_file "submission.jsonl"
```