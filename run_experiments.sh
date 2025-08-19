#!/bin/bash

# --- Experiment Configuration ---
# CHOOSE YOUR MODEL: Uncomment the line for the model you want to sweep.
MODEL_NAME="deepseek-r1:8b"
# MODEL_NAME="gemma3:4b-it-q4_K_M"

DATASET_PATH="training_data_st1_scot.jsonl" # Change to _st2 for subtask 2
TEST_DATA_PATH="pilot_dataset.json"
LOG_FILE="experiment_log.txt"

# --- Hyperparameters to Sweep ---
LEARNING_RATES=(1e-4 2e-4)
LORA_RANKS=(16 32)
NUM_EPOCHS=(3)

# --- Start Sweep ---
echo "Starting hyperparameter sweep for $MODEL_NAME at $(date)" > $LOG_FILE

for lr in "${LEARNING_RATES[@]}"; do
  for rank in "${LORA_RANKS[@]}"; do
    for epochs in "${NUM_EPOCHS[@]}"; do
      
      MODEL_ID_TAG=$(echo $MODEL_NAME | cut -d':' -f1) # a tag like "deepseek-r1" or "gemma3"
      OUTPUT_DIR="./experiments/${MODEL_ID_TAG}_lr${lr}_r${rank}_e${epochs}"
      
      echo "--------------------------------------------------" | tee -a $LOG_FILE
      echo "Running experiment: ${OUTPUT_DIR}" | tee -a $LOG_FILE
      echo "--------------------------------------------------" | tee -a $LOG_FILE

      # 1. Train the model
      python train.py \
        --model_name "$MODEL_NAME" \
        --dataset_path "$DATASET_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --epochs "$epochs" # Add other params like --learning_rate if you modify train.py

      if [ $? -ne 0 ]; then
        echo "Training failed for ${OUTPUT_DIR}" | tee -a $LOG_FILE
        continue
      fi

      # 2. Evaluate the model
      echo "\n--- Evaluation Results for ${OUTPUT_DIR} ---" >> $LOG_FILE
      python evaluate_subtask1.py \
        --base_model_name "$MODEL_NAME" \
        --adapter_path "$OUTPUT_DIR" \
        --test_data_path "$TEST_DATA_PATH" >> $LOG_FILE
        
    done
  done
done

echo "Hyperparameter sweep finished at $(date)" | tee -a $LOG_FILE