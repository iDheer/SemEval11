#!/bin/bash

# --- Ollama-based Experiment Configuration ---
# STUDENT MODEL: Gets fine-tuned with different few-shot examples
MODEL_NAME="gemma3:4b-it-q4_K_M"
# MODEL_NAME="deepseek-r1:8b"  # Uncomment to experiment with deepseek

DATASET_PATH="training_data_st1_scot.jsonl" # Change to _st2 for subtask 2
TEST_DATA_PATH="pilot_dataset.json"
LOG_FILE="experiment_log.txt"

# --- Experiment Variations ---
DATASETS=("training_data_st1_scot.jsonl" "training_data_st2_scot.jsonl")
TEMPERATURES=(0.1 0.3)

# --- Start Sweep ---
echo "Starting Ollama experiment sweep for $MODEL_NAME at $(date)" > $LOG_FILE

for dataset in "${DATASETS[@]}"; do
  for temp in "${TEMPERATURES[@]}"; do
    
    SUBTASK=$(echo $dataset | grep -o "st[12]")
    MODEL_ID_TAG=$(echo $MODEL_NAME | cut -d':' -f1)
    OUTPUT_DIR="./experiments/${MODEL_ID_TAG}_${SUBTASK}_temp${temp}"
    
    echo "--------------------------------------------------" | tee -a $LOG_FILE
    echo "Running experiment: ${OUTPUT_DIR}" | tee -a $LOG_FILE
    echo "Dataset: $dataset, Temperature: $temp" | tee -a $LOG_FILE
    echo "--------------------------------------------------" | tee -a $LOG_FILE

    # 1. Create fine-tuned model
    python train.py \
      --model_name "$MODEL_NAME" \
      --dataset_path "$dataset" \
      --output_dir "$OUTPUT_DIR"

    if [ $? -ne 0 ]; then
      echo "Training failed for ${OUTPUT_DIR}" | tee -a $LOG_FILE
      continue
    fi

    # 2. Get the created model name
    MODEL_INFO_FILE="${OUTPUT_DIR}/model_info.json"
    if [ -f "$MODEL_INFO_FILE" ]; then
      FINETUNED_MODEL=$(python -c "import json; print(json.load(open('$MODEL_INFO_FILE'))['finetuned_model_name'])" 2>/dev/null)
      
      if [ ! -z "$FINETUNED_MODEL" ]; then
        echo "Evaluating with model: $FINETUNED_MODEL" | tee -a $LOG_FILE
        
        # 3. Evaluate the model
        if [[ $dataset == *"st1"* ]]; then
          python evaluate_subtask1.py \
            --model_name "$FINETUNED_MODEL" \
            --test_data_path "$TEST_DATA_PATH" >> $LOG_FILE 2>&1
        else
          python evaluate_subtask2.py \
            --model_name "$FINETUNED_MODEL" \
            --test_data_path "$dataset" >> $LOG_FILE 2>&1
        fi
      else
        echo "Could not find fine-tuned model name" | tee -a $LOG_FILE
      fi
    else
      echo "Model info file not found: $MODEL_INFO_FILE" | tee -a $LOG_FILE
    fi
        
  done
done

echo "Experiment sweep finished at $(date)" | tee -a $LOG_FILE
        echo "Using fine-tuned model: $FINETUNED_MODEL" >> $LOG_FILE
        
        python evaluate_subtask1.py \
          --model_name "$FINETUNED_MODEL" \
          --test_data_path "$TEST_DATA_PATH" >> $LOG_FILE
      else
        echo "Model info file not found, using base model for evaluation" >> $LOG_FILE
        python evaluate_subtask1.py \
          --model_name "$MODEL_NAME" \
          --test_data_path "$TEST_DATA_PATH" >> $LOG_FILE
      fi
        
    done
  done
done

echo "Hyperparameter sweep finished at $(date)" | tee -a $LOG_FILE