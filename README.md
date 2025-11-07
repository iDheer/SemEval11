Weights:-https://drive.google.com/drive/folders/1zmXdUrSL6SgIN7QiJi67h181xEtg6Jom?usp=sharing

# Syllogistic Reasoning with SDU (Selective Disentanglement Unlearning)

## Project Overview
Implementation of Selective Disentanglement Unlearning (SDU) for reducing plausibility bias in syllogistic reasoning tasks.


## Models
- **Base Model**: Qwen/Qwen3-8B (4-bit quantized)
- **Fine-tuning**: LoRA adapters (r=8, α=16)

## Pipeline

### Stage 0: S-CoT Baseline Training
```bash
python train_stage0_scot_fixed.py
```
**Output**: `scot_base_model/` (Baseline model)
**Time**: ~30 minutes

### Stage 1: Shadow Model + Saliency Calculation
```bash
python train_sdu_stage1_fixed.py
```
**Output**: 
- `shadow_model_qwen_final/` (Shadow model)
- `saliency_masks.pt` (Weight importance masks)
**Time**: ~2.5 hours

### Stage 2: SDU Application
```bash
python train_sdu_stage2_fixed.py
```
**Output**: 
- `sdu_model_qwen_final/` (Final SDU model)
- `delta_z_bias_direction.pt` (Activation steering vectors)
**Time**: ~15 minutes

## Evaluation

### Baseline Evaluation
```bash
python evaluate_scot_baseline_expanded.py
```
**Output**: `scot_baseline_results_expanded.json`

### SDU Evaluation
```bash
python evaluate_final_expanded.py
```
**Output**: `sdu_final_results_expanded.json`


## Requirements
See `requirements.txt`

## Hardware
- GPU: NVIDIA RTX 4060 (8GB VRAM)
- RAM: 16GB+
- Storage: ~20GB for models

## Contact
[Your Name]
[Your Email]
