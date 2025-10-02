# SLM-Trainer

A research framework for developing Speech Language Models (SLM) with a focus on LLM-based Automatic Speech Recognition (ASR) systems using NVIDIA NeMo toolkit.

## Overview

SLM-Trainer provides optimized training recipes and implementations for speech language models, particularly leveraging the NeMo speechlm2 framework. The project emphasizes efficient, scalable ASR model development using state-of-the-art pretrained language models.

## Environment

**Tested Configuration:**
- PyTorch: `2.6.0.dev20241112+cu121`
- NeMo Toolkit: `2.5.0rc1`
- CUDA: 12.1
- Python: 3.10+

## Repository Structure

```
slm-trainer/
├── nemo/                          # NeMo toolkit core modules
│   ├── collections/
│   │   ├── speechlm2/            # Speech LLM implementations
│   │   ├── asr/                  # ASR models and utilities
│   │   └── common/               # Shared components
│   └── ...
├── recipes/                       # Training recipes (custom implementations)
│   ├── CanaryQwenASR/            # Canary-Qwen SALM ASR recipe
│   │   ├── configs/              # Model and training configurations
│   │   ├── speech_to_text_salm.py  # Training script
│   │   └── train.sh              # Execution script
│   └── ...
├── examples/                      # NeMo official examples
└── README.md
```

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url> slm-trainer
cd slm-trainer

# Install dependencies
pip install -e .
```

### Training

#### SALM Canary-Qwen ASR Model

```bash
# Navigate to project root
cd slm-trainer

# Run training
bash recipes/CanaryQwenASR/train.sh

# With custom overrides
bash recipes/CanaryQwenASR/train.sh \
    trainer.max_steps=50000 \
    data.train_ds.batch_tokens=4000
```

#### Direct Python Execution

```bash
python recipes/CanaryQwenASR/speech_to_text_salm.py \
    --config-path=recipes/CanaryQwenASR/configs \
    --config-name=salm_canary_qwen_2.5b \
    exp_manager.exp_dir=./outputs
```

## Configuration

### Model Configuration

Key configuration parameters in `recipes/CanaryQwenASR/configs/salm_canary_qwen_2.5b.yaml`:

```yaml
model:
  pretrained_llm: /path/to/Qwen3-1.7B/
  pretrained_asr: /path/to/encoder.nemo
  prompt_format: canary_qwen

  lora:
    r: 16
    lora_alpha: 32
    target_modules: [q_proj, v_proj, k_proj, ...]

data:
  train_ds:
    batch_tokens: 3800
    max_tokens: 3072
    use_bucketing: true

trainer:
  precision: bf16-true
  max_steps: 90000
```

### Data Format

This framework uses Lhotse shar format for all datasets. Prepare your data as follows:

```python
# Dataset configuration in YAML
data:
  train_ds:
    shar_path:
      - [/path/to/dataset1, weight1]
      - [/path/to/dataset2, weight2]
```

## Features

- **SALM Architecture**: Speech-Augmented Language Models combining pretrained LLM with ASR encoders
- **Lhotse Integration**: Efficient data loading with bucketing and multimodal sampling
- **LoRA Fine-tuning**: Parameter-efficient adaptation of large language models
- **Distributed Training**: Multi-GPU and multi-node support via MPI/DDP
- **Memory Optimization**: Configurable batch sizes and gradient accumulation for 80GB A100 GPUs

## Development

### Adding New Recipes

1. Create recipe directory: `recipes/YourRecipe/`
2. Add configuration: `recipes/YourRecipe/configs/config.yaml`
3. Implement training script: `recipes/YourRecipe/train.py`
4. Add execution script: `recipes/YourRecipe/train.sh`

## Reference

Built upon [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) framework and inspired by the [Canary-Qwen-2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b) model architecture.
