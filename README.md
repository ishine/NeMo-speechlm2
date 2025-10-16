# NeMo SpeechLM2: SALM Training Framework

A research framework for training Speech-Augmented Language Models (SALM) using NVIDIA NeMo toolkit, focusing on LLM-based Automatic Speech Recognition (ASR) with Canary-Qwen-2.5B architecture.

## Overview

This framework provides production-ready implementations for training SALM models that combine pretrained LLM (Qwen) with ASR encoders (Canary) for state-of-the-art speech recognition. Key features include:

- **SALM Architecture**: Audio embeddings → LLM → Text generation
- **Efficient Training**: LoRA fine-tuning, gradient accumulation, mixed precision
- **Advanced Logging**: Token-aware prediction logging with audio frame counts
- **Flexible I/O**: Checkpoint (.ckpt) and NeMo (.nemo) format support
- **Lhotse Integration**: Bucketing, multimodal sampling, shar format

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SALM Model Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Audio Input (16kHz)                                        │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────┐                                       │
│  │ ASR Encoder     │ (Canary/Parakeet)                     │
│  │ (Frozen)        │ → Audio Embeddings (1024-dim)         │
│  └─────────────────┘                                       │
│       │                                                      │
│       ▼                                                      │
│  ┌─────────────────┐                                       │
│  │ Modality        │ (Conformer Adapter)                   │
│  │ Adapter         │ → Adapted Audio Embeddings            │
│  │ (Trainable)     │                                       │
│  └─────────────────┘                                       │
│       │                                                      │
│       ├──────────────────────┐                             │
│       │                      │                             │
│       ▼                      ▼                             │
│  Text Tokens          Audio Embeddings                     │
│       │                      │                             │
│       └──────────┬───────────┘                             │
│                  │                                          │
│                  ▼                                          │
│  ┌─────────────────────────────────┐                      │
│  │ Qwen LLM (Frozen/LoRA)          │                      │
│  │ • Base LLM: Frozen              │                      │
│  │ • LoRA Adapters: Trainable      │                      │
│  │ • Input: [text_emb + audio_emb] │                      │
│  └─────────────────────────────────┘                      │
│                  │                                          │
│                  ▼                                          │
│            Text Output (ASR Transcription)                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Key Components:
- ASR Encoder: Pretrained frozen (Canary-1B, Parakeet-TDT-0.6B)
- Modality Adapter: Trainable Conformer layers (2-4 layers)
- LLM: Pretrained frozen with LoRA adapters (q_proj, v_proj)
- Training: Only adapter + LoRA layers (~1-5% parameters)
```

## Environment Setup

**Tested Configuration:**
```
PyTorch: 2.6.0.dev20241112+cu121
NeMo Toolkit: 2.5.0rc1
CUDA: 12.1
Python: 3.10+
```

**Installation:**
```bash
# Clone repository
git clone <repository-url> NeMo-speechlm2
cd NeMo-speechlm2

# Install NeMo with editable mode
pip install -e .

# Install additional dependencies for speech
pip install -e ".[asr]"
```

## Quick Start

### 1. Data Preparation

Prepare your datasets in **lhotse shar format**:

```python
# Example: Create lhotse cuts and convert to shar format
from lhotse import CutSet, RecordingSet, SupervisionSet
from lhotse.shar import SharWriter

# Create cuts from your audio + transcripts
recordings = RecordingSet.from_dir("/path/to/audio")
supervisions = SupervisionSet.from_segments([...])
cuts = CutSet.from_manifests(recordings, supervisions)

# Write to shar format
with SharWriter("/path/to/output_shar") as writer:
    for cut in cuts:
        writer.write(cut)
```

### 2. Training

**Configuration File:** `recipes/CanaryQwenASR/configs/salm_canary_qwen_2.5b.yaml`

**Key Settings to Modify:**
```yaml
model:
  pretrained_llm: /path/to/Qwen3-1.7B/  # Your Qwen model path
  pretrained_asr: /path/to/pretrained_asr.nemo  # Your ASR encoder

  # Prediction logging (new feature!)
  log_prediction_train: true
  log_prediction_train_samples: 5      # Log 5 samples every N steps
  log_prediction_train_interval: 50    # Log every 50 steps
  log_prediction_valid: true
  log_prediction_valid_samples: 20     # Log 20 samples per validation

data:
  train_ds:
    shar_path:
      - [/path/to/train_shar, 1.0]  # [path, sampling_weight]

  validation_ds:
    datasets:
      val_set:
        manifest_filepath: /path/to/val_manifest.json
```

**Run Training:**
```bash
# Option 1: Using shell script (recommended)
bash recipes/CanaryQwenASR/train.sh

# Option 2: Direct Python execution
python recipes/CanaryQwenASR/speech_to_text_salm.py \
    --config-path=recipes/CanaryQwenASR/configs \
    --config-name=salm_canary_qwen_2.5b \
    exp_manager.exp_dir=./outputs \
    trainer.max_steps=50000

# Option 3: With Hydra overrides
python recipes/CanaryQwenASR/speech_to_text_salm.py \
    --config-path=recipes/CanaryQwenASR/configs \
    --config-name=salm_canary_qwen_2.5b \
    model.pretrained_llm=/path/to/qwen \
    model.lora.r=128 \
    trainer.max_steps=100000 \
    exp_manager.exp_dir=./my_experiment
```

### 3. Enhanced Prediction Logging

The framework includes advanced prediction logging with token counts:

**Example Training Log Output:**
```
====================================================================================================
Training Sample 1 | WER: 8.33%
Full LLM Input : <|im_start|>user Transcribe: <|audio|> <|im_start|>assistant 롯데카드... (67)
Prompt         : <|im_start|>user Transcribe the following: (10)
Audio Token    : <|audio|> → [audio_embeddings] (25)
Ground Truth   : <|im_start|>assistant 롯데카드 0056 8380 7809 9195로... (32)
Prediction     : <|im_start|>assistant 롯데카드 0056 8380 7809 9195로... (32)
====================================================================================================
```

Features:
- **Aligned format**: Clean vertical alignment for readability
- **Token counts**: Shows actual tokenizer counts for text, estimated frames for audio
- **Full LLM Input**: Complete input structure including audio placeholder location
- **Audio Token Line**: Shows `<|audio|>` → `[audio_embeddings]` transformation

### 4. Model Checkpointing & Saving

**Automatic .nemo Saving (Recommended):**
```yaml
exp_manager:
  checkpoint_callback_params:
    always_save_nemo: true          # Auto-save .nemo on each checkpoint
    save_nemo_on_train_end: true    # Save final .nemo at training end
    save_top_k: 20                   # Keep top 20 checkpoints by WER
```

**Manual Conversion (.ckpt → .nemo):**
```python
from nemo.collections.speechlm2 import SALM

# Load from checkpoint
model = SALM.load_from_checkpoint("checkpoint.ckpt")

# Save to .nemo format (self-contained, includes config + weights)
model.save_to("model.nemo")
```

**Format Comparison:**
- **`.ckpt`**: PyTorch Lightning checkpoint (for training resumption)
- **`.nemo`**: NeMo format (for deployment, inference, offline usage)

### 5. Evaluation

**Using Built-in Evaluation Script:**
```bash
# Evaluate from .nemo file (recommended for deployment)
python examples/speechlm2/salm_eval.py \
    checkpoint_path=model.nemo \
    data.test_ds.datasets.test_set.manifest_filepath=/path/to/test.json

# Evaluate from .ckpt file (for training checkpoints)
python examples/speechlm2/salm_eval.py \
    checkpoint_path=checkpoint.ckpt \
    data.test_ds.datasets.test_set.manifest_filepath=/path/to/test.json

# Evaluate from HuggingFace model ID
python examples/speechlm2/salm_eval.py \
    checkpoint_path=nvidia/canary-qwen-2.5b \
    data.test_ds.datasets.test_set.manifest_filepath=/path/to/test.json
```

**Evaluation Metrics:**
- **WER (Word Error Rate)**: Primary ASR metric
- **Token-level Accuracy**: Per-token prediction accuracy
- **Loss**: Validation loss

### 6. Inference & Generation

**Generate Transcriptions:**
```python
from nemo.collections.speechlm2 import SALM

# Load model (supports .nemo, .ckpt, or HuggingFace ID)
model = SALM.from_pretrained("model.nemo")
model = model.cuda().eval()

# Generate transcription
prompts = [
    [{"role": "user",
      "content": "Transcribe the following: <|audio|>",
      "audio": ["/path/to/audio.wav"]}]
]

answer_ids = model.generate(
    prompts=prompts,
    max_new_tokens=128,
    do_sample=False
)

# Decode
transcription = model.tokenizer.ids_to_text(answer_ids[0].tolist())
print(transcription)
```

**Using Generation Script:**
```bash
python examples/speechlm2/salm_generate.py \
    checkpoint_path=model.nemo \
    audio_paths="['/path/to/audio1.wav', '/path/to/audio2.wav']" \
    max_new_tokens=128
```

## Training Configuration Guide

### Key Hyperparameters

**Model Configuration:**
```yaml
model:
  # Pretrained model paths
  pretrained_llm: /path/to/Qwen3-1.7B/
  pretrained_asr: /path/to/encoder.nemo
  pretrained_weights: true  # Load pretrained weights (false for random init)

  # LoRA settings (parameter-efficient fine-tuning)
  lora:
    r: 128              # Rank (higher = more parameters, better quality)
    lora_alpha: 256     # Scaling factor (usually 2x rank)
    target_modules: ["q_proj", "v_proj"]  # Which LLM layers to adapt

  # Freezing strategy (freeze LLM, train only adapters)
  freeze_params:
    - "^llm\\..+$"              # Freeze entire LLM
    - "^embed_tokens\\..+$"     # Freeze embeddings
  prevent_freeze_params:
    - ".*lora_.*"               # Keep LoRA trainable
```

**Training Configuration:**
```yaml
trainer:
  devices: -1          # Use all available GPUs
  num_nodes: -1        # Auto-detect via MPI
  precision: bf16-true # Mixed precision (bf16 recommended for A100)

  max_steps: 100000    # Total training steps
  val_check_interval: 400  # Validate every 400 steps
  accumulate_grad_batches: 4  # Gradient accumulation (effective_bs = bs * accum)

  strategy:
    _target_: lightning.pytorch.strategies.DDPStrategy
    find_unused_parameters: true  # Required for frozen parameters
```

**Data Configuration:**
```yaml
data:
  train_ds:
    # Bucketing for efficient batching by audio duration
    use_bucketing: true
    num_buckets: 16
    bucket_duration_bins: [99, 110, 117, ..., 1024]  # Duration buckets (tokens)
    bucket_batch_size: [69, 64, 60, ..., 2]          # Batch size per bucket

    # Audio constraints
    min_duration: 0.3   # Minimum audio length (seconds)
    max_duration: 40.0  # Maximum audio length (seconds)

    # Token constraints
    min_tokens: 2
    max_tokens: 1024    # Maximum sequence length
```

### Distributed Training

**Multi-GPU (Single Node):**
```bash
# Automatically uses all available GPUs
python recipes/CanaryQwenASR/speech_to_text_salm.py \
    --config-path=recipes/CanaryQwenASR/configs \
    --config-name=salm_canary_qwen_2.5b
```

**Multi-Node (MPI):**
```bash
# Example: 4 nodes, 8 GPUs each (32 total GPUs)
mpirun -np 32 \
    --npernode 8 \
    --bind-to none \
    python recipes/CanaryQwenASR/speech_to_text_salm.py \
    --config-path=recipes/CanaryQwenASR/configs \
    --config-name=salm_canary_qwen_2.5b \
    trainer.num_nodes=4 \
    trainer.devices=8
```

## Repository Structure

```
NeMo-speechlm2/
├── nemo/
│   ├── collections/
│   │   ├── speechlm2/           # SALM implementations
│   │   │   ├── models/
│   │   │   │   └── salm.py      # Core SALM model (enhanced logging)
│   │   │   ├── data/
│   │   │   │   └── salm_dataset.py  # Lhotse dataset loader
│   │   │   ├── modules/
│   │   │   │   └── perception.py    # Audio encoder + adapter
│   │   │   └── parts/
│   │   │       ├── pretrained.py    # Model loading utilities
│   │   │       └── save_nemo_callback.py  # Auto .nemo saving
│   │   ├── asr/                 # ASR utilities
│   │   └── common/              # Shared components
│   │       ├── prompts/
│   │       │   └── canary_qwen.py   # Prompt formatter
│   │       └── tokenizers/
│   └── lightning/               # PyTorch Lightning integration
│
├── recipes/
│   └── CanaryQwenASR/          # Training recipe
│       ├── configs/
│       │   └── salm_canary_qwen_2.5b.yaml  # Main config
│       ├── speech_to_text_salm.py          # Training script
│       └── train.sh                         # Execution script
│
├── examples/
│   └── speechlm2/              # Official examples
│       ├── salm_train.py       # Basic training example
│       ├── salm_eval.py        # Evaluation script
│       ├── salm_generate.py    # Inference script
│       └── to_hf.py            # HuggingFace export
│
└── README.md                    # This file
```

## Advanced Features

### 1. SaveNemoCallback - Automatic .nemo Conversion

Automatically converts checkpoints to .nemo format during training:

```yaml
exp_manager:
  checkpoint_callback_params:
    always_save_nemo: true       # Enable auto-conversion
    save_top_k: 20                # Save top 20 by WER
```

**Benefits:**
- Self-contained: Includes model config + weights
- Deployment-ready: Can be loaded without training code
- Offline inference: No HuggingFace/internet dependency

### 2. Enhanced Prediction Logging

Token-aware logging for debugging and analysis:

```yaml
model:
  # Training logging
  log_prediction_train: true
  log_prediction_train_samples: 5
  log_prediction_train_interval: 50

  # Validation logging
  log_prediction_valid: true
  log_prediction_valid_samples: 20
```

**Features:**
- Aligned formatting for readability
- Token counts (text tokens + audio frames)
- Full LLM input structure visualization
- Audio embedding frame counts

### 3. Flexible Model Loading

The `from_pretrained` method supports multiple sources:

```python
from nemo.collections.speechlm2 import SALM

# From .nemo file
model = SALM.from_pretrained("model.nemo")

# From .ckpt file
model = SALM.from_pretrained("checkpoint.ckpt")

# From HuggingFace
model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")

# From local HF directory
model = SALM.from_pretrained("/path/to/hf_model/")
```

### 4. Custom Prompt Formats

Add custom prompt formatters for different LLM architectures:

```python
# In nemo/collections/common/prompts/
from nemo.collections.common.prompts.formatter import PromptFormatter

class MyPromptFormatter(PromptFormatter):
    NAME = "my_format"
    # ... implement template
```

Then use in config:
```yaml
model:
  prompt_format: my_format
```

## Performance Optimization

### Memory Optimization for 80GB A100

```yaml
# Balanced configuration for 8x A100 80GB
trainer:
  devices: 8
  precision: bf16-true
  accumulate_grad_batches: 4

model:
  lora:
    r: 128  # Higher rank = more memory but better quality

data:
  train_ds:
    bucket_batch_size: [69, 64, 60, 40, 28, 22, 16, 14, 12, 11, 10, 6, 4, 4, 4, 2]
```

### Training Speed

**Expected throughput (8x A100 80GB):**
- ~400-600 tokens/sec per GPU
- ~3200-4800 tokens/sec total
- ~100k steps in 24-36 hours

**Optimization tips:**
1. Use `bf16-true` precision (faster than fp16 on A100)
2. Enable `foreach: true` in optimizer (faster updates)
3. Use bucketing to maximize GPU utilization
4. Tune `num_workers` for data loading (typically 8-16)

## Troubleshooting

### Common Issues

**Issue: OOM (Out of Memory)**
```yaml
# Solution: Reduce batch sizes or increase accumulation
data:
  train_ds:
    bucket_batch_size: [34, 32, 30, 20, 14, 11, 8, 7, 6, 5, 5, 3, 2, 2, 2, 1]
trainer:
  accumulate_grad_batches: 8  # Increase from 4
```

**Issue: No audio placeholders found**
```
# Check prompt format in data loader
data:
  train_ds:
    prompt_format: ${model.prompt_format}  # Must match model
    asr_context_prompt: "Transcribe the following: "  # Must be set
```

**Issue: Training loss not decreasing**
```yaml
# Check if LoRA is enabled and modules are trainable
model:
  freeze_params:
    - "^llm\\..+$"
  prevent_freeze_params:
    - ".*lora_.*"  # Ensure LoRA is NOT frozen
  lora:
    inference_mode: false  # Must be false for training
```

**Issue: Checkpoint resumption fails**
```yaml
# Ensure perception config is saved in checkpoint
# This is handled automatically by on_save_checkpoint hook
# If using old checkpoints, may need to retrain
```

## Citation & References

**SALM Model Architecture:**
```bibtex
@misc{nvidia2024canary,
  title={Canary: Multilingual ASR and Speech Translation},
  author={NVIDIA},
  year={2024},
  url={https://huggingface.co/nvidia/canary-qwen-2.5b}
}
```

**NeMo Toolkit:**
```bibtex
@misc{nemo2024,
  title={NVIDIA NeMo},
  author={NVIDIA},
  year={2024},
  url={https://github.com/NVIDIA/NeMo}
}
```

## License

This project is based on NVIDIA NeMo and follows the Apache License 2.0.

## Contributing

Contributions are welcome! Please ensure:
1. Code passes style checks: `python setup.py style --scope . --fix`
2. Tests pass: `pytest tests/collections/speechlm2/`
3. Commits are signed: `git commit -s -m "message"`

## Support

For issues and questions:
- NeMo Documentation: https://docs.nvidia.com/nemo-framework/
- NeMo GitHub Issues: https://github.com/NVIDIA/NeMo/issues
- SALM Paper/Model: https://huggingface.co/nvidia/canary-qwen-2.5b
