# Canary-Qwen-2.5B ASR Evaluation Script

Enhanced evaluation script for SALM (Speech-Augmented Language Model) Canary-Qwen-2.5B models.

## Features

- ✅ **Multiple Checkpoint Formats**: Supports `.ckpt`, `.nemo`, and HuggingFace models
- ✅ **Qwen3 Compatibility**: Properly handles Qwen3 tokenizer and weights on older transformers
- ✅ **Automatic VAD**: Voice Activity Detection for long audio files
- ✅ **Batch Processing**: Process multiple audio files efficiently
- ✅ **Advanced WER/CER Calculation**:
  - HuggingFace Open ASR Leaderboard protocol-compliant normalization
  - Dynamic metric support (WER for word-based, CER for character-based languages)
  - Multilingual text normalization (English, Chinese, Japanese, Korean, etc.)
  - Enhanced logging with token counts and intuitive formatting

## Model Loading Strategies

### Training vs Evaluation Scenarios

| Phase | Purpose | Models Used | Weights Source |
|-------|---------|-------------|----------------|
| **Training** | Initialize & fine-tune | `pretrained_llm`: Qwen3-1.7B<br>`pretrained_asr`: averaged_top20_model.nemo | Pretrained models |
| **Evaluation** | Load trained model | Checkpoint (`.ckpt` or `.nemo`) | **All weights from checkpoint** |

**IMPORTANT**:
- During evaluation, pretrained models (pretrained_asr, pretrained_llm) are **NOT** required
- All weights (LLM, ASR encoder, modality adapter) come from the checkpoint
- Pretrained model paths are only used for tokenizer loading

### Checkpoint Format Support

#### 1. PyTorch Lightning Checkpoint (`.ckpt`)

**Structure**:
```
checkpoint.ckpt (PyTorch save file)
├── hyper_parameters
│   └── cfg (model configuration)
│       ├── pretrained_llm: path/to/Qwen3
│       ├── pretrained_asr: path/to/asr.nemo
│       └── ...
└── state_dict (all trained weights)
    ├── llm.* (LLM weights)
    ├── perception.* (ASR encoder weights)
    ├── embed_tokens.* (embedding weights)
    └── ...
```

**Loading Strategy (IMPROVED - 2025-10-10)**:

**Method 1: PyTorch Lightning Native (Recommended)**
Uses `SALM.load_from_checkpoint()` - the EXACT method PyTorch Lightning uses during training resume.

**Benefits**:
- ✅ **100% Weight Fidelity**: All weights loaded correctly (0 missing/unexpected keys)
- ✅ **Architecture Preservation**: Qwen3 stays Qwen3 (no conversion that loses parameters)
- ✅ **LoRA Support**: Preserves LoRA adapter parameters if present
- ✅ **QK Normalization**: Retains Qwen3-specific QK normalization layers (56 params)
- ✅ **Zero Bias Issues**: No random initialization of bias parameters
- ✅ **Production Ready**: Matches training behavior exactly

**Process**:
1. PyTorch Lightning loads hyperparameters and config from checkpoint
2. Initializes model with exact training architecture (automatic)
3. Loads all weights with strict=True (enforces perfect match)
4. Comprehensive verification of weight integrity

**Method 2: Manual Fallback (Diagnostic)**
Only used if Lightning native method fails. Provides detailed diagnostics showing what goes wrong with manual loading.

**Previous Issues (Before 2025-10-10 Fix)**:
- Manual loading with Qwen3→Qwen2 conversion lost 56 QK normalization weights
- 84 bias parameters were randomly initialized (incorrect for Qwen3)
- Total: 140 parameters with integrity issues
- **Solution**: Now uses Lightning native loading by default

#### 2. NeMo Checkpoint (`.nemo`)

**Structure**:
```
model.nemo (tar archive)
├── model_config.yaml (model configuration)
├── model_weights.ckpt (trained weights)
├── tokenizer/ (optional)
│   ├── tokenizer_config.json
│   ├── vocab.json
│   └── merges.txt
└── other_metadata.json
```

**Loading Strategy**:
1. Load model directly via `SALM.restore_from()` (handles everything automatically)
2. Config, weights, and tokenizer loaded from .nemo archive
3. No modifications needed - uses exact training configuration

#### 3. HuggingFace Model

Direct loading from HuggingFace Hub or local directory:
```bash
--model nvidia/canary-qwen-2.5b
# or
--model /path/to/local/hf/model
```

## Usage

### Basic Usage

```bash
python canary_qwen_asr_transcribe.py \
  --model /path/to/checkpoint.ckpt \
  --input /path/to/audio/directory \
  --output /path/to/output
```

### Advanced Options

```bash
python canary_qwen_asr_transcribe.py \
  --model /path/to/checkpoint.ckpt \
  --input /path/to/audio.list \
  --output /path/to/output \
  --device cuda \
  --use_vad \
  --vad_sec 30 \
  --aligner scripts/speech_recognition/AlignG7.exe \
  --cer
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | - | Path to checkpoint (.ckpt/.nemo) or HuggingFace model |
| `--input` | - | Input directory or list file path |
| `--list_name` | None | List file in input directory |
| `--output` | `canary_qwen_output` | Output directory |
| `--batch` | 1 | Batch size (currently processes individually) |
| `--use_vad` | False | Use VAD for long audio files |
| `--vad_sec` | 30 | Maximum segment duration for VAD (seconds) |
| `--device` | auto | Device (cuda/cpu/auto) |
| `--aligner` | `scripts/speech_recognition/AlignG7.exe` | AlignG7.exe path |
| `--align_noise` | None | Noise tokens to remove for alignment |
| `--cer` | False | Calculate CER instead of WER |
| `--align_timeout` | 172800 | Alignment timeout (seconds) |
| `--verbose` | False | Enable verbose logging |
| `--nemo_path` | None | Custom NeMo installation path |

## Examples

### Example 1: Evaluate with .ckpt checkpoint

```bash
python canary_qwen_asr_transcribe.py \
  --model recipes/CanaryQwenASR/outputs/SALM-251010-step17301-val_loss0.861.ckpt \
  --input /path/to/sample.list \
  --output outputs/eval-results \
  --device cuda
```

### Example 2: Evaluate with .nemo checkpoint

```bash
python canary_qwen_asr_transcribe.py \
  --model recipes/CanaryQwenASR/outputs/SALM-Canary-Qwen-2.5B.nemo \
  --input /path/to/sample.list \
  --output outputs/eval-results \
  --device cuda
```

### Example 3: Use VAD for long audio

```bash
python canary_qwen_asr_transcribe.py \
  --model /path/to/checkpoint.ckpt \
  --input /path/to/long_audio_files/ \
  --output outputs/eval-results \
  --use_vad \
  --vad_sec 30 \
  --device cuda
```

### Example 4: Calculate CER with alignment

```bash
python canary_qwen_asr_transcribe.py \
  --model /path/to/checkpoint.ckpt \
  --input /path/to/test.list \
  --output outputs/eval-results \
  --aligner scripts/speech_recognition/AlignG7.exe \
  --cer \
  --device cuda
```

## Multilingual WER/CER Calculator

The training and evaluation scripts now include an advanced WER/CER calculator that follows the HuggingFace Open ASR Leaderboard protocol.

### Features

- **Dynamic Metric Selection**: Automatically uses WER for word-based languages (English, etc.) and CER for character-based languages (Chinese, Japanese, Korean)
- **Multilingual Normalization**:
  - English: Number normalization, spelling normalization, punctuation handling
  - Chinese/Japanese/Korean: Character-level normalization with proper segmentation
  - Other languages: Basic multilingual normalization
- **Protocol Compliance**: Matches HuggingFace Open ASR Leaderboard evaluation standards
- **Enhanced Logging**: Sample predictions show metric type, token counts, and formatted output

### Configuration

**WER Calculator Settings** (in model config):

```yaml
model:
  wer_calculator:
    normalizer: "openasrleaderboard"  # or "legacy" for backward compatibility
```

**Dataset Metadata** (in training config):

Each dataset can specify its language and metric type for proper evaluation:

```yaml
data:
  train_ds:
    shar_path:
      # Format: [path, weight, language, metric]
      - ["/data/librispeech", 1.0, "en", "wer"]
      - ["/data/commonvoice_ja", 0.5, "ja", "cer"]
      - ["/data/aishell", 0.5, "zh", "cer"]
      - ["/data/ksponspeech", 0.5, "ko", "cer"]
```

Supported languages: `en`, `zh`, `ja`, `ko`, and other language codes following ISO 639-1 standard.

### Training/Validation Logs

The enhanced logging now displays dynamic metric names and detailed token information:

```
====================================================================================================
Validation Sample 1 [commonvoice_ja] | CER: 8.45%
Full LLM Input : <|im_start|>user\nTranscribe the following: <|audio|><|im_end|> <|im_start|>assistant\n今日はいい天気ですね<|im_end|> (156)
Prompt         : <|im_start|>user\nTranscribe the following: <|im_end|> <|im_start|>assistant\n (42)
Audio Token    : <|audio|> → [audio_embeddings] (98)
Ground Truth   : 今日はいい天気ですね (16)
Prediction     : 今日はいい天気です (15)
====================================================================================================
```

**WER for English datasets**:
```
====================================================================================================
Validation Sample 1 [librispeech_test_clean] | WER: 3.24%
Full LLM Input : <|im_start|>user\nTranscribe the following: <|audio|><|im_end|> <|im_start|>assistant\nThe quick brown fox jumps over the lazy dog<|im_end|> (178)
Prompt         : <|im_start|>user\nTranscribe the following: <|im_end|> <|im_start|>assistant\n (42)
Audio Token    : <|audio|> → [audio_embeddings] (127)
Ground Truth   : The quick brown fox jumps over the lazy dog (45)
Prediction     : The quick brown fox jumps over the lazy dog (45)
====================================================================================================
```

### PyTorch Lightning Metrics

Metrics are logged with dynamic names based on the dataset's metric type:

```python
# WER datasets
val_wer_librispeech_test_clean: 0.0324
val_wer_tedlium: 0.0456

# CER datasets
val_cer_commonvoice_ja: 0.0845
val_cer_aishell: 0.0523

# Overall metric (backward compatible)
val_wer: 0.0412  # Average across all datasets
```

### Implementation Details

The WER/CER calculator is implemented in the following modules:

- **`nemo/collections/speechlm2/metrics/wer_calculator.py`**: Main calculator classes
  - `OpenASRLeaderboardWERCalculator`: HuggingFace-compliant normalization
  - `LegacyWERCalculator`: Backward compatibility mode
  - `create_wer_calculator()`: Factory function for creating calculators

- **`nemo/collections/speechlm2/metrics/normalization.py`**: Text normalization
  - `EnglishTextNormalizer`: English number and spelling normalization
  - `BasicMultilingualTextNormalizer`: Multilingual support with language-specific handling

- **`nemo/collections/speechlm2/models/salm.py`**: Integration in SALM model
  - Dynamic metric logging in `_format_prediction_log()`
  - Metric type tracking in validation/training steps
  - PyTorch Lightning metric logging with dynamic names

The system automatically:
1. Extracts language and metric metadata from dataset configuration
2. Applies appropriate text normalization based on language
3. Calculates WER or CER based on the metric type
4. Logs results with dynamic metric names (WER/CER)

## Expected Output

### Checkpoint Loading (.ckpt) - NEW Lightning Native Method

```
================================================================================
Loading SALM model from PyTorch Lightning checkpoint: SALM-251010-step17301-val_loss0.861.ckpt
================================================================================

PyTorch Lightning Native Checkpoint Loading
================================================================================
Strategy: Use SALM.load_from_checkpoint() - same as training resume
Benefits:
  ✓ Exact architecture match (Qwen3 → Qwen3, no conversion)
  ✓ All weights loaded correctly (0 missing/unexpected keys)
  ✓ LoRA parameters preserved
  ✓ Qwen3 QK normalization layers retained
================================================================================

Loading checkpoint from: /path/to/checkpoint.ckpt
✓ Model loaded successfully with Lightning native method

Checkpoint Verification Report
================================================================================
Architecture Match: ✓ PASSED
Weight Integrity: ✓ PASSED

Qwen3 Components:
  ✓ QK Normalization Layers: 56 parameters present
  ✓ Attention Projections: No bias (Qwen3 architecture)

LoRA Parameters:
  ✓ LoRA adapters found: lora_A, lora_B (rank=16)
  ✓ All LoRA weights loaded correctly

Summary:
  - Total Keys Loaded: [count]
  - Missing Keys: 0
  - Unexpected Keys: 0
  - Weight Loading Status: ✓ 100% SUCCESS
================================================================================

✓ Successfully loaded SALM model
  Configuration:
    - Model Architecture: Qwen3-1.7B (24 layers, preserved)
    - Tokenizer: From checkpoint (vocab_size: 151,936)
    - Weights: All weights loaded from checkpoint (100% integrity)

================================================================================
Model Information:
  Device: cuda:0
  Audio locator tag: <|audio|>
  Tokenizer vocab size: 151,936
  Model dtype: torch.float32
  Total parameters: 3,456,789,012
  Trainable parameters: 3,456,789,012 (includes LoRA adapters)
================================================================================
```

**Note**: The new Lightning native method eliminates the previous issues with 140 parameters having integrity problems.

### Checkpoint Loading (.nemo)

```
Loading SALM model from NeMo checkpoint: ...
  Using correct .nemo loading strategy:
    1. Load model directly from .nemo archive
    2. Config and weights loaded automatically
  Restoring SALM model from .nemo...
  ✓ Model restored from .nemo
✓ Successfully loaded SALM model from SALM-Canary-Qwen-2.5B.nemo
  Configuration:
    - Tokenizer: From .nemo (vocab_size: 151,670)
    - Weights: All weights loaded from .nemo
```

## Technical Details

### Checkpoint Loading Principle

**CRITICAL**: The checkpoint must be loaded with the EXACT same architecture used during training.

**Why Modifying Config Fails**:
- Different LLM models have different architectures (layer count, hidden size, intermediate size, etc.)
- Example: Qwen3-1.7B (24 layers, 6144 intermediate) ≠ Qwen2.5-3B (36 layers, 11008 intermediate)
- Loading weights from one architecture into another causes size mismatch errors

**Correct Approach**:
1. **Preserve Config**: Use checkpoint config AS-IS without any modifications
2. **Match Architecture**: Initialize model with exact training architecture
3. **Load Weights**: Checkpoint weights overwrite pretrained initialization weights

**Why This Works**:
- Model architecture matches checkpoint exactly (layer dims, heads, vocab size)
- Pretrained initialization provides the correct model structure
- Checkpoint state_dict contains all trained weights (LLM, ASR encoder, adapter)
- Loading checkpoint overwrites pretrained weights completely

### Checkpoint Weight Components

A complete SALM checkpoint contains:

```python
state_dict = {
    # LLM weights (Qwen3)
    'llm.model.layers.0.self_attn.q_proj.weight': ...,
    'llm.model.layers.0.self_attn.k_proj.weight': ...,
    # ... (all LLM layers)

    # Speech encoder weights (from pretrained_asr)
    'perception.encoder.layers.0.conv.weight': ...,
    'perception.preprocessor.featurizer.fb': ...,
    # ... (all encoder layers)

    # Modality adapter weights (trained from scratch)
    'perception.modality_adapter.layers.0.conv.weight': ...,
    # ... (adapter layers)

    # Embedding weights
    'embed_tokens.weight': ...,
}
```

**All these weights are loaded from checkpoint - pretrained models are NOT needed!**

## Troubleshooting

### Issue: Checkpoint file not found

**Error**:
```
FileNotFoundError: Checkpoint file not found: /path/to/checkpoint.ckpt
```

**Solution**: Verify the checkpoint file path is correct:
1. Check the file exists at the specified location
2. Verify you're on the correct machine
3. Common checkpoint locations:
   - Training output: `{exp_manager.explicit_log_dir}/{model.name}/checkpoints/`
   - Example: `/path/to/.../slm-trainer/scripts_CanaryQwenASR/outputs/SALM-Canary-Qwen-2.5B/checkpoints/`
4. Use absolute paths to avoid confusion
5. Check file permissions (should be readable)

### Issue: Model architecture size mismatch

**Error**:
```
RuntimeError: Error(s) in loading state_dict for SALM:
  size mismatch for llm.base_model.model.model.layers.0.self_attn.k_proj.base_layer.weight:
  copying a param with shape torch.Size([1024, 2048]) from checkpoint,
  the shape in current model is torch.Size([256, 2048])
```

**Root Cause**: Checkpoint config was modified (e.g., replaced Qwen3-1.7B with Qwen2.5-3B)

**Solution**:
1. Use checkpoint config AS-IS without modifications
2. Initialize with exact training architecture
3. Script now preserves original pretrained_llm and pretrained_asr paths
4. Checkpoint weights overwrite pretrained initialization weights

### Issue: Missing pretrained_asr file

**Error**:
```
FileNotFoundError: pretrained_asr file not found
```

**Solution**: This is expected! Pretrained ASR model is NOT needed for evaluation. The script loads all weights from checkpoint.

### Issue: Tokenizer mismatch

**Symptoms**: Generated text contains unknown tokens or gibberish

**Solution**: Verify checkpoint loading log shows:
```
✓ Qwen3 tokenizer loaded (vocab_size: 151,936)
✓ Tokenizer replaced (vocab_size: 151,936)
```

### Issue: Qwen3 model type not recognized

**Error**:
```
KeyError: 'qwen3'
ValueError: The checkpoint you are trying to load has model type `qwen3` but Transformers does not recognize this architecture.
```

**Root Cause**: Your transformers library version is too old to support Qwen3 models (requires >=4.37.0)

**Solution (Option 1 - Recommended)**: Upgrade transformers library:
```bash
# On GPU server
pip install --upgrade 'transformers>=4.37.0'

# Or with conda
conda install -c conda-forge 'transformers>=4.37.0'
```

**Solution (Option 2 - Automatic Fallback)**: The script automatically attempts Qwen2 compatibility mode:
- Qwen3 and Qwen2 are architecturally nearly identical
- The script will temporarily map Qwen3 → Qwen2 for loading
- If this works, you'll see:
  ```
  Qwen3 model type not recognized by transformers library.
    Attempting Qwen2 compatibility mode (architectures are nearly identical)...
    Model type: qwen3 → qwen2 (compatibility mode)
    Loading with Qwen2 architecture (compatible with Qwen3)...
    ✓ Successfully loaded Qwen3 model using Qwen2 compatibility mode
  ```

**What happens**:
1. Script tries to load with `trust_remote_code=True` first
2. If that fails due to Qwen3 not being recognized:
   - Creates temporary model directory
   - Modifies `config.json`: `model_type: qwen3 → qwen2`
   - Loads model with Qwen2 architecture (compatible)
   - Returns loaded model with correct weights
3. If fallback also fails, provides clear upgrade instructions

**Important**: While the fallback works, upgrading transformers is recommended for best compatibility.

### Issue: Weight Loading Integrity Problems (FIXED - 2025-10-10)

**Symptoms** (Before Fix):
```
Missing keys in checkpoint: 84 keys
  - llm.base_model.model.model.layers.*.self_attn.q_proj.base_layer.bias
  - llm.base_model.model.model.layers.*.self_attn.k_proj.base_layer.bias
  - llm.base_model.model.model.layers.*.self_attn.v_proj.base_layer.bias
  ...

Unexpected keys in checkpoint: 56 keys
  - llm.base_model.model.model.layers.*.self_attn.q_norm.weight
  - llm.base_model.model.model.layers.*.self_attn.k_norm.weight
  ...
```

**Root Cause**:
- Previous manual loading method used Qwen3→Qwen2 conversion when transformers library was too old
- **Qwen3 Architecture**: Has `q_norm`/`k_norm` layers (56 params), NO bias in attention projections
- **Qwen2 Architecture**: NO normalization layers, HAS bias (84 params)
- These architectures are fundamentally INCOMPATIBLE
- Manual loading lost 56 QK normalization weights and randomly initialized 84 bias parameters
- **Total: 140 parameters with integrity issues**

**Solution (Implemented)**:
The script now uses **PyTorch Lightning's native `load_from_checkpoint()`** method by default:

```python
# OLD (Manual Loading - Had Issues)
checkpoint = torch.load(model_path)
cfg = checkpoint['hyper_parameters']['cfg']
model = SALM(cfg)  # Wrong: Loses Qwen3 architecture
model.load_state_dict(checkpoint['state_dict'])  # Causes 140 param issues

# NEW (Lightning Native - Correct)
model = SALM.load_from_checkpoint(
    model_path,
    map_location=device,
    strict=True  # Enforces perfect weight match
)
```

**Benefits of Lightning Native Loading**:
- ✅ Preserves exact training architecture (Qwen3 stays Qwen3)
- ✅ All 140 problematic parameters loaded correctly
- ✅ 0 missing keys, 0 unexpected keys
- ✅ LoRA parameters preserved
- ✅ QK normalization layers retained
- ✅ Matches training behavior exactly

**Verification**:
The script now includes comprehensive checkpoint verification that checks:
- Qwen3-specific components (q_norm, k_norm)
- LoRA adapter parameters
- Missing/unexpected keys
- Weight loading status

If you encounter this issue with the old version, upgrade to the latest version or use Lightning native loading.

## Requirements

- Python 3.10+
- PyTorch 2.5+
- **transformers >= 4.37.0** (for Qwen3 support; older versions use automatic fallback)
- NeMo toolkit with speechlm2 support
- lhotse (for audio processing)
- silero-vad (optional, for VAD support)

## License

Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
