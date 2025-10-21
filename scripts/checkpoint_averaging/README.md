Checkpoint Averaging
====================

Overview
--------
The checkpoint averaging scripts are used to compute the weighted average of multiple checkpoints. This ensemble technique can improve model performance by combining multiple training states. Different scripts are provided for different model architectures and checkpoint formats.

Available Scripts
-----------------

### 1. SALM Checkpoint Averaging (`salm_checkpoint_averaging.py`)

For SALM (Speech-Augmented Language Model) checkpoints from the speechlm2 collection.

```bash
# Average top 3 checkpoints by WER and save as .nemo
python scripts/checkpoint_averaging/salm_checkpoint_averaging.py \
    --checkpoint_dir /path/to/checkpoints \
    --output_dir /path/to/output \
    --output_name salm_averaged \
    --top_n 3 \
    --output_nemo salm_averaged.nemo

# Average specific checkpoints with custom weights
python scripts/checkpoint_averaging/salm_checkpoint_averaging.py \
    --checkpoint_dir /path/to/checkpoints \
    --steps 14000 20500 25000 \
    --weights 0.2 0.3 0.5 \
    --output_nemo salm_averaged.nemo \
    --model_config /path/to/config.yaml
```

**Arguments**:
- `--checkpoint_dir`: Directory containing .ckpt files
- `--output_dir`: Output directory for averaged checkpoint
- `--output_name`: Base name for output files
- `--steps`: List of specific checkpoint steps to average
- `--top_n`: Select N checkpoints with lowest validation WER
- `--weights`: Custom weights for averaging (must sum to 1.0)
- `--output_nemo`: Path to save as .nemo format
- `--model_config`: Path to model configuration YAML (for .nemo creation)

### 2. Traditional ASR Checkpoint Averaging (`weighted_checkpoint_averaging.py`)

For traditional CTC-based ASR models (EncDecCTCModelBPE).

```bash
# Average top 3 checkpoints by WER
python scripts/checkpoint_averaging/weighted_checkpoint_averaging.py \
    --checkpoint_dir /path/to/checkpoints \
    --output_dir /path/to/output \
    --top_n 3

# Update existing .nemo file with averaged weights
python scripts/checkpoint_averaging/weighted_checkpoint_averaging.py \
    --checkpoint_dir /path/to/checkpoints \
    --top_n 3 \
    --input_nemo model.nemo \
    --output_nemo model_averaged.nemo
```

**Arguments**:
- `--checkpoint_dir`: Directory containing .ckpt files
- `--output_dir`: Output directory for averaged checkpoint
- `--output_name`: Base name for output files
- `--steps`: List of specific checkpoint steps to average
- `--top_n`: Select N checkpoints with lowest validation WER
- `--weights`: Custom weights for averaging
- `--input_nemo`: Existing .nemo file to update
- `--output_nemo`: Path to save updated .nemo file

### 3. Zarr Distributed Checkpoint Averaging (`zarr_distributed_checkpoint_averaging.py`)

Use the following command to run the checkpoint averaging script for zarr distributed checkpoints:

```shell
python scripts/checkpoint_averaging/zarr_distributed_checkpoint_averaging.py \
    --name_prefix <output checkpoint name> \
    --checkpoint_dir <folder with zarr distriubted checkpoints> \
    --steps <optionally a list of checkpoint steps to average, if not provided, it will average all the checkpoints>
```
**Arguments**:
- `--name_prefix`: Specifies the prefix for the generated averaged checkpoint.
- `--checkpoint_dir`: Specifies the folder containing zarr distributed checkpoints.
- `--steps`: (Optional) A comma-separated list of checkpoint steps to average (e.g., 1000, 2000, 3000). If not provided, the script will average all the checkpoints in the directory.

After execution, the script generates averaged checkpoint in `<checkpoint_dir>` named `<name_prefix>-averaged`.

Checkpoint Naming Convention
----------------------------

For SALM and traditional ASR scripts, checkpoints should follow this naming pattern:
```
step=14000-val_wer=0.0652-val_loss=1.50.ckpt
```

The scripts extract:
- `step`: Training step number
- `val_wer`: Validation WER (Word Error Rate) or CER (Character Error Rate)

**Note**: Files ending with `-last.ckpt` are automatically excluded from averaging.

Weight Calculation
------------------

When weights are not explicitly provided, the scripts automatically calculate weights based on validation metrics:

1. **WER-based weighting** (default):
   - Extract WER values from checkpoint filenames
   - Calculate inverse WER (1/WER) - lower WER gets higher weight
   - Normalize weights to sum to 1.0

2. **Equal weighting**:
   - Used when WER values are not available in filenames
   - Each checkpoint gets equal weight (1/N)

3. **Custom weighting**:
   - Specify exact weights using `--weights` argument
   - Weights are automatically normalized to sum to 1.0

Tips for Best Results
---------------------

1. **Checkpoint Selection**:
   - Use `--top_n 3` to `--top_n 5` for optimal ensemble performance
   - Select checkpoints from the final training phase for better results
   - Ensure selected checkpoints have similar validation performance

2. **For SALM Models**:
   - Always use `salm_checkpoint_averaging.py` for speechlm2 models
   - Provide `--model_config` for complete .nemo file creation
   - The script handles complex state_dict structure (LLM, perception, adapters)

3. **For Traditional ASR Models**:
   - Use `weighted_checkpoint_averaging.py` for CTC-based models
   - Can update existing .nemo files with `--input_nemo` and `--output_nemo`

4. **Memory Management**:
   - Scripts load checkpoints with `map_location='cpu'` to avoid GPU memory issues
   - Process checkpoints sequentially to minimize memory usage

5. **Validation**:
   - Always validate the averaged model on a test set
   - Compare performance against individual checkpoints
   - Monitor for significant performance improvements (typically 2-5% relative)
