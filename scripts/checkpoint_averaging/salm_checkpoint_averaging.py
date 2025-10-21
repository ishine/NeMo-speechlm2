#!/usr/bin/env python3
"""
SALM Checkpoint Averaging Script

This script performs weighted averaging of SALM model checkpoints and can save
the result in both .ckpt and .nemo formats.

Example usage:
    python scripts/checkpoint_averaging/salm_checkpoint_averaging.py \\
        --checkpoint_dir recipes/CanaryQwenASR/outputs/SALM-Canary-Qwen-2.5B-20251019/checkpoints \\
        --output_dir recipes/CanaryQwenASR/outputs/SALM-Canary-Qwen-2.5B-20251019 \\
        --top_n 20 \\
        --output_nemo canary_qwen_top20_averaged.nemo \\
        --model_config salm_canary_qwen_2.5b_20251019.yaml

Options:
    1. Specific steps: --steps 14000 20500 ...
    2. Top N by WER: --top_n 20

Execution context:
    - Run from: /path/to/slm-trainer
    - Uses local NeMo source code (not installed package)
"""

import sys
import os

# ============================================================================
# CRITICAL: Set up Python path BEFORE any other imports
# This ensures we use the local NeMo source code instead of installed packages
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NEMO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))

if NEMO_ROOT not in sys.path:
    sys.path.insert(0, NEMO_ROOT)
    print(f"[INIT] Added NeMo root to Python path: {NEMO_ROOT}")
else:
    print(f"[INIT] NeMo root already in Python path: {NEMO_ROOT}")

# Verify the speechlm2 module path exists
SPEECHLM2_PATH = os.path.join(NEMO_ROOT, 'nemo', 'collections', 'speechlm2')
if os.path.exists(SPEECHLM2_PATH):
    print(f"[INIT] Verified speechlm2 module path: {SPEECHLM2_PATH}")
else:
    print(f"[WARNING] speechlm2 module path not found: {SPEECHLM2_PATH}")
    print(f"[WARNING] .nemo file creation may fail")

# Now import all other dependencies
import argparse
import logging
import re
import torch
import numpy as np
import tarfile
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from omegaconf import OmegaConf

# ============================================================================
# Logging Configuration
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger("salm_checkpoint_averaging")


def extract_checkpoint_info(filename: str) -> dict:
    """
    Extract step number and validation WER from checkpoint filename

    Args:
        filename: Checkpoint filename like 'step=14000-val_wer=0.0652-val_loss=1.50.ckpt'

    Returns:
        dict: Dictionary with 'step' and 'val_wer' keys
    """
    info = {}

    # Extract step number
    step_match = re.search(r'step=(\d+)', filename)
    if step_match:
        info['step'] = int(step_match.group(1))
    else:
        info['step'] = -1  # Default if no step found

    # Extract validation WER
    wer_match = re.search(r'val_wer=([\d.]+)', filename)
    if wer_match:
        info['val_wer'] = float(wer_match.group(1))
    else:
        info['val_wer'] = float('inf')  # Default if no WER found

    return info


def find_checkpoints(checkpoint_dir: str, steps: Optional[List[int]] = None, top_n: Optional[int] = None) -> List[dict]:
    """
    Find checkpoints in the given directory based on criteria

    Args:
        checkpoint_dir: Directory containing checkpoints
        steps: List of specific step numbers to select
        top_n: Number of checkpoints to select based on lowest val_wer

    Returns:
        list: List of checkpoint information dictionaries
    """
    logger.info(f"Searching for checkpoints in directory: {checkpoint_dir}")

    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    all_checkpoints = []

    # Gather all checkpoint files, excluding last checkpoints
    file_count = 0
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.ckpt') and not file.endswith('-last.ckpt'):
            file_count += 1
            checkpoint_path = os.path.join(checkpoint_dir, file)
            info = extract_checkpoint_info(file)
            all_checkpoints.append({
                'filename': file,
                'path': checkpoint_path,
                'step': info['step'],
                'val_wer': info['val_wer']
            })
            logger.debug(f"Found checkpoint: {file} (step={info['step']}, wer={info['val_wer']:.4f})")

    logger.info(f"Total checkpoint files found: {file_count}")

    if not all_checkpoints:
        raise ValueError(f"No valid checkpoint files found in {checkpoint_dir}")

    # Filter by specific steps if provided
    if steps:
        logger.info(f"Filtering checkpoints by specific steps: {steps}")
        filtered_checkpoints = []
        for step in steps:
            matching = [ckpt for ckpt in all_checkpoints if ckpt['step'] == step]
            if matching:
                filtered_checkpoints.extend(matching)
                logger.info(f"Found checkpoint for step {step}: {matching[0]['filename']}")
            else:
                logger.warning(f"No checkpoint found for step {step}")

        if not filtered_checkpoints:
            raise ValueError("None of the specified steps matched any checkpoints")

        logger.info(f"Selected {len(filtered_checkpoints)} checkpoints by steps")
        return filtered_checkpoints

    # Select top N checkpoints with lowest val_wer
    elif top_n:
        logger.info(f"Selecting top {top_n} checkpoints by lowest validation WER")
        sorted_checkpoints = sorted(all_checkpoints, key=lambda x: x['val_wer'])
        selected = sorted_checkpoints[:top_n]

        logger.info(f"Top {top_n} checkpoints selected:")
        for i, ckpt in enumerate(selected, 1):
            logger.info(f"  {i}. {ckpt['filename']} - WER: {ckpt['val_wer']:.4f}")

        return selected

    # Return all checkpoints if no filter specified
    else:
        logger.info(f"Using all {len(all_checkpoints)} checkpoints")
        return all_checkpoints


def average_checkpoints(checkpoints: List[dict], weights: Optional[np.ndarray] = None) -> Tuple[dict, Optional[dict]]:
    """
    Perform weighted averaging of checkpoint state_dicts

    Args:
        checkpoints: List of checkpoint info dictionaries
        weights: Optional numpy array of weights (must sum to 1.0)

    Returns:
        tuple: (averaged_state_dict, original_hyperparameters)
    """
    n = len(checkpoints)
    logger.info(f"Starting checkpoint averaging process for {n} checkpoints")

    # Calculate or validate weights
    if weights is None:
        # Calculate weights based on validation WER
        if all(ckpt['val_wer'] < float('inf') for ckpt in checkpoints):
            # Get WER values and invert them (lower WER = higher weight)
            wer_values = np.array([ckpt['val_wer'] for ckpt in checkpoints])
            logger.info(f"WER values: min={wer_values.min():.4f}, max={wer_values.max():.4f}, mean={wer_values.mean():.4f}")

            # Calculate inverse WER (higher = better)
            inverse_wer = 1.0 / wer_values
            # Normalize to sum to 1.0
            weights = inverse_wer / inverse_wer.sum()
            logger.info("Calculated weights using inverse WER (lower WER = higher weight)")
        else:
            # Equal weights if WER not available
            weights = np.ones(n, dtype=np.float32) / n
            logger.info("Using equal weights (WER values not available for all checkpoints)")

    # Normalize weights to sum to 1.0
    weights = weights / weights.sum()
    logger.info(f"Weight sum verification: {weights.sum():.6f} (should be 1.0)")

    # Log checkpoint and weight information
    logger.info("=" * 80)
    logger.info(f"Checkpoint averaging configuration:")
    for i, (ckpt, weight) in enumerate(zip(checkpoints, weights)):
        logger.info(f"  [{i+1}/{n}] {ckpt['filename']}")
        logger.info(f"       Step: {ckpt['step']}, WER: {ckpt['val_wer']:.4f}, Weight: {weight:.4f}")
    logger.info("=" * 80)

    # Create state dict for averaged model
    averaged_state_dict = {}
    param_count = 0
    float_param_count = 0
    original_hyperparameters = None

    # Process each checkpoint with its corresponding weight
    for i, (ckpt, weight) in enumerate(zip(checkpoints, weights)):
        logger.info(f"Processing checkpoint {i+1}/{n}: {ckpt['filename']}")

        # Load the checkpoint (weights_only=False needed for Lightning checkpoints)
        logger.debug(f"Loading checkpoint from: {ckpt['path']}")
        checkpoint = torch.load(ckpt['path'], map_location='cpu', weights_only=False)

        # Preserve hyperparameters from first checkpoint for .nemo creation
        if i == 0 and "hyper_parameters" in checkpoint:
            original_hyperparameters = checkpoint["hyper_parameters"]
            logger.info("Preserved hyperparameters from first checkpoint")

        # Extract state dict from Lightning checkpoint
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            logger.debug("Extracted state_dict from Lightning checkpoint")
        else:
            state_dict = checkpoint
            logger.debug("Using checkpoint as raw state_dict")

        if i == 0:
            param_count = len(state_dict)
            logger.info(f"Total parameters in state_dict: {param_count}")

        # Initialize or update the averaged state dict
        if i == 0:
            # For first checkpoint, initialize the averaged state dict
            for key, value in state_dict.items():
                if value.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    # Only average floating point parameters
                    averaged_state_dict[key] = value.clone() * weight
                    float_param_count += 1
                else:
                    # For non-float parameters (like embeddings indices), just copy
                    averaged_state_dict[key] = value.clone()

            logger.info(f"Initialized averaged state_dict: {float_param_count} float params, {param_count - float_param_count} non-float params")
        else:
            # For subsequent checkpoints, add weighted values
            mismatched_keys = []
            for key, value in state_dict.items():
                if key in averaged_state_dict:
                    if value.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                        averaged_state_dict[key] += value * weight
                    # Non-float parameters should be identical across checkpoints
                else:
                    mismatched_keys.append(key)
                    logger.warning(f"Key '{key}' found in checkpoint {i+1} but not in first checkpoint")

            if mismatched_keys:
                logger.warning(f"Total mismatched keys in checkpoint {i+1}: {len(mismatched_keys)}")

        logger.info(f"Completed processing checkpoint {i+1}/{n}")

    logger.info("Checkpoint averaging completed successfully")
    return averaged_state_dict, original_hyperparameters


def save_averaged_checkpoint(
    averaged_state_dict: dict,
    output_dir: str,
    output_name: str,
    checkpoint_info: dict,
    original_checkpoint_hyperparams: Optional[dict] = None
) -> str:
    """
    Save the averaged checkpoint as a .ckpt file

    Args:
        averaged_state_dict: Averaged state dictionary
        output_dir: Directory to save the checkpoint
        output_name: Base name for the output file
        checkpoint_info: Information about averaging (for filename)
        original_checkpoint_hyperparams: Hyperparameters from original checkpoint to preserve

    Returns:
        str: Path to the saved checkpoint
    """
    logger.info("Preparing to save averaged checkpoint")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Create descriptive filename
    if checkpoint_info.get('steps'):
        steps_str = '_'.join(str(step) for step in checkpoint_info['steps'])
        output_filename = f"{output_name}_steps_{steps_str}.ckpt"
    elif checkpoint_info.get('top_n'):
        output_filename = f"{output_name}_top_{checkpoint_info['top_n']}.ckpt"
    else:
        output_filename = f"{output_name}.ckpt"

    output_path = os.path.join(output_dir, output_filename)

    # Create a proper Lightning checkpoint structure
    checkpoint = {
        "state_dict": averaged_state_dict,
        "epoch": 0,  # Reset epoch counter
        "global_step": 0,  # Reset global step
        "pytorch-lightning_version": "2.0.0",  # Placeholder version
    }

    # IMPORTANT: Preserve hyperparameters from original checkpoint for .nemo creation
    if original_checkpoint_hyperparams:
        checkpoint["hyper_parameters"] = original_checkpoint_hyperparams
        logger.info("Preserved original checkpoint hyperparameters for .nemo compatibility")

    logger.info(f"Saving averaged checkpoint to: {output_path}")
    logger.info(f"Checkpoint size: {len(averaged_state_dict)} parameters")

    torch.save(checkpoint, output_path)

    # Verify the saved file
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        logger.info(f"Checkpoint saved successfully")
        logger.info(f"File size: {file_size / (1024**3):.2f} GB")
    else:
        logger.error(f"Failed to save checkpoint at {output_path}")
        raise RuntimeError(f"Checkpoint file was not created at {output_path}")

    return output_path


def create_nemo_from_checkpoint(
    checkpoint_path: str,
    output_nemo_path: str,
    model_config_path: Optional[str] = None
) -> None:
    """
    Create a .nemo file from the averaged checkpoint

    Args:
        checkpoint_path: Path to the averaged checkpoint
        output_nemo_path: Path where the .nemo file should be saved
        model_config_path: Optional path to model config (will try to extract from checkpoint if not provided)
    """
    logger.info("=" * 80)
    logger.info("Starting .nemo file creation process")
    logger.info("=" * 80)

    # Import SALM model class (should work since we set up Python path at the beginning)
    try:
        from nemo.collections.speechlm2.models import SALM
        logger.info("Successfully imported SALM model class")
    except ImportError as e:
        logger.error(f"Failed to import SALM model class: {e}")
        logger.error(f"Python path: {sys.path[:3]}")
        logger.error(f"Expected speechlm2 at: {SPEECHLM2_PATH}")
        logger.error("")
        logger.error("Troubleshooting steps:")
        logger.error("  1. Verify you are running from the correct directory")
        logger.error("  2. Check that nemo/collections/speechlm2/ exists in the source tree")
        logger.error("  3. Ensure all __init__.py files are present in the module path")
        raise ImportError(f"Cannot import SALM model class. {e}")

    # Load the checkpoint to get hyperparameters
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    logger.info("Checkpoint loaded successfully")

    # Extract or load configuration
    model_cfg = None

    if model_config_path:
        logger.info(f"Loading model configuration from: {model_config_path}")
        if not os.path.exists(model_config_path):
            logger.error(f"Model config file not found: {model_config_path}")
            raise FileNotFoundError(f"Model config file not found: {model_config_path}")

        config = OmegaConf.load(model_config_path)
        model_cfg = OmegaConf.to_container(config.model if 'model' in config else config, resolve=True)
        logger.info("Model configuration loaded from file")

    elif 'hyper_parameters' in checkpoint and 'cfg' in checkpoint['hyper_parameters']:
        logger.info("Extracting model configuration from checkpoint hyperparameters")
        model_cfg = checkpoint['hyper_parameters']['cfg']
        logger.info("Model configuration extracted from checkpoint")

    else:
        logger.error("Cannot create .nemo file without model configuration")
        logger.error("Configuration not found in checkpoint and no --model_config provided")
        logger.error("")
        logger.error("Solutions:")
        logger.error("  1. Provide --model_config pointing to the training config YAML file")
        logger.error("  2. Use a checkpoint that contains hyper_parameters")
        raise ValueError("Model configuration not available")

    # Log configuration details
    logger.info("Model configuration details:")
    if isinstance(model_cfg, dict):
        logger.info(f"  pretrained_llm: {model_cfg.get('pretrained_llm', 'N/A')}")
        logger.info(f"  pretrained_asr: {model_cfg.get('pretrained_asr', 'N/A')}")
        logger.info(f"  audio_locator_tag: {model_cfg.get('audio_locator_tag', 'N/A')}")

    # Create model instance
    logger.info("Initializing SALM model instance")
    logger.info("Note: This may take several minutes for large models")

    # IMPORTANT: Do NOT set pretrained_weights=False!
    # Instead, use skip_perception_setup=True to skip perception initialization
    # We'll initialize perception module separately from checkpoint config
    logger.info("Creating model with skip_perception_setup=True")
    logger.info("Perception module will be initialized separately from checkpoint config")

    try:
        # Keep pretrained_weights as-is (usually True), but skip perception setup
        model = SALM(cfg=model_cfg, skip_perception_setup=True)
        logger.info("SALM model instance created successfully (without perception)")
    except Exception as e:
        logger.error(f"Failed to create SALM model instance: {e}")
        raise RuntimeError(f"Model instantiation failed: {e}")

    # Initialize perception module from checkpoint config
    # This is the same approach used in SALM.restore_from()
    logger.info("Initializing perception module from checkpoint config")

    if 'perception' not in model.cfg or model.cfg.perception is None:
        logger.error("Perception config not found in model configuration")
        raise RuntimeError("Perception config missing from model configuration")

    # Check and repair incomplete perception config
    perception_cfg = model.cfg.perception
    logger.info("Checking perception config completeness")

    # Check if perception config is complete
    has_preprocessor = hasattr(perception_cfg, 'preprocessor') and perception_cfg.preprocessor is not None
    has_encoder = hasattr(perception_cfg, 'encoder') and perception_cfg.encoder is not None
    has_output_dim = hasattr(perception_cfg, 'output_dim') and perception_cfg.output_dim is not None

    if not all([has_preprocessor, has_encoder, has_output_dim]):
        logger.warning("Perception config is incomplete in the loaded config file")
        logger.info("Attempting to restore perception config from checkpoint or pretrained ASR model")

        # First try: Restore from checkpoint hyperparameters
        complete_perception_found = False
        if 'hyper_parameters' in checkpoint and 'cfg' in checkpoint['hyper_parameters']:
            ckpt_cfg = checkpoint['hyper_parameters']['cfg']
            if 'perception' in ckpt_cfg and ckpt_cfg['perception'] is not None:
                ckpt_perception = ckpt_cfg['perception']

                # Check checkpoint's perception config
                ckpt_has_preprocessor = 'preprocessor' in ckpt_perception and ckpt_perception['preprocessor'] is not None
                ckpt_has_encoder = 'encoder' in ckpt_perception and ckpt_perception['encoder'] is not None
                ckpt_has_output_dim = 'output_dim' in ckpt_perception and ckpt_perception['output_dim'] is not None

                if all([ckpt_has_preprocessor, ckpt_has_encoder, ckpt_has_output_dim]):
                    logger.info("Found complete perception config in checkpoint, using it instead")
                    # Update model's perception config with checkpoint's complete config
                    model.cfg.perception = OmegaConf.create(ckpt_perception)
                    perception_cfg = model.cfg.perception
                    complete_perception_found = True
                    logger.info("Perception config restored from checkpoint successfully")
                else:
                    logger.warning("Checkpoint's perception config is also incomplete")
                    logger.warning(f"  Has preprocessor: {ckpt_has_preprocessor}")
                    logger.warning(f"  Has encoder: {ckpt_has_encoder}")
                    logger.warning(f"  Has output_dim: {ckpt_has_output_dim}")

        # Second try: Load from pretrained_asr .nemo file if specified in config
        if not complete_perception_found:
            pretrained_asr_path = model.cfg.get('pretrained_asr')
            if pretrained_asr_path and os.path.exists(pretrained_asr_path) and pretrained_asr_path.endswith('.nemo'):
                logger.info(f"Attempting to load perception config from pretrained ASR: {pretrained_asr_path}")
                try:
                    import tarfile
                    import tempfile
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Extract .nemo file
                        with tarfile.open(pretrained_asr_path, 'r') as tar:
                            tar.extractall(temp_dir)

                        # Load ASR model config
                        asr_config_path = os.path.join(temp_dir, 'model_config.yaml')
                        if os.path.exists(asr_config_path):
                            asr_config = OmegaConf.load(asr_config_path)

                            # Extract preprocessor and encoder from ASR config
                            if 'preprocessor' in asr_config and 'encoder' in asr_config:
                                # Create complete perception config
                                perception_dict = {
                                    'preprocessor': OmegaConf.to_container(asr_config.preprocessor),
                                    'encoder': OmegaConf.to_container(asr_config.encoder),
                                    'output_dim': model.cfg.perception.get('output_dim', asr_config.encoder.d_model)
                                }

                                # Add modality adapter if it exists
                                if hasattr(model.cfg.perception, 'modality_adapter'):
                                    perception_dict['modality_adapter'] = OmegaConf.to_container(model.cfg.perception.modality_adapter)

                                model.cfg.perception = OmegaConf.create(perception_dict)
                                perception_cfg = model.cfg.perception
                                complete_perception_found = True
                                logger.info("Perception config successfully extracted from pretrained ASR model")
                            else:
                                logger.warning("Pretrained ASR model doesn't contain required preprocessor/encoder")
                        else:
                            logger.warning("Cannot find model_config.yaml in pretrained ASR .nemo file")
                except Exception as e:
                    logger.warning(f"Failed to extract perception config from pretrained ASR: {e}")
            else:
                logger.warning("No valid pretrained_asr path found in config")

        if not complete_perception_found:
            logger.error("Cannot find complete perception config in checkpoint or pretrained ASR")
            logger.error("Solutions:")
            logger.error("  1. Provide a checkpoint with complete hyperparameters")
            logger.error("  2. Ensure pretrained_asr path points to a valid .nemo file")
            logger.error("  3. Use a complete model config YAML file")
            raise RuntimeError("Cannot restore perception config - all sources incomplete")
    else:
        logger.info("Perception config is complete")

    try:
        from nemo.collections.speechlm2.modules import AudioPerceptionModule

        model.perception = AudioPerceptionModule(model.cfg.perception).train()
        logger.info("Perception module initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize perception module: {e}")
        logger.error("This might be due to:")
        logger.error("  1. Missing or invalid perception configuration")
        logger.error("  2. Incompatible ASR model format")
        logger.error("  3. Missing required fields in perception config")
        raise RuntimeError(f"Perception module initialization failed: {e}")

    # Load the averaged state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        logger.info("Using 'state_dict' from checkpoint")
    else:
        state_dict = checkpoint
        logger.info("Using checkpoint as raw state_dict")

    logger.info(f"State dict contains {len(state_dict)} parameters")

    # Load state dict with strict=False to handle potential mismatches
    logger.info("Loading averaged weights into model")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # Log any key mismatches
    if missing_keys:
        logger.warning(f"Missing {len(missing_keys)} keys when loading state_dict")
        logger.warning(f"First 5 missing keys: {missing_keys[:5]}")
    else:
        logger.info("No missing keys - all model parameters matched")

    if unexpected_keys:
        logger.warning(f"Found {len(unexpected_keys)} unexpected keys in state_dict")
        logger.warning(f"First 5 unexpected keys: {unexpected_keys[:5]}")
    else:
        logger.info("No unexpected keys - state_dict structure matches model")

    # Save as .nemo file
    logger.info(f"Saving .nemo file to: {output_nemo_path}")
    logger.info("Note: This may take several minutes for large models")

    try:
        model.save_to(output_nemo_path)
        logger.info(".nemo file saved successfully")
    except Exception as e:
        logger.error(f"Failed to save .nemo file: {e}")
        raise RuntimeError(f".nemo file creation failed: {e}")

    # Verify the saved file
    if os.path.exists(output_nemo_path):
        file_size = os.path.getsize(output_nemo_path)
        logger.info(f".nemo file size: {file_size / (1024**3):.2f} GB")
        logger.info("=" * 80)
        logger.info(f"SUCCESS: .nemo file created at {output_nemo_path}")
        logger.info("=" * 80)
    else:
        logger.error(f".nemo file was not created at {output_nemo_path}")
        raise RuntimeError(".nemo file verification failed")


def main():
    """
    Main function for SALM checkpoint averaging
    """
    parser = argparse.ArgumentParser(
        description="Weighted averaging of SALM model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Average top 20 checkpoints by WER and create .nemo file
  python scripts/checkpoint_averaging/salm_checkpoint_averaging.py \\
      --checkpoint_dir recipes/CanaryQwenASR/outputs/model/checkpoints \\
      --output_dir recipes/CanaryQwenASR/outputs/model \\
      --top_n 20 \\
      --output_nemo model_averaged.nemo \\
      --model_config config.yaml

  # Average specific checkpoint steps
  python scripts/checkpoint_averaging/salm_checkpoint_averaging.py \\
      --checkpoint_dir checkpoints/ \\
      --steps 10000 15000 20000 \\
      --output_name custom_averaged
        """
    )

    parser.add_argument(
        '--checkpoint_dir',
        required=True,
        help='Directory containing checkpoint files (.ckpt)',
    )
    parser.add_argument(
        '--output_dir',
        help='Directory to save the averaged checkpoint (default: checkpoint_dir)',
    )
    parser.add_argument(
        '--output_name',
        default='salm_averaged',
        help='Name of the output checkpoint file (default: salm_averaged)',
    )

    # Checkpoint selection options
    parser.add_argument(
        '--steps',
        nargs='+',
        type=int,
        help='List of specific checkpoint steps to average',
    )
    parser.add_argument(
        '--top_n',
        type=int,
        help='Select top N checkpoints with lowest val_wer',
    )

    # Weighting options
    parser.add_argument(
        '--weights',
        nargs='+',
        type=float,
        help='List of weights for each checkpoint. If not specified, weights based on WER will be used.',
    )

    # NeMo file options
    parser.add_argument(
        '--output_nemo',
        help='Path to save the .nemo file (optional)',
    )
    parser.add_argument(
        '--model_config',
        help='Path to model config YAML file (needed for .nemo creation if not in checkpoint)',
    )

    args = parser.parse_args()

    # Print banner
    logger.info("=" * 80)
    logger.info("SALM Checkpoint Averaging Script")
    logger.info("=" * 80)
    logger.info(f"Checkpoint directory: {args.checkpoint_dir}")
    logger.info(f"Output directory: {args.output_dir if args.output_dir else args.checkpoint_dir}")
    logger.info(f"Output name: {args.output_name}")

    # Validate arguments
    if args.steps and args.top_n:
        logger.error("Cannot specify both --steps and --top_n. Choose one method.")
        raise ValueError("Cannot specify both --steps and --top_n. Choose one method.")

    if not args.steps and not args.top_n:
        logger.warning("Neither --steps nor --top_n specified. Will use all checkpoints.")

    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.checkpoint_dir

    try:
        # Find checkpoints based on criteria
        logger.info("")
        logger.info("STEP 1: Finding and selecting checkpoints")
        logger.info("-" * 80)
        selected_checkpoints = find_checkpoints(
            args.checkpoint_dir,
            steps=args.steps,
            top_n=args.top_n
        )

        n = len(selected_checkpoints)
        logger.info(f"Selected {n} checkpoints for averaging")

        # Validate and prepare weights
        if args.weights:
            if len(args.weights) != n:
                logger.error(f"Number of weights ({len(args.weights)}) must match number of checkpoints ({n})")
                raise ValueError(f"Number of weights ({len(args.weights)}) must match number of checkpoints ({n})")
            weights = np.array(args.weights, dtype=np.float32)
            logger.info("Using user-provided weights")
        else:
            weights = None  # Will be calculated based on WER
            logger.info("Weights will be calculated based on validation WER")

        # Perform averaging
        logger.info("")
        logger.info("STEP 2: Performing weighted checkpoint averaging")
        logger.info("-" * 80)
        averaged_state_dict, original_hyperparameters = average_checkpoints(selected_checkpoints, weights)

        # Save averaged checkpoint
        logger.info("")
        logger.info("STEP 3: Saving averaged checkpoint")
        logger.info("-" * 80)
        checkpoint_info = {
            'steps': args.steps if args.steps else None,
            'top_n': args.top_n if args.top_n else None
        }
        checkpoint_path = save_averaged_checkpoint(
            averaged_state_dict,
            output_dir,
            args.output_name,
            checkpoint_info,
            original_hyperparameters
        )

        # Create .nemo file if requested
        if args.output_nemo:
            logger.info("")
            logger.info("STEP 4: Creating .nemo file")
            logger.info("-" * 80)
            create_nemo_from_checkpoint(
                checkpoint_path,
                args.output_nemo,
                args.model_config
            )
        else:
            logger.info("")
            logger.info("Skipping .nemo file creation (--output_nemo not specified)")

        # Final summary
        logger.info("")
        logger.info("=" * 80)
        logger.info("CHECKPOINT AVERAGING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Averaged checkpoint: {checkpoint_path}")
        if args.output_nemo:
            logger.info(f".nemo file: {args.output_nemo}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("CHECKPOINT AVERAGING FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error("=" * 80)
        raise


if __name__ == '__main__':
    main()