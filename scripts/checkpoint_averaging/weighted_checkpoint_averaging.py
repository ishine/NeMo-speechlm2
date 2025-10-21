"""
Weighted checkpoint averaging for NeMo ASR models (traditional CTC-based models).

NOTE: For SALM (Speech-Augmented Language Model) checkpoints, use salm_checkpoint_averaging.py instead.
      This script is designed for traditional ASR models like EncDecCTCModelBPE.

Example: python weighted_checkpoint_averaging.py \
             --checkpoint_dir=<folder containing checkpoints> \
             --output_dir=<directory to save the averaged checkpoint> \
             --output_name=<name of the output checkpoint> \
             --steps <list of checkpoint steps to average> \
             --weights <list of weights for each checkpoint> \
             --top_n <number of checkpoints to select based on best val_wer> \
             --input_nemo <path to input .nemo file for updating with averaged checkpoint> \
             --output_nemo <path to save the updated .nemo file>

You can either:
1. Provide specific steps to average: --steps 14000 20500 ...
2. Select top N checkpoints by val_wer: --top_n 3
"""

import argparse
import logging
import os
import re
import shutil
import torch
import numpy as np
import zarr
from pathlib import Path
import tempfile
import importlib.util
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weighted_checkpoint_averaging")

def extract_checkpoint_info(filename):
    """
    Extract step number and validation WER from checkpoint filename
   
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
    wer_match = re.search(r'val_wer=(\d+\.\d+)', filename)
    if wer_match:
        info['val_wer'] = float(wer_match.group(1))
    else:
        info['val_wer'] = float('inf')  # Default if no WER found
   
    # This field is no longer needed since we filter out -last.ckpt files
    info['is_last'] = False
       
    return info

def find_checkpoints(checkpoint_dir, steps=None, top_n=None):
    """
    Find checkpoints in the given directory based on criteria
   
    Args:
        checkpoint_dir (str): Directory containing checkpoints
        steps (list): List of specific step numbers to select
        top_n (int): Number of checkpoints to select based on lowest val_wer
       
    Returns:
        list: List of checkpoint filenames matching the criteria
    """
    all_checkpoints = []
   
    # Gather all checkpoint files, excluding last checkpoints
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.ckpt') and not file.endswith('-last.ckpt'):
            checkpoint_path = os.path.join(checkpoint_dir, file)
            info = extract_checkpoint_info(file)
            all_checkpoints.append({
                'filename': file,
                'path': checkpoint_path,
                'step': info['step'],
                'val_wer': info['val_wer'],
                'is_last': info['is_last']
            })
   
    if not all_checkpoints:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
   
    # Filter by specific steps if provided
    if steps:
        filtered_checkpoints = []
        for step in steps:
            matching = [ckpt for ckpt in all_checkpoints if ckpt['step'] == step]
            if matching:
                filtered_checkpoints.extend(matching)
            else:
                logger.warning(f"No checkpoint found for step {step}")
               
        if not filtered_checkpoints:
            raise ValueError("None of the specified steps matched any checkpoints")
        return filtered_checkpoints
   
    # Select top N checkpoints with lowest val_wer
    elif top_n:
        sorted_checkpoints = sorted(all_checkpoints, key=lambda x: x['val_wer'])
        return sorted_checkpoints[:top_n]
   
    # Return all checkpoints if no filter specified
    else:
        return all_checkpoints

def load_nemo_asr_model(nemo_file_path):
    """
    Dynamically load NeMo ASR model class and restore the model
   
    Args:
        nemo_file_path (str): Path to the .nemo file
       
    Returns:
        model: The loaded NeMo model
    """
    try:
        # Try to import the NeMo ASR models module
        if importlib.util.find_spec("nemo.collections.asr.models") is None:
            raise ImportError("NeMo ASR module not found. Please install NeMo first.")
       
        import nemo.collections.asr.models as nemo_asr
       
        # Check if the module has ASR model classes
        model_class = nemo_asr.EncDecCTCModelBPE
        if not hasattr(nemo_asr, "EncDecCTCModelBPE"):
            # Try alternative ways
            if hasattr(nemo_asr, "speech_to_text"):
                model_class = nemo_asr.speech_to_text.models.EncDecCTCModelBPE
       
        logger.info(f"Restoring NeMo model from {nemo_file_path}")
        model = model_class.restore_from(nemo_file_path)
        return model
       
    except ImportError as e:
        logger.error(f"Failed to import NeMo modules: {e}")
        logger.error("Make sure NeMo is installed properly: pip install nemo_toolkit[all]")
        raise
    except Exception as e:
        logger.error(f"Failed to load NeMo model: {e}")
        raise

def update_nemo_with_averaged_checkpoint(input_nemo, checkpoint_path, output_nemo):
    """
    Update a .nemo file with a new averaged checkpoint
   
    Args:
        input_nemo (str): Path to the input .nemo file
        checkpoint_path (str): Path to the averaged checkpoint file
        output_nemo (str): Path to save the updated .nemo file
    """
    try:
        # Import required NeMo modules
        logger.info(f"Loading NeMo model from {input_nemo}")
        model = load_nemo_asr_model(input_nemo)
       
        # Load the new checkpoint
        logger.info(f"Loading averaged checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
       
        # Extract state dict
        if "state_dict" in ckpt:
            new_state_dict = ckpt["state_dict"]
        else:
            new_state_dict = ckpt
           
        # Update model with new weights
        logger.info("Updating model with averaged weights")
        model.load_state_dict(new_state_dict, strict=False)
       
        # Save the updated model
        logger.info(f"Saving updated model to {output_nemo}")
        model.save_to(output_nemo)
       
        logger.info("Successfully updated .nemo file with averaged checkpoint")
       
    except Exception as e:
        logger.error(f"Failed to update .nemo file: {e}")
        raise

def main():
    """
    Main function for weighted checkpoint averaging
    """
    parser = argparse.ArgumentParser()
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
        default='weighted_averaged',
        help='Name of the output checkpoint file',
    )
    # list of checkpoint steps to average
    parser.add_argument(
        '--steps',
        nargs='+',
        type=int,
        help='List of specific checkpoint steps to average',
    )
    # list of weights for each checkpoint
    parser.add_argument(
        '--weights',
        nargs='+',
        type=float,
        help='List of weights for each checkpoint. If not specified, equal weights will be used.',
    )
    # top N checkpoints based on val_wer
    parser.add_argument(
        '--top_n',
        type=int,
        help='Select top N checkpoints with lowest val_wer',
    )
    # NeMo file options
    parser.add_argument(
        '--input_nemo',
        help='Path to input .nemo file to update with the averaged checkpoint',
    )
    parser.add_argument(
        '--output_nemo',
        help='Path to save the updated .nemo file',
    )

    args = parser.parse_args()
   
    # Validate arguments
    if args.steps and args.top_n:
        raise ValueError("Cannot specify both --steps and --top_n. Choose one method.")
   
    if args.input_nemo and not args.output_nemo:
        raise ValueError("--output_nemo must be specified when --input_nemo is provided")
       
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.checkpoint_dir
    os.makedirs(output_dir, exist_ok=True)
   
    # Find checkpoints based on criteria
    selected_checkpoints = find_checkpoints(
        args.checkpoint_dir,
        steps=args.steps,
        top_n=args.top_n
    )
   
    n = len(selected_checkpoints)
    logger.info(f"Found {n} checkpoints to average:")
    for i, ckpt in enumerate(selected_checkpoints):
        logger.info(f"  {i+1}. {ckpt['filename']} - Step: {ckpt['step']}, WER: {ckpt['val_wer']}")
   
    # Validate and normalize weights
    if args.weights:
        if len(args.weights) != n:
            raise ValueError(f"Number of weights ({len(args.weights)}) must match number of checkpoints ({n})")
        weights = np.array(args.weights, dtype=np.float32)
    else:
        # If no weights provided, calculate weights based on validation WER
        # Better (lower) WER gets higher weight
        if args.top_n or all(ckpt['val_wer'] < float('inf') for ckpt in selected_checkpoints):
            # Get WER values and invert them (lower WER = higher weight)
            wer_values = np.array([ckpt['val_wer'] for ckpt in selected_checkpoints])
            # Calculate inverse WER (higher = better)
            inverse_wer = 1.0 / wer_values
            # Normalize to sum to 1.0
            weights = inverse_wer / inverse_wer.sum()
            logger.info("Calculated weights based on validation WER")
        else:
            # Equal weights if WER not available
            weights = np.ones(n, dtype=np.float32) / n
            logger.info("Using equal weights for all checkpoints")
   
    # Normalize weights to sum to 1.0
    weights = weights / weights.sum()
   
    for i, (ckpt, weight) in enumerate(zip(selected_checkpoints, weights)):
        logger.info(f"  {i+1}. {ckpt['filename']} - Weight: {weight:.4f}")
   
    # Create state dict for averaged model
    averaged_state_dict = {}
   
    # Process each checkpoint with its corresponding weight
    for i, (ckpt, weight) in enumerate(zip(selected_checkpoints, weights)):
        logger.info(f"Processing checkpoint {i+1}/{n}: {ckpt['filename']} with weight {weight:.4f}")
       
        # Load the checkpoint
        checkpoint = torch.load(ckpt['path'], map_location='cpu')
       
        # Extract state dict
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
       
        # Initialize or update the averaged state dict
        if i == 0:
            # For first checkpoint, initialize the averaged state dict
            for key, value in state_dict.items():
                averaged_state_dict[key] = value.clone() * weight
        else:
            # For subsequent checkpoints, add weighted values
            for key, value in state_dict.items():
                if key in averaged_state_dict:
                    averaged_state_dict[key] += value * weight
                else:
                    logger.warning(f"Key {key} found in checkpoint {i+1} but not in first checkpoint, skipping")
   
    # Create output checkpoint path
    if args.steps:
        steps_str = '_'.join(str(step) for step in args.steps)
        output_filename = f"{args.output_name}_steps_{steps_str}.ckpt"
    elif args.top_n:
        output_filename = f"{args.output_name}_top_{args.top_n}.ckpt"
    else:
        output_filename = f"{args.output_name}.ckpt"
   
    output_path = os.path.join(output_dir, output_filename)
   
    # Save the averaged checkpoint
    logger.info(f"Saving averaged checkpoint to {output_path}")
    torch.save(averaged_state_dict, output_path)
   
    # Update .nemo file if requested
    if args.input_nemo:
        if not args.output_nemo:
            raise ValueError("--output_nemo must be specified when --input_nemo is provided")
           
        logger.info(f"Updating .nemo file {args.input_nemo} with averaged checkpoint")
        update_nemo_with_averaged_checkpoint(
            args.input_nemo,
            output_path,
            args.output_nemo
        )
   
    logger.info("Checkpoint averaging completed successfully")

if __name__ == '__main__':
    main()
