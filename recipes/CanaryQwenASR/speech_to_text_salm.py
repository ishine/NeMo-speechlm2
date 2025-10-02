#!/usr/bin/env python3
# Training script for SALM Canary-Qwen-2.5B ASR model
# Based on speechlm2 SALM recipe with lhotse shar format

import os
import socket
import sys
import logging

# Add NeMo path to sys.path
nemo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if nemo_path not in sys.path:
    sys.path.insert(0, nemo_path)

# Configure logging to reduce verbosity
logging.getLogger("nemo.core.classes.common").setLevel(logging.WARNING)
logging.getLogger("nemo.collections.asr").setLevel(logging.WARNING)

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import TQDMProgressBar
from omegaconf import DictConfig, OmegaConf, open_dict

# Import SALM components directly to avoid peft dependency issue with duplex models
from nemo.collections.speechlm2.models.salm import SALM
from nemo.collections.speechlm2.data.salm_dataset import SALMDataset
from nemo.collections.speechlm2.data.datamodule import DataModule
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg


def get_node_count():
    """Get number of nodes in distributed training using MPI."""
    try:
        from mpi4py import MPI
    except ImportError:
        logging.warning("mpi4py not installed, assuming single node")
        return 1

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Get the hostname of the current process
    hostname = socket.gethostname()

    # Gather all hostnames at the root process
    all_hostnames = comm.gather(hostname, root=0)

    # Root process computes number of unique hostnames (nodes)
    if rank == 0:
        unique_nodes = set(all_hostnames)
        num_nodes = len(unique_nodes)
    else:
        num_nodes = None

    # Broadcast the node count to all processes
    num_nodes = comm.bcast(num_nodes, root=0)

    return num_nodes


BATCH_SIZE_KEY = 'batch size'

class GPUUsageTQDMProgressBar(TQDMProgressBar):
    """Custom progress bar showing GPU memory usage and formatted metrics."""

    def init_train_tqdm(self):
        """Override bar format to not have 's/it'."""
        self.bar = super().init_train_tqdm()
        self.bar.bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"
        return self.bar

    def get_metrics(self, trainer, pl_module):
        """Add GPU memory usage and format loss/lr values in scientific notation."""
        items = super().get_metrics(trainer, pl_module)

        # Format loss and learning rate values
        for key in list(items.keys()):
            if key.startswith("loss") or key.endswith("loss") or key.startswith("lr") or key.endswith("lr"):
                try:
                    value = items[key]
                    if isinstance(value, (float, int)) or (isinstance(value, torch.Tensor) and value.numel() == 1):
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        items[key] = f"{value:.2e}"
                except Exception:
                    pass

        # Clean up display
        items.pop("v_num", None)

        # Rename step timing
        STEP_TIME_STR = 'train_step_timing in s'
        if STEP_TIME_STR in items:
            step_time = items.get(STEP_TIME_STR)
            items.pop(STEP_TIME_STR)
            items['step_time'] = step_time

        # Format batch size
        if BATCH_SIZE_KEY in items:
            items[BATCH_SIZE_KEY] = f"{int(items[BATCH_SIZE_KEY])}"

        # Add GPU memory usage
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            mem_used_GB = (total - free) / 1024**3
            items["GPU_Mem"] = f"{mem_used_GB:.2f}GB"

        return items


@hydra_runner(
    config_path="configs/",
    config_name="salm_canary_qwen_2.5b",
)
def main(cfg: DictConfig):
    """Main training function for SALM model."""

    # Only show main logs on rank 0
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Configure logging based on rank
    if rank != 0:
        # Reduce logging on non-primary ranks
        logging.getLogger().setLevel(logging.WARNING)

    # Log configuration (only in debug mode to reduce verbosity)
    logging.debug(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    if rank == 0:
        logging.info("=" * 80)
        logging.info("SALM Training - Canary-Qwen ASR Model")
        logging.info("=" * 80)

    # Configure trainer
    trainer_cfg = resolve_trainer_cfg(cfg.trainer)

    # Auto-detect number of nodes if not specified
    if cfg.trainer.num_nodes == -1:
        nodes = get_node_count()
        trainer_cfg['num_nodes'] = nodes
        logging.info(f'Auto-detected {nodes} node(s)')
    else:
        nodes = cfg.trainer.num_nodes
        trainer_cfg['num_nodes'] = nodes

    # Auto-detect number of GPUs if not specified
    if cfg.trainer.devices == -1:
        cfg.trainer.devices = torch.cuda.device_count()
        trainer_cfg['devices'] = cfg.trainer.devices

    if rank == 0:
        logging.info(f"Training Configuration: {cfg.trainer.devices} GPU(s) on {nodes} node(s)")

    # Initialize trainer with custom progress bar
    trainer = Trainer(**trainer_cfg, callbacks=[GPUUsageTQDMProgressBar()])

    # Setup experiment manager
    exp_manager(trainer, cfg.get("exp_manager", None))

    # Initialize model
    if rank == 0:
        logging.info("Initializing SALM model...")
    with trainer.init_module():
        model = SALM(OmegaConf.to_container(cfg.model, resolve=True))

    if rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Model initialized successfully")
        logging.info(f"  Total parameters: {total_params:,}")
        logging.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        logging.info(f"  LLM Tokenizer: {model.tokenizer.__class__.__name__} with {model.tokenizer.vocab_size:,} tokens")
        logging.info(f"  Audio locator tag: '{model.audio_locator_tag}' (ID: {model.audio_locator_tag_id})")

    # Setup data module
    if rank == 0:
        logging.info("Setting up data module...")

    # Convert shar_path to input_cfg format if needed (for backward compatibility)
    with open_dict(cfg.data):
        if "train_ds" in cfg.data and "shar_path" in cfg.data.train_ds:
            shar_paths = cfg.data.train_ds.shar_path

            # Get context prompt from config, or use official default
            context_prompt = cfg.data.train_ds.get('asr_context_prompt', 'Transcribe the following: ')

            # Create input_cfg list
            input_cfg_list = []
            for shar_item in shar_paths:
                if isinstance(shar_item, list) and len(shar_item) >= 1:
                    shar_dir = shar_item[0]
                    weight = shar_item[1] if len(shar_item) >= 2 else 1.0

                    input_cfg_list.append({
                        'type': 'lhotse_as_conversation',
                        'shar_path': shar_dir,
                        'weight': weight,
                        'audio_locator_tag': cfg.model.audio_locator_tag,
                        'token_equivalent_duration': cfg.data.train_ds.token_equivalent_duration,
                        'prompt_format': cfg.model.prompt_format,  # Add prompt_format here
                        'tags': {
                            'context': context_prompt,
                        }
                    })

            # Only update if we have valid configs
            if input_cfg_list:
                cfg.data.train_ds.input_cfg = input_cfg_list
                # Remove the old shar_path configuration
                del cfg.data.train_ds.shar_path
            else:
                logging.warning("No valid shar_path entries found, keeping original config")

            cfg.data.train_ds.use_lhotse = True

    # Create dataset and datamodule
    # Fix the prompt_format interpolation issue in all dataset configs
    with open_dict(cfg.data):
        # Fix training dataset prompt_format if it contains interpolation
        if 'train_ds' in cfg.data and 'prompt_format' in cfg.data.train_ds:
            cfg.data.train_ds.prompt_format = cfg.model.prompt_format

        # Replace ${model.prompt_format} with actual value for validation datasets
        if 'validation_ds' in cfg.data:
            if 'prompt_format' in cfg.data.validation_ds:
                cfg.data.validation_ds.prompt_format = cfg.model.prompt_format
            # Also update for each individual dataset if needed
            if 'datasets' in cfg.data.validation_ds:
                for dataset_name in cfg.data.validation_ds.datasets:
                    if 'prompt_format' in cfg.data.validation_ds.datasets[dataset_name]:
                        cfg.data.validation_ds.datasets[dataset_name].prompt_format = cfg.model.prompt_format

        # Same for test datasets if they exist
        if 'test_ds' in cfg.data:
            if 'prompt_format' in cfg.data.test_ds:
                cfg.data.test_ds.prompt_format = cfg.model.prompt_format
            if 'datasets' in cfg.data.test_ds:
                for dataset_name in cfg.data.test_ds.datasets:
                    if 'prompt_format' in cfg.data.test_ds.datasets[dataset_name]:
                        cfg.data.test_ds.datasets[dataset_name].prompt_format = cfg.model.prompt_format

    data_cfg = cfg.data

    # Get prompt format from model config
    prompt_format = None
    if hasattr(cfg.model, 'prompt_format'):
        from nemo.collections.common.prompts import PromptFormatter
        prompt_format = PromptFormatter.resolve(cfg.model.prompt_format)(model.tokenizer)

    dataset = SALMDataset(tokenizer=model.tokenizer, prompt_format=prompt_format)
    datamodule = DataModule(data_cfg, tokenizer=model.tokenizer, dataset=dataset)

    # Start training
    if rank == 0:
        logging.info("-" * 80)
        logging.info("Starting training...")
        logging.info("-" * 80)
    trainer.fit(model, datamodule)

    # Run testing if test dataset is provided
    if hasattr(cfg.data, 'test_ds') and cfg.data.test_ds is not None:
        if rank == 0:
            logging.info("Running test evaluation...")
        trainer.test(model, datamodule)


if __name__ == '__main__':
    main()