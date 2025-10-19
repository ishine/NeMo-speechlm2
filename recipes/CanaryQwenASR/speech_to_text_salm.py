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
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

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


class SALMProgressBar(TQDMProgressBar):
    """
    Custom progress bar for SALM training with step-based display and comprehensive metrics.

    Displays:
        - Step progress (step/max_steps) instead of epoch-based
        - Training metrics: train_loss (4 decimals), lr (scientific), bs (integer)
        - GPU memory usage
        - Step timing

    Example output:
        Training: 23%|████| 1234/5000 [02:45<08:22, 3.5it/s, train_loss=2.1234, lr=3.00e-04, bs=4, GPU_Mem=45.23GB]
    """

    # Display format constants
    TRAIN_BAR_FORMAT = "Training: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    VAL_BAR_FORMAT = "Validation: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]"

    # Metric format specifications
    METRIC_FORMATS = {
        'train_loss': lambda v: f"{v:.4f}",      # 4 decimals
        'lr': lambda v: f"{v:.2e}",              # Scientific notation
        'bs': lambda v: f"{int(v)}",             # Integer
        'step_time': lambda v: f"{v:.2f}s",      # 2 decimals + unit
    }

    # Keys to remove from display
    EXCLUDE_KEYS = {'v_num'}

    # Key renames
    KEY_RENAMES = {
        'train_step_timing in s': 'step_time',
        'batch size': 'bs',
    }

    def init_train_tqdm(self):
        """Initialize training progress bar with custom format."""
        bar = super().init_train_tqdm()
        bar.bar_format = self.TRAIN_BAR_FORMAT
        return bar

    def init_validation_tqdm(self):
        """Initialize validation progress bar with custom format."""
        bar = super().init_validation_tqdm()
        bar.bar_format = self.VAL_BAR_FORMAT
        return bar

    def get_metrics(self, trainer, pl_module):
        """
        Collect and format metrics for progress bar display.

        Returns:
            dict: Formatted metrics ready for display
        """
        items = super().get_metrics(trainer, pl_module)

        # Remove excluded keys
        for key in self.EXCLUDE_KEYS:
            items.pop(key, None)

        # Rename keys
        for old_key, new_key in self.KEY_RENAMES.items():
            if old_key in items:
                items[new_key] = items.pop(old_key)

        # Format specific metrics
        for metric_key, format_fn in self.METRIC_FORMATS.items():
            if metric_key in items:
                items[metric_key] = format_fn(self._extract_value(items[metric_key]))

        # Format generic loss/lr metrics (backward compatibility)
        self._format_loss_lr_metrics(items)

        # Add GPU memory usage
        self._add_gpu_memory(items)

        return items

    # Helper methods
    @staticmethod
    def _extract_value(value):
        """Extract numeric value from Tensor or scalar."""
        return value.item() if isinstance(value, torch.Tensor) else value

    def _format_loss_lr_metrics(self, items):
        """Format loss and learning rate metrics with scientific notation."""
        for key in list(items.keys()):
            if key not in self.METRIC_FORMATS:
                if key.startswith(("loss", "lr")) or key.endswith(("loss", "lr")):
                    try:
                        value = self._extract_value(items[key])
                        items[key] = f"{value:.2e}"
                    except (ValueError, TypeError, AttributeError):
                        pass

    @staticmethod
    def _add_gpu_memory(items):
        """Add GPU memory usage to metrics if CUDA is available."""
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            mem_used_gb = (total - free) / (1024 ** 3)
            items["GPU_Mem"] = f"{mem_used_gb:.2f}GB"


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
    trainer = Trainer(**trainer_cfg, callbacks=[SALMProgressBar()])

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
                if isinstance(shar_item, (list, ListConfig)) and len(shar_item) >= 1:
                    shar_dir = shar_item[0]
                    weight = shar_item[1] if len(shar_item) >= 2 else 1.0
                    language = shar_item[2] if len(shar_item) >= 3 else 'en'
                    metric = shar_item[3] if len(shar_item) >= 4 else 'wer'

                    input_cfg_list.append({
                        'type': 'lhotse_as_conversation',
                        'shar_path': shar_dir,
                        'weight': weight,
                        'audio_locator_tag': cfg.model.audio_locator_tag,
                        'token_equivalent_duration': cfg.data.train_ds.token_equivalent_duration,
                        'prompt_format': cfg.model.prompt_format,  # Add prompt_format here
                        'tags': {
                            'context': context_prompt,
                            'lang': language,
                            'metric': metric,
                        }
                    })

            # Only update if we have valid configs
            if input_cfg_list:
                if rank == 0:
                    logging.info(f"Converted {len(input_cfg_list)} shar_path entries to input_cfg format")
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