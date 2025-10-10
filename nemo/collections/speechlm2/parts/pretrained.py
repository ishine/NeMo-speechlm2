# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from contextlib import contextmanager
from pathlib import Path
import logging

import torch
from omegaconf import open_dict
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM

from nemo.collections.asr.models import ASRModel
from nemo.collections.speechlm2.modules import AudioPerceptionModule

from nemo.collections.speechlm2.parts.precision import fp32_precision
from nemo.collections.tts.models import AudioCodecModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pretrained_nemo(cls, model_path_or_name: str):
    """
    Load pretrained NeMo 1.0 model (inheriting from ModelPT). Works with ASR, TTS, codec models.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.

    Enhanced with better error handling for incompatible checkpoints.
    """
    if Path(model_path_or_name).exists() and model_path_or_name.endswith(".nemo"):
        try:
            # Try standard loading first
            return cls.restore_from(model_path_or_name)
        except (ModuleNotFoundError, TypeError) as e:
            # Suppress verbose error logging for expected failures
            logger.debug(f"Standard loading failed (expected): {e}")

            # Try with specific model classes for ASR
            if cls == ASRModel:
                # Try known ASR model classes
                from nemo.collections.asr.models import (
                    EncDecCTCModelBPE,
                    EncDecHybridRNNTCTCBPEModel,
                    EncDecRNNTBPEModel,
                )
                # Also try the custom hybrid model
                from nemo.collections.asr.models.hybrid_transformer_transcribe_ctc_bpe_models import (
                    HybridTransformerTranscribeCTCBPEModel,
                )

                for model_class in [
                    EncDecHybridRNNTCTCBPEModel,
                    HybridTransformerTranscribeCTCBPEModel,
                    EncDecCTCModelBPE,
                    EncDecRNNTBPEModel,
                ]:
                    try:
                        logger.debug(f"Trying to load with {model_class.__name__}...")
                        model = model_class.restore_from(model_path_or_name)
                        logger.info(f"✓ Successfully loaded ASR model with {model_class.__name__}")
                        return model
                    except Exception as sub_e:
                        # Suppress verbose error messages from failed attempts
                        logger.debug(f"Failed with {model_class.__name__}: {sub_e}")
                        continue

                # If all specific classes fail, raise the original error
                raise RuntimeError(
                    f"Could not load ASR model from {model_path_or_name}. "
                    f"The checkpoint may be incompatible with the current NeMo version. "
                    f"Consider using an official pretrained model like 'nvidia/canary-1b-v2' instead. "
                    f"Original error: {e}"
                )
            else:
                # For non-ASR models, raise the original error
                raise e
    else:
        return cls.from_pretrained(model_path_or_name)


def load_custom_asr_encoder_only(nemo_path: str):
    """
    Load only the encoder and preprocessor from a custom ASR .nemo checkpoint.
    This is useful when the full model class is not available but the encoder structure is compatible.
    """
    import tarfile
    import tempfile
    import torch
    from omegaconf import OmegaConf
    from pathlib import Path

    logger.debug(f"Extracting encoder from: {nemo_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract the .nemo file (which is a tar archive)
        with tarfile.open(nemo_path, 'r') as tar:
            tar.extractall(tmpdir)

        # Load the config
        config_path = Path(tmpdir) / "model_config.yaml"
        if not config_path.exists():
            raise RuntimeError(f"Could not find model_config.yaml in {nemo_path}")

        config = OmegaConf.load(config_path)

        # Load the state dict
        weights_path = Path(tmpdir) / "model_weights.ckpt"
        if not weights_path.exists():
            raise RuntimeError(f"Could not find model_weights.ckpt in {nemo_path}")

        state_dict = torch.load(weights_path, map_location='cpu')

        # Create a simple namespace object to hold the config and state dict
        # This mimics the structure expected by setup_speech_encoder
        class CustomASRModel:
            def __init__(self, cfg, state_dict):
                self.cfg = cfg
                self._state_dict = state_dict

            def eval(self):
                return self

            def state_dict(self):
                return self._state_dict

        # Return the custom model object
        return CustomASRModel(config, state_dict)


def _load_qwen3_as_qwen2(model_path_or_name: str, dtype=torch.float32):
    """
    Load Qwen3 model using Qwen2 compatibility mode.

    Qwen3 and Qwen2 share nearly identical architectures, so we can load Qwen3 weights
    into a Qwen2 model structure when the transformers library doesn't support Qwen3.

    This creates a temporary modified config.json with model_type changed from 'qwen3' to 'qwen2',
    then loads the model with that modified config.

    Args:
        model_path_or_name: Path to Qwen3 model directory
        dtype: Model dtype for loading

    Returns:
        Loaded model with Qwen3 weights in Qwen2 architecture

    Raises:
        RuntimeError: If the model is not actually Qwen3 or if loading fails
    """
    import json
    import tempfile
    import shutil
    from pathlib import Path

    model_path = Path(model_path_or_name)

    # Verify this is a local path
    if not model_path.exists():
        raise RuntimeError(
            f"Qwen2 compatibility mode requires a local model directory, but path does not exist: {model_path}"
        )

    # Load original config
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Verify this is actually Qwen3
    original_model_type = config.get('model_type')
    if original_model_type != 'qwen3':
        raise RuntimeError(
            f"Expected model_type='qwen3' but got '{original_model_type}'. "
            f"Qwen2 compatibility mode only works for Qwen3 models."
        )

    logger.info(f"  Model type: {original_model_type} → qwen2 (compatibility mode)")

    # Create temporary directory with modified config
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Copy all files to temp directory (preserving structure)
        logger.debug(f"  Copying model files to temporary directory...")
        for item in model_path.iterdir():
            if item.is_file():
                shutil.copy2(item, tmpdir_path / item.name)
            elif item.is_dir():
                shutil.copytree(item, tmpdir_path / item.name)

        # Modify config.json: qwen3 → qwen2
        config['model_type'] = 'qwen2'
        modified_config_path = tmpdir_path / "config.json"
        with open(modified_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        logger.info(f"  Loading with Qwen2 architecture (compatible with Qwen3)...")

        # Load model with modified config
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(tmpdir_path),
                dtype=dtype,
                local_files_only=True  # Don't try to download anything
            )
            logger.info("  ✓ Successfully loaded Qwen3 model using Qwen2 compatibility mode")
            return model

        except Exception as e:
            logger.error(f"  Failed to load model in Qwen2 compatibility mode: {e}")
            raise RuntimeError(
                f"Could not load Qwen3 model even with Qwen2 compatibility. "
                f"Please upgrade transformers: pip install --upgrade 'transformers>=4.37.0'"
            ) from e


def load_pretrained_hf(model_path_or_name: str, pretrained_weights: bool = True, dtype=torch.float32):
    """
    Load pretrained HuggingFace AutoModelForCausalLM with Qwen3 compatibility.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.

    Enhanced with Qwen3 support:
    - Tries trust_remote_code first for newer architectures
    - Falls back to Qwen2 mapping for Qwen3 models (architecturally compatible)
    - Handles dtype parameter compatibility across transformers versions
    - Provides clear error messages for version issues
    """
    if pretrained_weights:
        try:
            # First attempt: trust_remote_code with dtype for custom/newer models
            return AutoModelForCausalLM.from_pretrained(
                model_path_or_name,
                dtype=dtype,
                trust_remote_code=True
            )
        except TypeError as type_err:
            # Handle dtype parameter not supported in older transformers versions
            if "dtype" in str(type_err) or "unexpected keyword argument" in str(type_err):
                logger.warning(
                    f"dtype parameter not supported in current transformers version.\n"
                    f"  Retrying without dtype parameter (will use model's default dtype)..."
                )
                try:
                    return AutoModelForCausalLM.from_pretrained(
                        model_path_or_name,
                        trust_remote_code=True
                    )
                except Exception as retry_error:
                    logger.error(f"Retry without dtype failed: {retry_error}")
                    logger.error(
                        "\n" + "="*80 + "\n"
                        "SOLUTION: Upgrade transformers to support dtype parameter\n"
                        "="*80 + "\n"
                        "Your transformers library is too old to support dtype parameter for Qwen3.\n"
                        "\n"
                        "Fix by running on your GPU server:\n"
                        "  pip install --upgrade 'transformers>=4.55.0'\n"
                        "\n"
                        "Or if using conda:\n"
                        "  conda install -c conda-forge 'transformers>=4.55.0'\n"
                        "\n"
                        "After upgrade, restart your script.\n"
                        "="*80
                    )
                    raise RuntimeError(
                        f"Cannot load model: transformers library too old (requires >=4.55.0 for full Qwen3 support)\n"
                        f"Original error: {type_err}"
                    ) from type_err
            else:
                # Different TypeError, re-raise
                raise
        except (KeyError, ValueError) as e:
            error_str = str(e).lower()

            # Check if this is a Qwen3 compatibility issue
            if 'qwen3' in error_str and 'does not recognize' in error_str:
                logger.warning(
                    f"Qwen3 model type not recognized by transformers library.\n"
                    f"  Attempting Qwen2 compatibility mode (architectures are nearly identical)..."
                )

                # Try Qwen2 compatibility mode
                try:
                    return _load_qwen3_as_qwen2(model_path_or_name, dtype)
                except Exception as qwen2_error:
                    logger.error(f"Qwen2 compatibility mode failed: {qwen2_error}")
                    logger.error(
                        "\n" + "="*80 + "\n"
                        "SOLUTION: Upgrade transformers to support Qwen3\n"
                        "="*80 + "\n"
                        "Your transformers library is too old to support Qwen3 models.\n"
                        "\n"
                        "Fix by running on your GPU server:\n"
                        "  pip install --upgrade 'transformers>=4.55.0'\n"
                        "\n"
                        "Or if using conda:\n"
                        "  conda install -c conda-forge 'transformers>=4.55.0'\n"
                        "\n"
                        "After upgrade, restart your script.\n"
                        "="*80
                    )
                    raise RuntimeError(
                        f"Cannot load Qwen3 model: transformers library too old (requires >=4.55.0)\n"
                        f"Original error: {e}"
                    ) from e
            else:
                # Different error, re-raise
                raise
    else:
        try:
            config = AutoConfig.from_pretrained(model_path_or_name, trust_remote_code=True)
            # Try with dtype first, fall back to without dtype if not supported
            try:
                return AutoModelForCausalLM.from_config(config, dtype=dtype)
            except TypeError as type_err:
                if "dtype" in str(type_err) or "unexpected keyword argument" in str(type_err):
                    logger.warning(
                        f"dtype parameter not supported for from_config in current transformers version.\n"
                        f"  Using model's default dtype instead..."
                    )
                    return AutoModelForCausalLM.from_config(config)
                else:
                    raise
        except (KeyError, ValueError) as e:
            if 'qwen3' in str(e).lower():
                logger.error(
                    "Qwen3 model type not supported. Please upgrade transformers:\n"
                    "  pip install --upgrade 'transformers>=4.55.0'"
                )
            raise


@contextmanager
def move_embedding(model):
    """Temporarily restores the embedding layer into HF LLM. Supports LoRA models."""
    if isinstance(model.llm, PeftModel):
        model.llm.base_model.model.model.embed_tokens = model.embed_tokens
    else:
        model.llm.model.embed_tokens = model.embed_tokens
    yield
    if isinstance(model.llm, PeftModel):
        del model.llm.base_model.model.model.embed_tokens
    else:
        del model.llm.model.embed_tokens


def setup_audio_codec(model: torch.nn.Module):
    """
    Sets up an ``AudioCodecModel``, initializing it from pretrained weights.
    The result is assigned to ``model.audio_codec`` attribute.

    Includes a workaround for PTL auto-downcasting the codec model to bf16 with bf16-true precision.
    """
    if hasattr(model, "audio_codec") and next(model.audio_codec.parameters()).dtype == torch.float:
        return  # skip if already set up and has the right dtype
    with fp32_precision():
        model.audio_codec = load_pretrained_nemo(AudioCodecModel, model.cfg.pretrained_audio_codec).eval()
    for p in model.audio_codec.parameters():
        p.requires_grad = False
    del model.audio_codec.discriminator  # free up some memory


def setup_speech_encoder(model: torch.nn.Module, pretrained_weights: bool = True):
    """
    Sets up an ``AudioPerceptionModule``, initializing its ``encoder`` and ``preprocessor``
    with a pretrained NeMo ``ASRModel``.
    The result is assigned to ``model.perception`` attribute and is trainable.

    Enhanced with better error handling and fallback options.
    """
    if pretrained_weights:
        # Temporarily suppress ASR model's tokenizer initialization log to avoid confusion
        # (SALM uses LLM's tokenizer, not ASR's tokenizer)
        asr_logger = logging.getLogger("nemo.collections.asr")
        original_level = asr_logger.level
        asr_logger.setLevel(logging.WARNING)

        try:
            asr = load_pretrained_nemo(ASRModel, model.cfg.pretrained_asr).eval()
        except RuntimeError as e:
            if "abstract class ASRModel" in str(e) or "abstract methods" in str(e):
                # This is likely a custom model class that's not available
                # Try to load just the encoder and preprocessor
                logger.debug(f"Expected error - model class not available: {e}")
                logger.info("Loading encoder from custom ASR checkpoint...")
                try:
                    asr = load_custom_asr_encoder_only(model.cfg.pretrained_asr)
                    logger.info("Successfully loaded encoder from custom ASR model")
                except Exception as custom_e:
                    logger.error(f"Failed to load custom ASR encoder: {custom_e}")
                    logger.info(
                        "Consider using an official pretrained model:\n"
                        "  pretrained_asr: nvidia/canary-1b-v2\n"
                        "  pretrained_asr: nvidia/canary-1b-flash"
                    )
                    raise
            else:
                logger.error(f"Failed to load ASR model: {e}")
                logger.info(
                    "Please consider using an official pretrained model by setting:\n"
                    "  pretrained_asr: nvidia/canary-1b-v2\n"
                    "or\n"
                    "  pretrained_asr: nvidia/canary-1b-flash\n"
                    "in your configuration file."
                )
                raise
        finally:
            # Restore original logging level
            asr_logger.setLevel(original_level)

        with open_dict(model.cfg):
            model.cfg.perception.preprocessor = asr.cfg.preprocessor
            model.cfg.perception.encoder = asr.cfg.encoder
            model.cfg.perception.output_dim = model.llm.config.hidden_size
        model.perception = AudioPerceptionModule(model.cfg.perception).train()
        model.perception.load_state_dict(asr.state_dict(), strict=False)
    else:
        raise NotImplementedError("Random initialization of speech encoder is not yet supported")