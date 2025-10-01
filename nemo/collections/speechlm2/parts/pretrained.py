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


def load_pretrained_hf(model_path_or_name: str, pretrained_weights: bool = True, dtype=torch.float32):
    """
    Load pretrained HuggingFace AutoModelForCausalLM.

    Setting ``pretrained_weights=False`` returns a model that has identical architecture with the checkpoint,
    but is randomly initialized.
    """
    if pretrained_weights:
        return AutoModelForCausalLM.from_pretrained(model_path_or_name, torch_dtype=dtype)
    else:
        config = AutoConfig.from_pretrained(model_path_or_name)
        return AutoModelForCausalLM.from_config(config, torch_dtype=dtype)


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