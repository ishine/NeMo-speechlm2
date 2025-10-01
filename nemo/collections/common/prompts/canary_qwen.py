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
# pylint: disable=C0115

import torch
from lhotse.cut import Cut, MixedCut

from nemo.collections.common.data.prompt_fn import registered_prompt_format_fn
from nemo.collections.common.prompts.formatter import Modality, PromptFormatter

CANARY_QWEN_BOT = "<|im_start|>"
CANARY_QWEN_EOT = "<|im_end|>"


class CanaryQwenPromptFormatter(PromptFormatter):
    """
    Canary-Qwen prompt formatter that supports system, user, and assistant roles.
    This is an extended version of QwenPromptFormatter specifically designed for Canary-Qwen models.
    """
    NAME = "canary_qwen"
    OUTPUT_ROLE = "assistant"
    INFERENCE_PREFIX = f"{CANARY_QWEN_BOT}assistant\n"
    TEMPLATE = {
        "system": {
            "template": f"{CANARY_QWEN_BOT}system\n|message|{CANARY_QWEN_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        "system_and_user": {
            "template": f"{CANARY_QWEN_BOT}system\n|system|{CANARY_QWEN_EOT}\n{CANARY_QWEN_BOT}user\n|message|{CANARY_QWEN_EOT}\n",
            "slots": {
                "system": Modality.Text,
                "message": Modality.Text,
            },
        },
        "user": {
            "template": f"{CANARY_QWEN_BOT}user\n|message|{CANARY_QWEN_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": f"{INFERENCE_PREFIX}|message|{CANARY_QWEN_EOT}\n",
            "slots": {
                "message": Modality.Text,
            },
        },
    }


@registered_prompt_format_fn(Cut, CanaryQwenPromptFormatter)
def canary_qwen_cut_prompt_format_fn(cut: Cut, prompt: CanaryQwenPromptFormatter) -> dict[str, torch.Tensor]:
    """
    Prompt format function for Canary-Qwen with Cut objects.
    """
    if isinstance(cut, MixedCut):
        cut = cut.first_non_padding_cut

    if cut.has_custom("context"):
        context = cut.context
    elif cut.has_custom("question"):
        context = cut.question
    elif hasattr(cut, "default_context"):
        context = cut.default_context
    else:
        # Fallback to a default ASR prompt if no context is provided
        context = "Transcribe the following: "

    # CRITICAL FIX: Add audio placeholder tag to the user message
    # This ensures the audio placeholder token is inserted into input_ids
    # The tag should match what's configured in the model (usually "<|audio|>")
    audio_locator_tag = cut.custom.get("audio_locator_tag", "<|audio|>") if cut.has_custom("audio_locator_tag") else "<|audio|>"

    # Append audio tag to the context
    user_message = f"{context}{audio_locator_tag}"

    turns = []

    # Add system prompt if available
    if cut.has_custom("system_prompt"):
        turns.append({"role": "system_and_user", "slots": {"system": cut.system_prompt, "message": user_message}})
    else:
        turns.append({"role": "user", "slots": {"message": user_message}})

    # Add assistant response if available
    if (answer := cut.supervisions[0].text) is not None:
        turns.append({"role": "assistant", "slots": {"message": answer}})

    return prompt.encode_dialog(turns)