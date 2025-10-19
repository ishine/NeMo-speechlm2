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

from nemo.collections.speechlm2.metrics.wer_calculator import (
    WERCalculatorBase,
    LegacyWERCalculator,
    OpenASRLeaderboardWERCalculator,
    create_wer_calculator,
)

# Text normalization utilities
from nemo.collections.speechlm2.metrics.normalization import (
    BasicMultilingualTextNormalizer,
    EnglishNumberNormalizer,
    EnglishSpellingNormalizer,
    EnglishTextNormalizer,
    english_spelling_normalizer,
)

# WER calculation utilities
from nemo.collections.speechlm2.metrics.wer_utils import calculate_wer_with_empty_handling

__all__ = [
    # WER Calculator classes
    "WERCalculatorBase",
    "LegacyWERCalculator",
    "OpenASRLeaderboardWERCalculator",
    "create_wer_calculator",
    # Text normalization classes
    "BasicMultilingualTextNormalizer",
    "EnglishNumberNormalizer",
    "EnglishSpellingNormalizer",
    "EnglishTextNormalizer",
    "english_spelling_normalizer",
    # WER calculation utilities
    "calculate_wer_with_empty_handling",
]
