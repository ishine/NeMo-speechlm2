"""
Compatibility shim for older NeMo ASR model checkpoints.

This module provides backward compatibility for checkpoints that reference
'hybrid_transformer_transcribe_ctc_bpe_models' which has been renamed/reorganized
in newer NeMo versions.
"""

# Import the actual model class from its current location
# Based on the model name "FastConformer-Hybrid-Transformer-CTC-BPE",
# this is likely a Conformer/Transformer hybrid CTC model with BPE tokenization
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models import EncDecHybridRNNTCTCBPEModel

# Create aliases for backward compatibility
# The exact class name might be one of these:
EncDecHybridTransfTranscribeCTCModelBPE = EncDecCTCModelBPE
EncDecHybridTransformerCTCModelBPE = EncDecCTCModelBPE
EncDecHybridTransfCTCBPEModel = EncDecCTCModelBPE

# Add specific compatibility for the custom model
HybridTransformerTranscribeCTCBPEModel = EncDecHybridRNNTCTCBPEModel

# Export all possible class names that might be referenced
__all__ = [
    'EncDecHybridTransfTranscribeCTCModelBPE',
    'EncDecHybridTransformerCTCModelBPE',
    'EncDecHybridTransfCTCBPEModel',
    'EncDecCTCModelBPE',
    'HybridTransformerTranscribeCTCBPEModel',
]