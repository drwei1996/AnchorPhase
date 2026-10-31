"""AnchorPhase: anchored negative-binomial inference of transcriptional phase.

Frozen manuscript version (B0 corrected-offset, task 10A, 2026-10-07/08).
Modules:
    anchorphase.model.katv2_engine       canonical B0..B7 engine (48-grid exact posterior)
    anchorphase.training                 frozen B0 trainers (original + corrected offset)
    anchorphase.evaluation               evaluation, CPI comparators, external-validation pipeline
    anchorphase.utils                    shared figure style helpers
"""

__version__ = "1.0.0"
FROZEN_MODEL_TAG = "KATv2-B0-corrected-offset-20261007"
ENGINE_SEED = 20261007
