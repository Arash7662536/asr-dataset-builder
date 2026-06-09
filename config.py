#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central configuration. Everything you are likely to tune lives here so you don't
have to hunt through modules. CLI flags in run.py override these at runtime.

The two knobs you specifically asked to be able to adjust:
  * ROUNDTRIP_BEAM_SIZE  -> beam search width for the round-trip verification pass
  * the purity thresholds (MAX_CER / MIN_LCS / LEN_TOL / MIN_MATCH / DROP_LCS_FLOOR)
"""

from dataclasses import dataclass, field
from typing import Tuple


# --------------------------------------------------------------------------- #
# Audio / models
# --------------------------------------------------------------------------- #
SR = 16000                       # everything works at 16 kHz mono

# Your own fine-tuned Whisper, already converted to CTranslate2 (ct2).
# Point --whisper-model at the directory that contains model.bin + config.json.
WHISPER_MODEL = "/path/to/your/ct2-whisper"     # MUST override via --whisper-model
COMPUTE_TYPE = "float16"                          # float16 | int8_float16 | int8 | float32

# Optional CTC acoustic model (only loaded when --aligner ctc).
CTC_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-persian"
INFER_WINDOW_S = 30              # wav2vec2 inference window (memory bound)


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
# "whisper" = Whisper word-timestamp + word-level text match (robust to
#             paraphrased / twisted narration; default).
# "ctc"     = CTC-segmentation forced alignment (best when audio is verbatim).
ALIGNER = "whisper"
CTC_FALLBACK_TO_WHISPER = True   # if CTC yields nothing (or "Audio shorter than
                                 # text"), retry that file with the whisper aligner.
MIN_MATCH = 0.30                 # [whisper aligner] drop a sentence if fewer than
                                 # this fraction of its words match the transcript.


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
MIN_CHUNK_S = 1.5
MAX_CHUNK_S = 28.0               # keep under Whisper's 30 s window
MAX_GAP_S = 1.2                  # flush a chunk if pause between sentences > this
PAD_S = 0.15                     # padding added to each chunk edge
USE_VAD_SNAP = False             # snap chunk edges to silero-vad speech boundaries


# --------------------------------------------------------------------------- #
# Round-trip verification (HARDENED — applies fix (4) + VAD)
# --------------------------------------------------------------------------- #
RUN_ROUNDTRIP = True
ROUNDTRIP_BEAM_SIZE = 3          # (4) beam search dramatically reduces the
                                 #     "سایم گفت" premature-stop truncations.
LOCATE_BEAM_SIZE = 1             # rough/locating pass only needs to be approximate.
CONDITION_ON_PREVIOUS_TEXT = False   # (4) stop one bad segment cascading.

# VAD on the round-trip pass: strips silence/music that triggers early stopping.
# NOTE: VAD is intentionally OFF in the word-timestamp ALIGNMENT pass, because
# VAD-shifted word timestamps would corrupt chunk boundaries.
ROUNDTRIP_VAD = True
ROUNDTRIP_VAD_MIN_SILENCE_MS = 500   # lower than the ~2000 default -> cuts the
                                     # medium inter-sentence pauses too.


# --------------------------------------------------------------------------- #
# Purity tiers (keep / review / drop)
# --------------------------------------------------------------------------- #
MAX_CER = 0.30                   # keep-tier needs round-trip CER <= this
MIN_LCS = 0.85                   # keep-tier needs this word-overlap (LCS) ratio
LEN_TOL = 0.25                   # keep-tier allows |len(hyp)/len(ref)-1| up to this
DROP_LCS_FLOOR = 0.35            # below this overlap -> hard drop (different words)

# Which tiers land in the training dataset. "review" is always saved to a
# SEPARATE dataset for human triage regardless of this setting.
TIERS_TO_EXPORT: Tuple[str, ...] = ("keep",)

# Precision = lcs / len(hyp). A truncation ("سایم گفت") has HIGH precision but
# LOW recall. This is reported for inspection only and does NOT gate by default,
# per your call to go with fix (4) over fix (3). Flip to True to route
# high-precision/low-recall (likely truncations) to "review" instead of "drop".
ENABLE_PRECISION_REVIEW = False
PRECISION_REVIEW_MIN = 0.90      # used only when ENABLE_PRECISION_REVIEW is True


# --------------------------------------------------------------------------- #
# Front/back-matter coarse drop (Persian keywords)
# --------------------------------------------------------------------------- #
FRONTBACK_KEYWORDS = [
    "فهرست", "فهرست مطالب", "شناسنامه", "حق چاپ", "کلیه حقوق",
    "تقدیم", "سپاسگزاری", "قدردانی", "پیشگفتار ناشر", "درباره نویسنده",
    "درباره مترجم", "کتابنامه", "منابع", "واژه‌نامه", "نمایه",
]
DROP_FRONTBACK = True


@dataclass
class Settings:
    """Runtime settings bundle (populated from config + CLI in run.py)."""
    whisper_model: str = WHISPER_MODEL
    compute_type: str = COMPUTE_TYPE
    ctc_model: str = CTC_MODEL
    aligner: str = ALIGNER
    ctc_fallback_to_whisper: bool = CTC_FALLBACK_TO_WHISPER
    min_match: float = MIN_MATCH

    use_vad_snap: bool = USE_VAD_SNAP

    run_roundtrip: bool = RUN_ROUNDTRIP
    roundtrip_beam_size: int = ROUNDTRIP_BEAM_SIZE
    locate_beam_size: int = LOCATE_BEAM_SIZE
    condition_on_previous_text: bool = CONDITION_ON_PREVIOUS_TEXT
    roundtrip_vad: bool = ROUNDTRIP_VAD
    roundtrip_vad_min_silence_ms: int = ROUNDTRIP_VAD_MIN_SILENCE_MS

    max_cer: float = MAX_CER
    min_lcs: float = MIN_LCS
    len_tol: float = LEN_TOL
    drop_lcs_floor: float = DROP_LCS_FLOOR
    tiers_to_export: Tuple[str, ...] = TIERS_TO_EXPORT
    enable_precision_review: bool = ENABLE_PRECISION_REVIEW
    precision_review_min: float = PRECISION_REVIEW_MIN

    drop_frontback: bool = DROP_FRONTBACK
    keep_rejected: bool = False
