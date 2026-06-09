#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F-metrics. Compare book text (ref) to Whisper hypothesis (hyp)."""

import difflib


def text_metrics(ref, hyp):
    """Return complementary signals so verbatim clips can be told apart from
    paraphrases AND from truncations.

      cer        - char error rate (jiwer)
      wer        - word error rate
      lcs_ratio  - longest common word subsequence / REF words   (recall-like)
      len_ratio  - hyp_words / ref_words
      precision  - longest common word subsequence / HYP words    (informational)
                   high precision + low lcs_ratio == a truncation, not a paraphrase
    """
    import jiwer
    r = ref.split()
    h = hyp.split()
    if not r:
        return dict(cer=1.0, wer=1.0, lcs_ratio=0.0, len_ratio=0.0, precision=0.0)
    cer = jiwer.cer(ref, hyp) if hyp else 1.0
    wer = jiwer.wer(ref, hyp) if hyp else 1.0
    sm = difflib.SequenceMatcher(None, r, h, autojunk=False)
    lcs = sum(b.size for b in sm.get_matching_blocks())
    return dict(
        cer=round(float(cer), 4),
        wer=round(float(min(wer, 1.0)), 4),
        lcs_ratio=round(lcs / len(r), 4),
        len_ratio=round(len(h) / len(r), 4) if r else 0.0,
        precision=round(lcs / len(h), 4) if h else 0.0,
    )


def purity_tier(m, max_cer, min_lcs, len_tol, drop_lcs_floor=0.35,
                enable_precision_review=False, precision_review_min=0.90):
    """keep / review / drop from metrics.

    keep   : near-verbatim (cer ok AND high word overlap AND length ~1)
    drop   : clearly different audio (very low overlap)
    review : everything in between -> flagged for the human

    When enable_precision_review is set, a low-overlap clip that nonetheless has
    very high precision (a likely Whisper truncation, NOT a paraphrase) is routed
    to review instead of being hard-dropped. Off by default."""
    if m["lcs_ratio"] < drop_lcs_floor:
        if (enable_precision_review
                and m.get("precision", 0.0) >= precision_review_min):
            return "review"                                  # likely truncation
        return "drop"                                        # different words
    verbatim = (m["cer"] <= max_cer
                and m["lcs_ratio"] >= min_lcs
                and abs(m["len_ratio"] - 1.0) <= len_tol)
    return "keep" if verbatim else "review"
