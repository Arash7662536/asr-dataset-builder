#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
F. Export chunks to 16 kHz mono WAV, run the round-trip filter, tier them, and
build a HuggingFace Arrow dataset shard with columns:

    audio   : Audio feature  -> decodes to {path, array, sampling_rate}
    text    : string         -> the BOOK ground truth
    metrics : struct         -> the signals used to keep/review/drop the chunk

Per book we produce:
    <shard>/audio/*.wav        (kept + review wavs; rejected too if --keep-rejected)
    <shard>/keep_ds/           (save_to_disk; tiers in TIERS_TO_EXPORT)
    <shard>/review_ds/         (save_to_disk; review tier, for human triage)
    <shard>/audit.jsonl        (every chunk, all tiers, all metrics, hyp)
    <shard>/_DONE              (idempotency marker)
"""

import json
import os

import numpy as np

from config import SR, PAD_S, MIN_CHUNK_S
from .metrics import text_metrics, purity_tier
from .whisper_io import transcribe_chunk


# Feature schema is built lazily so importing this module doesn't require the
# `datasets` package (keeps py_compile / partial environments happy).
def _features():
    from datasets import Features, Audio, Value
    metric_struct = {
        "tier": Value("string"),
        "match_ratio": Value("float32"),    # aligner score (ctc conf | whisper match)
        "cer": Value("float32"),
        "wer": Value("float32"),
        "lcs_ratio": Value("float32"),
        "len_ratio": Value("float32"),
        "precision": Value("float32"),
        "duration": Value("float32"),
        "source_mp3": Value("string"),
        "hyp": Value("string"),
    }
    return Features({
        "audio": Audio(sampling_rate=SR),
        "text": Value("string"),          # book ground truth, WITH punctuation
        "text_plain": Value("string"),    # same text, punctuation stripped
        "metrics": metric_struct,
    })


def export_and_filter(chunks, audio, mp3_name, audio_dir, settings,
                      whisper, log=None):
    """Write WAVs + classify. Returns (keep_rows, review_rows, audit_rows).

    *_rows are dicts shaped for the Arrow schema: {audio(path), text, metrics}.
    audit_rows are plain dicts (jsonl) for every chunk including drops.
    """
    import soundfile as sf

    base = os.path.splitext(os.path.basename(mp3_name))[0]
    keep_rows, review_rows, audit_rows = [], [], []

    for i, c in enumerate(chunks):
        a = max(0.0, c.start - PAD_S)
        b = min(len(audio) / SR, c.end + PAD_S)
        seg = audio[int(a * SR):int(b * SR)]
        dur = round(b - a, 3)
        fname = f"{base}_{i:04d}.wav"

        audit = {
            "file_name": None, "text": c.text, "text_plain": c.norm,
            "duration": dur,
            "source_mp3": os.path.basename(mp3_name),
            "match_ratio": round(float(c.score), 4),
            "cer": None, "wer": None, "lcs_ratio": None,
            "len_ratio": None, "precision": None, "hyp": None,
            "tier": None, "drop_reason": None,
        }

        if len(seg) < int(MIN_CHUNK_S * SR):
            audit["tier"] = "drop"
            audit["drop_reason"] = "too_short"
            audit_rows.append(audit)
            continue

        fpath = os.path.join(audio_dir, fname)
        sf.write(fpath, seg, SR, subtype="PCM_16")

        if settings.run_roundtrip:
            hyp = transcribe_chunk(
                whisper, fpath,
                beam_size=settings.roundtrip_beam_size,
                condition_on_previous_text=settings.condition_on_previous_text,
                vad_filter=settings.roundtrip_vad,
                vad_min_silence_ms=settings.roundtrip_vad_min_silence_ms,
            )
            # hyp is normalized punctuation-free, so compare against c.norm —
            # punctuation must never move a metric or a tier.
            m = text_metrics(c.norm, hyp)
            audit.update(hyp=hyp, **m)
            tier = purity_tier(
                m, settings.max_cer, settings.min_lcs, settings.len_tol,
                drop_lcs_floor=settings.drop_lcs_floor,
                enable_precision_review=settings.enable_precision_review,
                precision_review_min=settings.precision_review_min,
            )
        else:
            tier = "keep"                                    # trust the aligner
            m = dict(cer=None, wer=None, lcs_ratio=None,
                     len_ratio=None, precision=None)

        audit["tier"] = tier
        audit["drop_reason"] = None if tier == "keep" else tier

        metrics_col = {
            "tier": tier,
            "match_ratio": round(float(c.score), 4),
            "cer": m.get("cer"), "wer": m.get("wer"),
            "lcs_ratio": m.get("lcs_ratio"), "len_ratio": m.get("len_ratio"),
            "precision": m.get("precision"), "duration": dur,
            "source_mp3": os.path.basename(mp3_name), "hyp": audit["hyp"],
        }
        row = {"audio": fpath, "text": c.text, "text_plain": c.norm,
               "metrics": metrics_col}

        if tier in settings.tiers_to_export:
            audit["file_name"] = f"audio/{fname}"
            keep_rows.append(row)
        elif tier == "review":
            audit["file_name"] = f"audio/{fname}"
            review_rows.append(row)
        else:                                                # drop
            if settings.keep_rejected:
                audit["file_name"] = f"audio/{fname}"        # leave wav for review
            else:
                os.remove(fpath)
        audit_rows.append(audit)

    return keep_rows, review_rows, audit_rows


def save_shard(keep_rows, review_rows, audit_rows, shard_dir):
    """Persist per-book outputs (Arrow datasets + audit jsonl + DONE marker)."""
    from datasets import Dataset
    feats = _features()

    def _to_ds(rows):
        cols = {"audio": [r["audio"] for r in rows],
                "text": [r["text"] for r in rows],
                "text_plain": [r["text_plain"] for r in rows],
                "metrics": [r["metrics"] for r in rows]}
        return Dataset.from_dict(cols, features=feats)

    if keep_rows:
        _to_ds(keep_rows).save_to_disk(os.path.join(shard_dir, "keep_ds"))
    if review_rows:
        _to_ds(review_rows).save_to_disk(os.path.join(shard_dir, "review_ds"))

    with open(os.path.join(shard_dir, "audit.jsonl"), "w", encoding="utf-8") as f:
        for r in audit_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    open(os.path.join(shard_dir, "_DONE"), "w").close()
