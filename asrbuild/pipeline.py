#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Per-book orchestration: runs stages A..F for one book directory, with a log
line + progress bar per stage. Designed to be called by a worker that has
already loaded the models once.
"""

import glob
import os

import numpy as np

from config import SR
from .dtypes import Mp3Map
from .textextract import extract_book, MissingTextError
from .whisper_io import rough_transcribe
from .mapping import dedup_by_md5, map_mp3s, sentences_in_range
from .align import align_via_whisper, align_file
from .chunk import merge_segments, snap_to_silence
from .export import export_and_filter, save_shard
from .logging_utils import stage


class Models:
    """Container for the per-worker loaded models."""
    def __init__(self, whisper, ctc=None, vad=None):
        self.whisper = whisper
        self.ctc = ctc
        self.vad = vad


def _tqdm(iterable, **kw):
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kw)
    except Exception:
        return iterable


def process_book(book_dir, shard_dir, settings, models, log, position=0):
    """Process a single book. Returns a summary dict. Idempotent: a book whose
    _DONE marker exists is skipped by the caller before reaching here."""
    import librosa

    book_id = os.path.basename(book_dir.rstrip("/"))
    text_dir = os.path.join(book_dir, "text")
    audio_dir_in = os.path.join(book_dir, "audio")
    os.makedirs(shard_dir, exist_ok=True)
    audio_out = os.path.join(shard_dir, "audio")
    os.makedirs(audio_out, exist_ok=True)

    # ---- A. text ----
    try:
        with stage(log, book_id, "A"):
            sentences, book_text, (src, kind) = extract_book(
                text_dir, drop_frontback=settings.drop_frontback,
                keep_punct=settings.keep_punctuation)
        log.info("[%s][A] %s: %d sentences, %d chars",
                 book_id, kind, len(sentences), len(book_text))
    except MissingTextError as e:
        log.warning("[%s] SKIP — %s", book_id, e)
        open(os.path.join(shard_dir, "_DONE"), "w").close()      # don't retry
        return {"book": book_id, "status": "no_text", "keep": 0, "review": 0}

    # ---- audio discovery + dedup (note 2) ----
    mp3_paths = sorted(glob.glob(os.path.join(audio_dir_in, "*.mp3")))
    if not mp3_paths:
        log.warning("[%s] SKIP — no mp3s in %s", book_id, audio_dir_in)
        open(os.path.join(shard_dir, "_DONE"), "w").close()
        return {"book": book_id, "status": "no_audio", "keep": 0, "review": 0}
    mp3_paths = dedup_by_md5(mp3_paths, log=log)

    # ---- B. rough transcribe (note 3: order unknown) ----
    with stage(log, book_id, "B", f"({len(mp3_paths)} mp3s)"):
        mp3_texts = [
            rough_transcribe(models.whisper, p, beam_size=settings.locate_beam_size)
            for p in _tqdm(mp3_paths, desc=f"[{book_id}] B transcribe",
                           position=position, leave=False)
        ]

    # ---- C. map -> ranges ----
    with stage(log, book_id, "C"):
        maps = map_mp3s(mp3_paths, mp3_texts, book_text)
        for m in maps:
            log.info("[%s][C]   %-40s book[%d:%d] anchor %.0f",
                     book_id, os.path.basename(m.path),
                     m.start_off, m.end_off, m.anchor_score)

    keep_all, review_all, audit_all = [], [], []

    for m in _tqdm(maps, desc=f"[{book_id}] books", position=position, leave=False):
        name = os.path.basename(m.path)
        segs_text = sentences_in_range(sentences, m.start_off, m.end_off)
        if not segs_text:
            log.info("[%s] %s: no sentences in range, skip", book_id, name)
            continue

        audio, _ = librosa.load(m.path, sr=SR, mono=True)
        audio = audio.astype(np.float32)

        # ---- D. align (+ ctc fallback to whisper on the guard) ----
        with stage(log, book_id, "D", f"{name} ({len(segs_text)} candidates)"):
            if settings.aligner == "whisper":
                aligned = align_via_whisper(models.whisper, audio, segs_text,
                                            min_match=settings.min_match)
            else:
                aligned = align_file(models.ctc, audio, segs_text, log=log)
                if not aligned and settings.ctc_fallback_to_whisper:
                    log.info("[%s] %s: CTC empty -> whisper aligner fallback",
                             book_id, name)
                    aligned = align_via_whisper(models.whisper, audio, segs_text,
                                                min_match=settings.min_match)

        # ---- E. merge (+ optional vad snap) ----
        with stage(log, book_id, "E"):
            chunks = merge_segments(aligned)
            if settings.use_vad_snap and models.vad is not None:
                chunks = snap_to_silence(chunks, audio, models.vad)
        log.info("[%s] %s: %d aligned -> %d chunks",
                 book_id, name, len(aligned), len(chunks))

        # ---- F. export + filter ----
        with stage(log, book_id, "F"):
            keep, review, audit = export_and_filter(
                chunks, audio, m.path, audio_out, settings,
                models.whisper, log=log)
        n_drop = len(audit) - len(keep) - len(review)
        log.info("[%s] %s: keep %d · review %d · drop %d",
                 book_id, name, len(keep), len(review), n_drop)
        keep_all.extend(keep)
        review_all.extend(review)
        audit_all.extend(audit)

    save_shard(keep_all, review_all, audit_all, shard_dir)
    dur_h = sum(r["metrics"]["duration"] for r in keep_all) / 3600
    log.info("[%s] DONE — keep %d (~%.2fh) · review %d",
             book_id, len(keep_all), dur_h, len(review_all))
    return {"book": book_id, "status": "ok",
            "keep": len(keep_all), "review": len(review_all),
            "keep_hours": round(dur_h, 3)}
