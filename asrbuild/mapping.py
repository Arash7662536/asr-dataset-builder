#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C. Map each MP3 to a contiguous book char-range and recover order.

Handles your note 2 (duplicate audio -> md5 dedup) and note 3 (unknown MP3
order -> rapidfuzz anchoring then sort by matched offset).
"""

import hashlib
import os

from config import SR
from .dtypes import Mp3Map


def dedup_by_md5(paths, log=None):
    """Drop byte-identical duplicate MP3s (keeps first by sorted name)."""
    seen, kept = {}, []
    for p in paths:
        h = hashlib.md5()
        with open(p, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        digest = h.hexdigest()
        if digest in seen:
            if log:
                log.info("duplicate audio: %s == %s (skipped)",
                         os.path.basename(p), os.path.basename(seen[digest]))
            continue
        seen[digest] = p
        kept.append(p)
    return kept


def map_mp3s(mp3_paths, mp3_texts, book_text,
             probe_skip_words=30, probe_len_words=45):
    """Anchor each MP3 to a book offset (rapidfuzz partial alignment), then
    derive contiguous ranges by sorting on the matched offset."""
    from rapidfuzz import fuzz

    maps = []
    for path, txt in zip(mp3_paths, mp3_texts):
        words = txt.split()
        probe = " ".join(words[probe_skip_words:probe_skip_words + probe_len_words])
        if len(probe) < 20:                                  # very short file
            probe = " ".join(words[:probe_len_words])
        al = fuzz.partial_ratio_alignment(probe, book_text) if probe else None
        off = al.dest_start if al else 0
        maps.append(Mp3Map(path=path, start_off=off,
                           anchor_score=al.score if al else 0.0))

    maps.sort(key=lambda m: m.start_off)                     # recover order
    for i, m in enumerate(maps):
        m.end_off = maps[i + 1].start_off if i + 1 < len(maps) else len(book_text)
    return maps


def sentences_in_range(sentences, start_off, end_off):
    return [s for s in sentences if start_off <= s.offset < end_off]
