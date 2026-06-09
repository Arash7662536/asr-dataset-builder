#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E. Merge aligned sentences into <=30 s chunks (+ optional VAD snapping)."""

import numpy as np

from config import SR, MIN_CHUNK_S, MAX_CHUNK_S, MAX_GAP_S
from .dtypes import Chunk


def merge_segments(segs):
    chunks, buf = [], []

    def flush():
        if not buf:
            return
        start, end = buf[0].start, buf[-1].end
        if end - start < MIN_CHUNK_S:
            buf.clear()
            return
        chunks.append(Chunk(
            text=" ".join(s.text for s in buf),
            start=start, end=end,
            score=min(s.score for s in buf),
        ))
        buf.clear()

    for seg in segs:
        if seg.end <= seg.start:
            continue
        if buf:
            gap = seg.start - buf[-1].end
            span = seg.end - buf[0].start
            if gap > MAX_GAP_S or span > MAX_CHUNK_S:
                flush()
        buf.append(seg)
        if buf[-1].end - buf[0].start >= MAX_CHUNK_S:
            flush()
    flush()
    return chunks


def load_vad():
    import torch
    model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad",
                                  trust_repo=True)
    return model, utils[0]


def snap_to_silence(chunks, audio, vad):
    """Nudge chunk edges to the nearest silero-vad speech boundary."""
    import torch
    model, get_speech_timestamps = vad
    speech = get_speech_timestamps(torch.from_numpy(audio), model, sampling_rate=SR)
    starts = np.array([s["start"] / SR for s in speech])
    ends = np.array([s["end"] / SR for s in speech])
    if len(starts) == 0:
        return chunks
    for c in chunks:
        si = int(np.argmin(np.abs(starts - c.start)))
        ei = int(np.argmin(np.abs(ends - c.end)))
        if abs(starts[si] - c.start) < 0.4:
            c.start = float(starts[si])
        if abs(ends[ei] - c.end) < 0.4:
            c.end = float(ends[ei])
    return chunks
