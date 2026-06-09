#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small shared dataclasses, kept here to avoid circular imports."""

from dataclasses import dataclass


@dataclass
class Sentence:
    text: str            # normalized, faithful ground-truth
    offset: int          # char offset into the joined book_text


@dataclass
class Mp3Map:
    path: str
    start_off: int
    end_off: int = -1            # filled after sorting
    anchor_score: float = 0.0


@dataclass
class AlignedSeg:
    text: str
    start: float
    end: float
    score: float                 # aligner score (CTC conf, or whisper match ratio)


@dataclass
class Chunk:
    text: str
    start: float
    end: float
    score: float
