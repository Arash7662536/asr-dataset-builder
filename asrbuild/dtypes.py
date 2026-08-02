#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small shared dataclasses, kept here to avoid circular imports.

Each text-carrying type holds two forms of the same string:
  text  — the ground truth WITH punctuation; this is what gets exported.
  norm  — the same text punctuation-free; this is what all matching uses.
When --no-punctuation is set the two are identical, so the whole pipeline
degrades to the original punctuation-free behaviour.
"""

from dataclasses import dataclass


@dataclass
class Sentence:
    text: str            # normalized, faithful ground-truth (punctuated)
    offset: int          # char offset into the joined book_text (norm form)
    norm: str = ""       # matching form; defaults to text

    def __post_init__(self):
        if not self.norm:
            self.norm = self.text


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
    norm: str = ""               # matching form; defaults to text

    def __post_init__(self):
        if not self.norm:
            self.norm = self.text


@dataclass
class Chunk:
    text: str
    start: float
    end: float
    score: float
    norm: str = ""               # matching form; defaults to text

    def __post_init__(self):
        if not self.norm:
            self.norm = self.text
