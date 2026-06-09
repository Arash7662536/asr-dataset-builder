#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persian text normalization (two-stage)."""

import re
import unicodedata

_ARABIC_TO_PERSIAN = str.maketrans({
    "ي": "ی", "ك": "ک", "ﻲ": "ی", "ﻚ": "ک",
    "ۀ": "ه", "ة": "ه", "أ": "ا", "إ": "ا", "آ": "آ",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")   # harakat + tatweel
_NON_FA = re.compile(r"[^\u0621-\u064A\u067E\u0686\u0698\u06A9\u06AF\u06CC"
                     r"\u0621-\u06FF0-9\s\u200c]")          # keep fa letters/digits/ZWNJ
_WS = re.compile(r"\s+")

try:
    from hazm import Normalizer, sent_tokenize
    _HAZM = Normalizer()
    def _hazm_norm(t): return _HAZM.normalize(t)
    def sent_split(t): return sent_tokenize(t)
except Exception:                                            # graceful fallback
    _hazm_norm = lambda t: t
    def sent_split(t):
        parts = re.split(r"(?<=[\.\!\?\؟\:\؛])\s+|\n+", t)
        return [p for p in parts if p.strip()]

try:
    from num2fawords import words as _num2fa
    def _digits_to_words(t):
        return re.sub(r"\d+", lambda m: _num2fa(int(m.group())), t)
except Exception:
    _digits_to_words = lambda t: t


def pre_normalize(text):
    """Stage 1: unify characters but KEEP sentence-ending punctuation so the
    splitter has something to split on. Used only before splitting."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_ARABIC_TO_PERSIAN)
    text = _DIACRITICS.sub("", text)
    text = _hazm_norm(text)
    text = _WS.sub(" ", text).strip()
    return text


def normalize_fa(text, spoken_numbers=True):
    """Stage 2: final ground-truth form. Strips punctuation, maps digits to
    spoken words, collapses whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_ARABIC_TO_PERSIAN)
    text = _DIACRITICS.sub("", text)
    text = _hazm_norm(text)
    if spoken_numbers:
        text = _digits_to_words(text)
    text = _NON_FA.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text
