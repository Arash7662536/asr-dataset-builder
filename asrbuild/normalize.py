#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persian text normalization (two-stage), punctuation-aware.

Every book sentence is carried in TWO forms:

    display  = normalize_fa(raw, keep_punct=True)   -> the dataset `text` column
    matching = strip_punct(display)                 -> every comparison uses this

Whisper hypotheses are always normalized punctuation-free, so mp3->book mapping,
word-level alignment and the round-trip CER/WER all run on the matching form,
and the punctuation is re-attached for free by exporting the display form.

The invariant that makes this safe is

    strip_punct(normalize_fa(t, keep_punct=True)) == normalize_fa(t)

i.e. punctuation cannot change a single matching decision. It is checked by
tests/test_normalize_punct.py.
"""

import re
import unicodedata

from config import KEEP_PUNCT_CHARS, PUNCT_STYLE, ELLIPSIS_AS

_ARABIC_TO_PERSIAN = str.maketrans({
    "ي": "ی", "ك": "ک", "ﻲ": "ی", "ﻚ": "ک",
    "ۀ": "ه", "ة": "ه", "أ": "ا", "إ": "ا", "آ": "آ",
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
})
_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u0640]")   # harakat + tatweel
_FA_BODY = r"\u0621-\u06FF0-9\s\u200c"       # fa letters/digits/ZWNJ
_NON_FA = re.compile(r"[^" + _FA_BODY + r"]")
_WS = re.compile(r"\s+")


# --------------------------------------------------------------------------- #
# Punctuation handling
# --------------------------------------------------------------------------- #
# Variant spellings are folded onto the configured style BEFORE filtering, so
# KEEP_PUNCT_CHARS may be written in either style and still work.
_TO_PERSIAN_PUNCT = {",": "،", "?": "؟", ";": "؛",
                     "‚": "،", "，": "،", "﹐": "،",
                     "？": "؟", "！": "!", "：": ":",
                     "．": ".", "。": "."}
_TO_ASCII_PUNCT = {"،": ",", "؟": "?", "؛": ";",
                   "‚": ",", "，": ",", "﹐": ",",
                   "？": "?", "！": "!", "：": ":",
                   "．": ".", "。": "."}
_PUNCT_CANON = str.maketrans(_TO_ASCII_PUNCT if PUNCT_STYLE == "ascii"
                             else _TO_PERSIAN_PUNCT)

_ELL = "\u0001"                      # sentinel: an ellipsis, kept as one token
_ELL_RE = re.compile(r"…+|\.{3,}")
_ELL_OUT = ELLIPSIS_AS or ""

# dedup while preserving order, after folding onto the configured style
_PUNCT_KEEP = "".join(dict.fromkeys(KEEP_PUNCT_CHARS.translate(_PUNCT_CANON)))
_CLASS = re.escape(_PUNCT_KEEP + _ELL)         # punctuation char-class body

_NON_FA_KEEP = re.compile(r"[^" + _FA_BODY + _CLASS + r"]")
# a run of punctuation (possibly space-separated) collapses to its first mark
_PUNCT_RUN = re.compile(r"([" + _CLASS + r"])(?:[\s" + _CLASS + r"]*["
                        + _CLASS + r"])+")
_SPACE_BEFORE = re.compile(r"[ \t\u200c]+([" + _CLASS + r"])")
_SPACE_AFTER = re.compile(r"([" + _CLASS + r"])(?=[^\s" + _CLASS + r"])")
_LEADING_PUNCT = re.compile(r"^[\s" + _CLASS + r"]+")
# the strip class covers the sentinel AND whatever the sentinel expands to
_STRIP_CLASS = re.escape("".join(dict.fromkeys(_PUNCT_KEEP + _ELL + _ELL_OUT)))
_PUNCT_STRIP = re.compile(r"[" + _STRIP_CLASS + r"]") if _STRIP_CLASS else None

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


def _base_normalize(text):
    """Character-level unification shared by both stages."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_ARABIC_TO_PERSIAN)
    text = _DIACRITICS.sub("", text)
    return _hazm_norm(text)


def pre_normalize(text):
    """Stage 1: unify characters but KEEP all punctuation so the sentence
    splitter has something to split on. Used only before splitting."""
    return _WS.sub(" ", _base_normalize(text)).strip()


def _apply_punct(text):
    """Keep only the configured marks, one space after each, none before."""
    text = _ELL_RE.sub(_ELL, text)                 # protect ellipses first
    text = text.translate(_PUNCT_CANON)
    text = _NON_FA_KEEP.sub(" ", text)
    text = _WS.sub(" ", text)
    text = _PUNCT_RUN.sub(r"\1", text)             # "، ." -> "،"
    text = _SPACE_BEFORE.sub(r"\1", text)
    text = _SPACE_AFTER.sub(r"\1 ", text)
    text = _LEADING_PUNCT.sub("", text)            # never open with a mark
    text = text.replace(_ELL, _ELL_OUT)
    return _WS.sub(" ", text).strip()


def normalize_fa(text, spoken_numbers=True, keep_punct=False):
    """Stage 2: the final ground-truth form.

    keep_punct=False (default) is the *matching* form used for every comparison
    against Whisper output: letters, digits-as-words and spaces only.
    keep_punct=True additionally keeps config.KEEP_PUNCT_CHARS — this is the
    form that lands in the dataset.
    """
    text = _base_normalize(text)
    if spoken_numbers:
        text = _digits_to_words(text)
    if keep_punct and _PUNCT_KEEP:
        return _apply_punct(text)
    return _WS.sub(" ", _NON_FA.sub(" ", text)).strip()


def strip_punct(text):
    """Punctuated (display) form -> matching form. Inverse of the keep_punct
    branch above; see the module docstring's invariant."""
    if _PUNCT_STRIP is None:
        return _WS.sub(" ", text).strip()
    return _WS.sub(" ", _PUNCT_STRIP.sub(" ", text)).strip()
