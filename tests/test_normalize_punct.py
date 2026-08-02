#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guards the punctuation invariant.

    strip_punct(normalize_fa(t, keep_punct=True)) == normalize_fa(t)

If this ever breaks, keeping punctuation has started to change the matching
form — which would silently shift alignment scores and keep/drop tiers.

Run with `python tests/test_normalize_punct.py` (no pytest needed) or
`pytest tests/`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asrbuild.normalize import normalize_fa, strip_punct   # noqa: E402

SAMPLES = [
    "سلام، حال شما چطور است؟",
    "او گفت: «من نمی‌دانم!»",
    "این یک جمله است. و این جمله دوم؟ بله!",
    "چه خبر؟؟؟ هیچ...",
    "سایم گفت ... بعد رفت.",
    "بدون‌فاصله،چسبیده به کاما",
    "؛ شروع با علامت",
    "متن با  فاصله   زیاد  .",
    "عدد ۱۲۳ و 456 در متن.",
    "hello mixed لاتین (پرانتز) «گیومه» ـ تطویل",
    "یک — خط تیره ی بلند، و یک - کوتاه.",
    "آيا كاراكتر عربي تبديل مي‌شود؟",
    "!!!",
    "...",
    "",
    "تک",
]


def check(sample):
    disp = normalize_fa(sample, keep_punct=True)
    plain = normalize_fa(sample, keep_punct=False)
    assert strip_punct(disp) == plain, (
        f"invariant broken\n  in    : {sample!r}\n"
        f"  disp  : {disp!r}\n  strip : {strip_punct(disp)!r}\n"
        f"  plain : {plain!r}")
    return disp, plain


def test_invariant():
    for s in SAMPLES:
        check(s)


def test_punctuation_is_actually_kept():
    disp = normalize_fa("سلام، حال شما چطور است؟", keep_punct=True)
    assert "،" in disp and "؟" in disp
    assert disp.endswith("؟")
    # no space before a mark, exactly one after
    assert " ،" not in disp
    assert "،ح" not in disp


def test_ascii_marks_folded_to_persian():
    disp = normalize_fa("خوبی? بله, حتما;", keep_punct=True)
    assert "?" not in disp and "," not in disp and ";" not in disp
    assert "؟" in disp and "،" in disp and "؛" in disp


def test_ellipsis_survives_as_one_token():
    disp = normalize_fa("خب... بعد چه شد؟", keep_punct=True)
    assert "..." in disp
    assert "...." not in disp


def test_no_leading_mark():
    assert not normalize_fa("، بعد از آن رفت.", keep_punct=True).startswith("،")


def test_runs_collapse():
    assert normalize_fa("چرا؟!", keep_punct=True).count("؟") == 1
    assert "!" not in normalize_fa("چرا؟!", keep_punct=True)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nexamples:")
    for s in SAMPLES[:9]:
        disp, plain = check(s)
        print(f"  in    {s}\n  keep  {disp}\n  plain {plain}\n")
