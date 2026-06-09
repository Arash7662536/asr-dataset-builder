#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-worker logging + a small stage timer that logs A..F boundaries."""

import logging
import os
import sys
import time
from contextlib import contextmanager

STAGE_NAMES = {
    "A": "extract text",
    "B": "rough-transcribe",
    "C": "map MP3s -> ranges",
    "D": "align",
    "E": "merge chunks",
    "F": "export + filter",
}


def get_logger(name, log_path=None, level=logging.INFO):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter(f"%(asctime)s [{name}] %(levelname)s %(message)s",
                            "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


@contextmanager
def stage(log, book_id, letter, extra=""):
    """Log '[book][A] extract text ...' on enter and '... done (1.23s)' on exit."""
    desc = STAGE_NAMES.get(letter, letter)
    head = f"[{book_id}][{letter}] {desc}"
    log.info("%s %s", head, extra)
    t0 = time.time()
    try:
        yield
    finally:
        log.info("%s done (%.2fs)", head, time.time() - t0)
