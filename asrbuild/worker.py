#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A worker process pinned to ONE GPU. It loads the models once, then drains a
shared queue of book directories. One worker per GPU; books are load-balanced
dynamically (longer books don't stall a static shard).
"""

import os
import queue as _queue


def worker_main(gpu_id, book_queue, result_queue, out_dir, settings, position):
    # Pin BEFORE importing torch / faster-whisper.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from asrbuild.logging_utils import get_logger
    from asrbuild.whisper_io import load_whisper
    from asrbuild.pipeline import Models, process_book

    # position is globally unique across workers, so two workers pinned to the
    # same physical GPU still get their own logger + log file.
    name = f"gpu{gpu_id}-w{position}"
    log = get_logger(name, os.path.join(out_dir, "logs", f"worker_{name}.log"))
    log.info("worker up; loading models (compute=%s, aligner=%s)",
             settings.compute_type, settings.aligner)

    # device index is 0 because CUDA_VISIBLE_DEVICES masks everything else.
    whisper = load_whisper(settings.whisper_model, "cuda",
                           settings.compute_type, device_index=0)

    ctc = None
    if settings.aligner == "ctc" or settings.ctc_fallback_to_whisper:
        try:
            from asrbuild.align import load_ctc
            if settings.aligner == "ctc":
                ctc = load_ctc(settings.ctc_model, "cuda")
        except Exception as e:                               # fallback path stays whisper
            log.warning("CTC model not loaded (%s); whisper aligner only", e)

    vad = None
    if settings.use_vad_snap:
        try:
            from asrbuild.chunk import load_vad
            vad = load_vad()
        except Exception as e:
            log.warning("VAD not loaded (%s); skipping snap", e)

    models = Models(whisper=whisper, ctc=ctc, vad=vad)
    shards_root = os.path.join(out_dir, "shards")

    processed = 0
    while True:
        try:
            book_dir = book_queue.get_nowait()
        except _queue.Empty:
            break
        book_id = os.path.basename(book_dir.rstrip("/"))
        shard_dir = os.path.join(shards_root, book_id)
        if os.path.exists(os.path.join(shard_dir, "_DONE")):
            log.info("[%s] already done, skip", book_id)
            continue
        try:
            summary = process_book(book_dir, shard_dir, settings, models, log,
                                   position=position)
            result_queue.put(summary)
            processed += 1
        except Exception as e:                               # never kill the worker
            log.exception("[%s] FAILED: %s", book_id, e)
            result_queue.put({"book": book_id, "status": "error", "error": str(e),
                              "keep": 0, "review": 0})

    log.info("worker done; processed %d books", processed)
