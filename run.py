#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator. Discovers books under --root (each <book>/audio/*.mp3 + <book>/
text/*.epub|*.pdf), distributes them across one worker process PER GPU, and
merges the per-book Arrow shards at the end.

Concurrency model
-----------------
* WITHIN a machine: one process per GPU (--gpus 0,1,2,3), pulling from a shared
  queue (dynamic load balancing). One process per GPU — large-v3 is compute
  bound, so two processes on one GPU just time-slice.
* ACROSS machines (multiple vast.ai instances): give each instance a slice with
  --num-shards N --shard-id k. Each instance then spreads its slice over its own
  --gpus. Point every instance's --out at its own dir, then collect/merge.

Idempotent: a book whose <out>/shards/<book>/_DONE exists is skipped, so a
preempted spot instance can resume.

Example
-------
    python run.py \
        --root /workspace/part_02/extracted \
        --out  /workspace/out \
        --whisper-model /workspace/models/my-ct2-whisper \
        --gpus 0,1,2,3 \
        --compute-type float16 \
        --roundtrip-beam-size 5

    # across 2 machines:
    #   machine A: ... --num-shards 2 --shard-id 0
    #   machine B: ... --num-shards 2 --shard-id 1
"""

import argparse
import glob
import multiprocessing as mp
import os
import sys

from config import (Settings, ALIGNER, COMPUTE_TYPE, MIN_MATCH, MAX_CER, MIN_LCS,
                    LEN_TOL, ROUNDTRIP_BEAM_SIZE, LOCATE_BEAM_SIZE)
from asrbuild.worker import worker_main
from asrbuild.logging_utils import get_logger


def discover_books(root):
    """A book dir is any immediate subdir of root that has an audio/ folder."""
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isdir(d) and os.path.isdir(os.path.join(d, "audio")):
            out.append(d)
    return out


def shard(books, num_shards, shard_id):
    if num_shards <= 1:
        return books
    return [b for i, b in enumerate(books) if i % num_shards == shard_id]


def build_settings(args):
    s = Settings()
    s.whisper_model = args.whisper_model
    s.compute_type = args.compute_type
    s.ctc_model = args.ctc_model
    s.aligner = args.aligner
    s.min_match = args.min_match
    s.use_vad_snap = args.vad_snap
    s.run_roundtrip = not args.no_roundtrip
    s.roundtrip_beam_size = args.roundtrip_beam_size
    s.locate_beam_size = args.locate_beam_size
    s.condition_on_previous_text = args.condition_on_previous_text
    s.roundtrip_vad = not args.no_roundtrip_vad
    s.roundtrip_vad_min_silence_ms = args.roundtrip_vad_min_silence_ms
    s.max_cer = args.max_cer
    s.min_lcs = args.min_lcs
    s.len_tol = args.len_tol
    s.enable_precision_review = args.precision_review
    s.keep_rejected = args.keep_rejected
    if args.export_review:
        s.tiers_to_export = ("keep", "review")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="dir of <book>/audio + <book>/text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--whisper-model", required=True,
                    help="path to your LOCAL ct2 whisper directory")
    ap.add_argument("--gpus", default="0",
                    help="comma list, one worker per GPU, e.g. 0,1,2,3")
    ap.add_argument("--compute-type", default=COMPUTE_TYPE)
    ap.add_argument("--ctc-model", default=None)
    ap.add_argument("--aligner", choices=["whisper", "ctc"], default=ALIGNER)
    ap.add_argument("--min-match", type=float, default=MIN_MATCH)

    ap.add_argument("--no-roundtrip", action="store_true")
    ap.add_argument("--roundtrip-beam-size", type=int, default=ROUNDTRIP_BEAM_SIZE)
    ap.add_argument("--locate-beam-size", type=int, default=LOCATE_BEAM_SIZE)
    ap.add_argument("--condition-on-previous-text", action="store_true",
                    help="(default off) re-enable Whisper context carry-over")
    ap.add_argument("--no-roundtrip-vad", action="store_true",
                    help="disable VAD on the round-trip pass")
    ap.add_argument("--roundtrip-vad-min-silence-ms", type=int, default=500)

    ap.add_argument("--max-cer", type=float, default=MAX_CER)
    ap.add_argument("--min-lcs", type=float, default=MIN_LCS)
    ap.add_argument("--len-tol", type=float, default=LEN_TOL)
    ap.add_argument("--precision-review", action="store_true",
                    help="route high-precision/low-recall (likely truncations) "
                         "to review instead of drop")
    ap.add_argument("--export-review", action="store_true",
                    help="include review tier in the training dataset too")
    ap.add_argument("--vad-snap", action="store_true",
                    help="snap chunk edges to silero-vad boundaries")
    ap.add_argument("--keep-rejected", action="store_true")

    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--no-merge", action="store_true",
                    help="skip the final merge (run merge.py later)")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    log = get_logger("main", os.path.join(args.out, "logs", "main.log"))

    books = discover_books(args.root)
    books = shard(books, args.num_shards, args.shard_id)
    if not books:
        sys.exit("no books found (need <root>/<book>/audio/*.mp3)")
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip() != ""]
    log.info("%d books · %d gpu workers (%s) · shard %d/%d",
             len(books), len(gpus), ",".join(gpus), args.shard_id, args.num_shards)

    settings = build_settings(args)
    if settings.aligner == "ctc" and not settings.ctc_model:
        sys.exit("--aligner ctc requires --ctc-model")

    ctx = mp.get_context("spawn")
    book_q = ctx.Queue()
    result_q = ctx.Queue()
    for b in books:
        book_q.put(b)

    procs = []
    for pos, gpu in enumerate(gpus):
        p = ctx.Process(target=worker_main,
                        args=(gpu, book_q, result_q, args.out, settings, pos))
        p.start()
        procs.append(p)

    # collect results as they arrive
    done = 0
    n_keep = n_review = 0
    keep_h = 0.0
    expected = len(books)
    while done < expected:
        try:
            r = result_q.get(timeout=1)
        except Exception:
            if not any(p.is_alive() for p in procs):
                break
            continue
        done += 1
        n_keep += r.get("keep", 0)
        n_review += r.get("review", 0)
        keep_h += r.get("keep_hours", 0.0)
        log.info("[%d/%d] %s -> %s (keep %d, review %d)",
                 done, expected, r.get("book"), r.get("status"),
                 r.get("keep", 0), r.get("review", 0))

    for p in procs:
        p.join()

    log.info("ALL WORKERS DONE — keep %d clips (~%.2f h) · review %d",
             n_keep, keep_h, n_review)

    if not args.no_merge:
        log.info("merging shards ...")
        from merge import merge as merge_shards
        merge_shards(args.out)
    log.info("finished. datasets in %s/dataset_keep (+ dataset_review)", args.out)


if __name__ == "__main__":
    main()
