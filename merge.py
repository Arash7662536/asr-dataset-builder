#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge per-book Arrow shards into final datasets:

    <out>/dataset_keep     (training data)
    <out>/dataset_review   (human-triage data)

Safe to run repeatedly (after a partial run, or to refresh after more books
finish). Reads every <out>/shards/<book>/{keep_ds,review_ds}.
"""

import argparse
import glob
import os


def _load_all(shard_glob, sub):
    from datasets import load_from_disk
    parts = []
    for shard in sorted(glob.glob(shard_glob)):
        path = os.path.join(shard, sub)
        if os.path.isdir(path):
            try:
                parts.append(load_from_disk(path))
            except Exception as e:
                print(f"  warn: could not load {path}: {e}")
    return parts


def _check_same_schema(parts, sub):
    """concatenate_datasets() fails obscurely on mixed schemas. The usual cause
    is shards from before the `text_plain` (punctuation) column existed sitting
    next to new ones, so say that plainly instead."""
    cols = {tuple(sorted(p.column_names)) for p in parts}
    if len(cols) > 1:
        raise SystemExit(
            f"error: {sub} shards have different columns: "
            + " vs ".join(str(list(c)) for c in cols)
            + "\n  Shards written before/after the punctuation change cannot be "
              "merged.\n  Delete the older <out>/shards/* (or merge them into a "
              "separate dataset) and re-run.")


def merge(out_dir):
    from datasets import concatenate_datasets
    shard_glob = os.path.join(out_dir, "shards", "*")

    keep = _load_all(shard_glob, "keep_ds")
    review = _load_all(shard_glob, "review_ds")

    if keep:
        _check_same_schema(keep, "keep_ds")
        ds = concatenate_datasets(keep)
        ds.save_to_disk(os.path.join(out_dir, "dataset_keep"))
        hrs = sum(ds["metrics"][i]["duration"] for i in range(len(ds))) / 3600 \
            if len(ds) else 0.0
        print(f"dataset_keep:   {len(ds)} clips (~{hrs:.2f} h)")
    else:
        print("dataset_keep:   0 clips")

    if review:
        _check_same_schema(review, "review_ds")
        rv = concatenate_datasets(review)
        rv.save_to_disk(os.path.join(out_dir, "dataset_review"))
        print(f"dataset_review: {len(rv)} clips")
    else:
        print("dataset_review: 0 clips")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dir used by run.py")
    args = ap.parse_args()
    merge(args.out)
