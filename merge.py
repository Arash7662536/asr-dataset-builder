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


def merge(out_dir):
    from datasets import concatenate_datasets
    shard_glob = os.path.join(out_dir, "shards", "*")

    keep = _load_all(shard_glob, "keep_ds")
    review = _load_all(shard_glob, "review_ds")

    if keep:
        ds = concatenate_datasets(keep)
        ds.save_to_disk(os.path.join(out_dir, "dataset_keep"))
        hrs = sum(ds["metrics"][i]["duration"] for i in range(len(ds))) / 3600 \
            if len(ds) else 0.0
        print(f"dataset_keep:   {len(ds)} clips (~{hrs:.2f} h)")
    else:
        print("dataset_keep:   0 clips")

    if review:
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
