# asr-dataset-builder

Build a Whisper-ready ASR dataset from Persian audiobooks: each book is an
EPUB/PDF + a folder of MP3s. The ground-truth text always comes from the **book**,
never from Whisper. Output is an **Arrow dataset** (`audio`, `text`, `metrics`).

This is the *new* alignment setup (Whisper word-timestamp matching by default,
robust to twisted/paraphrased narration), with the round-trip verifier
**hardened** against premature truncations and silence/music.

## Input layout

```
<root>/
  130127/
    audio/  09b1....mp3  61c9....mp3  ...
    text/   131179.epub          # or a .pdf, or (skipped) missing
  130241/
    ...
```

## Output

```
<out>/
  dataset_keep/      # Arrow dataset — training data (keep tier)
  dataset_review/    # Arrow dataset — borderline clips for human triage
  shards/<book>/     # per-book: keep_ds/ review_ds/ audio/*.wav audit.jsonl _DONE
  logs/              # main.log + worker_gpu*.log
```

Each row:

| column    | type                                             |
|-----------|--------------------------------------------------|
| `audio`   | HF `Audio` → decodes to `{path, array, sampling_rate}` (16 kHz) |
| `text`    | `string` — book ground truth                     |
| `metrics` | struct — `tier, match_ratio, cer, wer, lcs_ratio, len_ratio, precision, duration, source_mp3, hyp` |

Load it later with:

```python
from datasets import load_from_disk
ds = load_from_disk("out/dataset_keep")
ds[0]["audio"]    # {'path':..., 'array': np.float32[...], 'sampling_rate':16000}
ds[0]["text"]     # 'سایم گفت ...'
ds[0]["metrics"]  # {'tier':'keep','cer':0.04, ...}
```

## Run

```bash
pip install -r requirements.txt

python run.py \
  --root /workspace/part_02/extracted \
  --out  /workspace/out \
  --whisper-model /workspace/models/my-ct2-whisper \   # YOUR local ct2 dir
  --gpus 0,1,2,3
```

Across multiple machines (each its own `--out`, then collect):

```bash
# machine A
python run.py ... --num-shards 3 --shard-id 0 --gpus 0,1,2,3
# machine B
python run.py ... --num-shards 3 --shard-id 1 --gpus 0,1,2,3
# machine C
python run.py ... --num-shards 3 --shard-id 2 --gpus 0,1,2,3
```

Re-merge anytime (e.g. after a partial run):

```bash
python merge.py --out /workspace/out
```

## What was changed vs the old script (your requests)

* **Hardened round-trip (fix 4).** `transcribe_chunk` now uses `beam_size=5`
  and `condition_on_previous_text=False`, which turns most `سایم گفت`
  premature-stop truncations into full transcriptions. Tune with
  `--roundtrip-beam-size`.
* **VAD on the round-trip pass.** Silence/music that pushes the decoder to stop
  early is stripped (`min_silence_duration_ms=500`, tune with
  `--roundtrip-vad-min-silence-ms`, disable with `--no-roundtrip-vad`). VAD is
  deliberately **off** in the word-timestamp *alignment* pass, because
  VAD-shifted timestamps would corrupt chunk boundaries.
* **Local ct2 Whisper (note 1).** `--whisper-model` is a path to your converted
  CTranslate2 directory; faster-whisper loads it directly.
* **Duplicate audio (note 2).** Byte-identical MP3s are dropped by md5 per book.
* **Unknown MP3 order (note 3).** rapidfuzz anchors each MP3 into the book and
  the order is recovered by sorting on the matched offset.
* **Missing / PDF text (note 4).** `text/` is resolved to EPUB → PDF (PyMuPDF,
  then pdfplumber) → otherwise the book is skipped with a logged reason (and a
  `_DONE` marker so it isn't retried).
* **`Audio is shorter than text!`** This CTC-segmentation assertion fires when
  the text has more tokens than the audio has frames (usually an over-sized /
  mis-anchored range). It is guarded: the file is logged and skipped, and with
  the default `--aligner whisper` it can't occur at all. With `--aligner ctc`,
  such a file falls back to the Whisper aligner automatically.
* **Per-stage logs + bars.** Every worker logs `[gpu0][130127][A] extract text …
  done (1.2s)` for stages **A–F**, with tqdm bars on the transcribe/per-file
  loops. Each worker also writes its own `logs/worker_gpu*.log` (the reliable
  source of truth, since console bars from parallel workers interleave).

### Truncation note (fix 3, kept as an *optional* safety net)

You preferred fix (4) over fix (3), so precision-based gating is **off** by
default. But `precision = lcs/len(hyp)` is still computed and stored, because it
cleanly separates a *truncation* (high precision, low recall — the audio is fine,
Whisper just stopped) from a *paraphrase* (low precision). If after the beam+VAD
change you still see verbatim clips getting dropped, add `--precision-review` to
route those high-precision/low-recall clips to **review** instead of **drop**.

## Tuning the defaults

All defaults live in `config.py`; the ones you'll touch most:

| knob | flag | default | meaning |
|------|------|---------|---------|
| round-trip beam | `--roundtrip-beam-size` | 5 | ↑ fewer truncations, slower |
| keep CER ceiling | `--max-cer` | 0.30 | lower = stricter keep tier |
| keep word-overlap | `--min-lcs` | 0.85 | higher = more verbatim-only |
| keep length tol | `--len-tol` | 0.25 | catches dropped/added clauses |
| aligner | `--aligner` | whisper | `ctc` for verbatim audiobooks |
| min word match | `--min-match` | 0.30 | drop sentence below this overlap |

## Recommended GPU setup (vast.ai)

The workload is **embarrassingly parallel at the book level** (~300 independent
books), and VRAM is tiny (large-v3 fp16 ≈ 3–5 GB), so the right move is **many
cheap single-GPU workers**, not a few big cards.

* **GPU: RTX 4090** (24 GB) — best fp16 inference throughput per dollar here.
  3090 is a fine cheaper fallback. **Skip A100/H100/L40S** — you'd pay for
  40–80 GB you can't use.
* **Per worker:** ~4–8 vCPUs + 16–32 GB RAM (MP3 decode/resample can starve a
  fast GPU), and **fast local NVMe** (don't read the audio over a network mount).
* **One process per GPU.** large-v3 is compute-bound; two processes per GPU just
  time-slice.
* **How many GPUs is a time knob, not a cost knob.** Total GPU-hours are roughly
  fixed; more GPUs only compress wall-clock. Benchmark one representative book
  to get *wall-hours/book*, multiply by ~300, divide by worker count. As a rough
  placeholder (3 passes, beam-5 round-trip): ~8×4090 → days; ~16–32×4090 →
  faster, same dollar total.
* **Cut total compute** (bigger lever than bigger GPUs): keep large-v3 + beam-5
  only on the round-trip; the locating pass can use a smaller/distilled model and
  `int8_float16`; batch the many short round-trip clips if you extend
  `transcribe_chunk` to faster-whisper's `BatchedInferencePipeline`.

> Verifier caveat for fine-tuning: alignment **and** the round-trip both use
> Whisper. If that Whisper is the model you're training, the filter quietly
> selects clips it already gets right. Consider verifying with a *different*
> model (or `--aligner ctc` + a CTC gate) so you don't discard the hard examples
> the model most needs.
