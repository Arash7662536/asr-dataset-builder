#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D. Force/soft alignment of book sentences to audio.

align_via_whisper : match-based; robust to twisted narration (default).
align_file (ctc)  : CTC-segmentation forced alignment; best for verbatim audio.
                    Guarded against the "Audio is shorter than text!" assertion.
"""

import difflib

import numpy as np

from config import SR, INFER_WINDOW_S
from .dtypes import AlignedSeg
from .whisper_io import whisper_word_timestamps


# --------------------------------------------------------------------------- #
# D'. Whisper word-timestamp alignment (default)
# --------------------------------------------------------------------------- #
def _interp(arr):
    """Fill None entries by linear interpolation between known neighbors;
    edges extend the nearest known value."""
    n = len(arr)
    known = [k for k in range(n) if arr[k] is not None]
    if not known:
        return
    for k in range(known[0]):
        arr[k] = arr[known[0]]
    for k in range(known[-1] + 1, n):
        arr[k] = arr[known[-1]]
    for a, b in zip(known, known[1:]):
        if b > a + 1:
            va, vb = arr[a], arr[b]
            for k in range(a + 1, b):
                arr[k] = va + (vb - va) * (k - a) / (b - a)


def align_via_whisper(whisper, audio, sentences, min_match=0.3):
    """Align book sentences to audio via Whisper word timestamps + word-level
    text alignment. score = match ratio (0..1)."""
    ww = whisper_word_timestamps(whisper, audio)
    if not ww:
        return []
    W = [w[0] for w in ww]
    Wt0 = [w[1] for w in ww]
    Wt1 = [w[2] for w in ww]

    B, owner = [], []
    for si, s in enumerate(sentences):
        for tok in s.text.split():
            B.append(tok)
            owner.append(si)
    if not B:
        return []

    sm = difflib.SequenceMatcher(None, B, W, autojunk=False)
    bt0 = [None] * len(B)
    bt1 = [None] * len(B)
    matched = [False] * len(B)
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            bt0[i + k] = Wt0[j + k]
            bt1[i + k] = Wt1[j + k]
            matched[i + k] = True
    _interp(bt0)
    _interp(bt1)

    n_sent = len(sentences)
    first_t = [None] * n_sent
    last_t = [None] * n_sent
    n_tot = [0] * n_sent
    n_match = [0] * n_sent
    for idx in range(len(B)):
        si = owner[idx]
        n_tot[si] += 1
        if matched[idx]:
            n_match[si] += 1
        if first_t[si] is None:
            first_t[si] = bt0[idx]
        last_t[si] = bt1[idx]

    out = []
    for si, s in enumerate(sentences):
        if n_tot[si] == 0 or first_t[si] is None or last_t[si] is None:
            continue
        ratio = n_match[si] / n_tot[si]
        if ratio < min_match:                                # likely not spoken here
            continue
        start, end = first_t[si], last_t[si]
        if end <= start:
            continue
        out.append(AlignedSeg(text=s.text, start=start, end=end, score=ratio))
    out.sort(key=lambda a: a.start)
    return out


# --------------------------------------------------------------------------- #
# D. CTC-segmentation forced alignment (optional; guarded)
# --------------------------------------------------------------------------- #
class CtcModel:
    def __init__(self, model, processor, vocab, blank_id, char_list, device):
        self.model = model
        self.processor = processor
        self.vocab = vocab
        self.blank_id = blank_id
        self.char_list = char_list
        self.device = device


def load_ctc(model_name, device):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device).eval()
    vocab = processor.tokenizer.get_vocab()
    char_list = [tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    blank_id = processor.tokenizer.pad_token_id
    if blank_id is None:
        blank_id = vocab.get("<pad>", 0)
    return CtcModel(model, processor, vocab, blank_id, char_list, device)


def get_logprobs(ctc, audio):
    """Windowed inference; concatenate frame log-probs."""
    import torch
    win = INFER_WINDOW_S * SR
    out = []
    for start in range(0, len(audio), win):
        chunk = audio[start:start + win]
        if len(chunk) < SR // 10:
            continue
        inputs = ctc.processor(chunk, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            logits = ctc.model(inputs.input_values.to(ctc.device)).logits
        lp = torch.log_softmax(logits, dim=-1)[0].cpu().numpy().astype(np.float32)
        out.append(lp)
    return (np.concatenate(out, axis=0) if out
            else np.zeros((0, len(ctc.char_list)), dtype=np.float32))


def _tokenize(text, vocab, word_delim="|"):
    ids = []
    for ch in text:
        if ch in (" ", "\u200c"):
            ch = word_delim
        if ch in vocab:
            ids.append(vocab[ch])
    return np.array(ids, dtype=np.int64)


def align_file(ctc, audio, sentences, log=None):
    """Force-align book sentences to audio with CTC-segmentation.

    Guards the well-known `AssertionError: Audio is shorter than text!`:
    that fires when total text tokens exceed the available CTC frames (more
    text than the audio can possibly hold — usually a wrong/oversized mapping).
    On that condition we log and return [] so the caller can fall back."""
    import ctc_segmentation as cs

    logprobs = get_logprobs(ctc, audio)
    n_frames = logprobs.shape[0]
    if n_frames == 0:
        return []

    config = cs.CtcSegmentationParameters(char_list=ctc.char_list)
    config.blank = ctc.blank_id
    config.index_duration = (len(audio) / SR) / n_frames

    texts = [s.text for s in sentences]
    word_delim = "|" if "|" in ctc.vocab else " "
    token_list = [_tokenize(t, ctc.vocab, word_delim) for t in texts]

    keep = [i for i, tk in enumerate(token_list) if len(tk) > 0]
    texts = [texts[i] for i in keep]
    token_list = [token_list[i] for i in keep]
    if not token_list:
        return []

    # --- guard against "Audio is shorter than text!" ---
    total_tokens = int(sum(len(tk) for tk in token_list))
    if total_tokens >= n_frames:
        if log:
            log.warning("CTC: %d text tokens >= %d audio frames "
                        "('Audio is shorter than text!'); range likely "
                        "over-assigned. Skipping CTC for this file.",
                        total_tokens, n_frames)
        return []

    try:
        ground_truth_mat, utt_begin_indices = cs.prepare_token_list(config, token_list)
        timings, char_probs, _ = cs.ctc_segmentation(config, logprobs, ground_truth_mat)
        segments = cs.determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, texts
        )
    except AssertionError as e:                              # belt-and-suspenders
        if log:
            log.warning("CTC assertion (%s); skipping CTC for this file.", e)
        return []

    out = []
    for text, (start, end, score) in zip(texts, segments):
        out.append(AlignedSeg(text=text, start=float(start),
                              end=float(end), score=float(score)))
    return out
