#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B. faster-whisper inference wrappers around your LOCAL ct2 model.

Three distinct passes, each tuned differently:
  rough_transcribe        -> locating only; cheap (beam 1) + VAD.
  whisper_word_timestamps -> alignment; word timestamps, VAD OFF (boundaries!).
  transcribe_chunk        -> round-trip verification; HARDENED (fix 4) + VAD.
"""

from config import SR
from .normalize import normalize_fa


def load_whisper(model_dir, device, compute_type, device_index=0):
    """model_dir is your local CTranslate2 directory (already converted)."""
    from faster_whisper import WhisperModel
    return WhisperModel(model_dir, device=device, device_index=device_index,
                        compute_type=compute_type)


def rough_transcribe(whisper, mp3_path, beam_size=1):
    """Locating pass. VAD on; only used to find where the file sits in the book."""
    segments, _ = whisper.transcribe(mp3_path, language="fa",
                                     beam_size=beam_size, vad_filter=True)
    return normalize_fa(" ".join(seg.text for seg in segments))


def whisper_word_timestamps(whisper, audio):
    """Full-file transcription with per-word timestamps, normalized.
    VAD is OFF here on purpose: VAD-shifted timestamps would corrupt boundaries."""
    segs, _ = whisper.transcribe(audio, language="fa", beam_size=1,
                                 word_timestamps=True, vad_filter=False)
    words = []
    for s in segs:
        for w in (s.words or []):
            for tok in normalize_fa(w.word).split():         # a word may split
                words.append((tok, float(w.start), float(w.end)))
    return words


def transcribe_chunk(whisper, wav_path, beam_size=5,
                     condition_on_previous_text=False,
                     vad_filter=True, vad_min_silence_ms=500):
    """Round-trip verification pass — HARDENED.

    fix (4): beam_size>1 + condition_on_previous_text=False cut the premature
             end-of-segment truncations ("سایم گفت" for a 50-word clip).
    VAD:     strip silence/music that pushes the decoder to stop early.
    """
    vad_params = dict(min_silence_duration_ms=vad_min_silence_ms) if vad_filter else None
    segments, _ = whisper.transcribe(
        wav_path,
        language="fa",
        beam_size=beam_size,
        condition_on_previous_text=condition_on_previous_text,
        vad_filter=vad_filter,
        vad_parameters=vad_params,
    )
    return normalize_fa(" ".join(seg.text for seg in segments))
