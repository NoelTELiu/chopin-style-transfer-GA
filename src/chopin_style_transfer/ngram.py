"""Pitch n-gram utilities used by the genetic algorithm."""

from __future__ import annotations

import ast
import random
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable

NGramModel = DefaultDict[tuple[int, ...], DefaultDict[int, float]]


def build_ngram_model(pitches: Iterable[int], n: int = 2) -> NGramModel:
    """Build an n-gram pitch model with 0.1 additive support for MIDI pitches."""
    pitches = list(pitches)
    model: NGramModel = defaultdict(lambda: defaultdict(int))
    for i in range(len(pitches) - n + 1):
        context = tuple(pitches[i : i + n - 1])
        if context not in model:
            for p in range(128):
                model[context][p] = 0.1
        next_note = pitches[i + n - 1]
        model[context][next_note] += 1
    return model



def build_ngram_model_from_sequences(sequences: Iterable[Iterable[int]], n: int = 2) -> NGramModel:
    """Build one smoothed n-gram model without creating transitions across piece boundaries."""
    model: NGramModel = defaultdict(lambda: defaultdict(int))
    for pitches in sequences:
        seq = list(pitches)
        for i in range(len(seq) - n + 1):
            context = tuple(seq[i : i + n - 1])
            if context not in model:
                for p in range(128):
                    model[context][p] = 0.1
            next_note = seq[i + n - 1]
            model[context][next_note] += 1
    return model

def write_ngram_model_to_file(model: NGramModel, filepath: str | Path) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        for context, next_notes in model.items():
            f.write(f"{context}: {dict(next_notes)}\n")


def read_ngram_model_from_file(filepath: str | Path) -> NGramModel:
    """Load a model written by :func:`write_ngram_model_to_file`.

    ``ast.literal_eval`` is used instead of ``eval`` so model files are treated as data,
    not executable Python.
    """
    model: NGramModel = defaultdict(lambda: defaultdict(int))
    with Path(filepath).open("r", encoding="utf-8") as f:
        for line in f:
            context_str, next_notes_str = line.strip().split(": ", 1)
            context = tuple(ast.literal_eval(context_str))
            next_notes = ast.literal_eval(next_notes_str)
            for note, count in next_notes.items():
                model[context][int(note)] = float(count)
    return model


def predict_next_pitch(ngram_model: NGramModel, context: list[int], n: int = 2) -> int | None:
    """Predict the next pitch using backoff from n-gram to shorter contexts."""
    for backoff_n in range(n, 1, -1):
        if len(context) < backoff_n - 1:
            continue
        key = tuple(context[-(backoff_n - 1) :])
        options = ngram_model.get(key)
        if options:
            candidates = list(options.keys())
            weights = list(options.values())
            return random.choices(candidates, weights=weights)[0]
    return None


def extract_valid_pitch_context(individual: list[list[int]], idx: int, n: int) -> list[int]:
    """Extract up to ``n-1`` preceding single-note pitches from a chromosome."""
    context: list[int] = []
    for i in range(max(0, idx - n + 1), idx):
        frame = individual[i]
        if isinstance(frame, list) and len(frame) == 1 and isinstance(frame[0], int):
            pitch = frame[0]
            if pitch not in (-1, -3):
                context.append(pitch % 128)
    return context
