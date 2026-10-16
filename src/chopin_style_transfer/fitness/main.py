


from __future__ import annotations

import math
from dataclasses import dataclass

from .utility import (
    ORNAMENT_CLASSES, SIMILARITY_WEIGHTS, SYNTAX_WEIGHTS,
    process_chord_data, get_scale, normalized_js,
    analyze_pitch, analyze_rhythm, analyze_harmony, analyze_syntax,
)
from .structure import analyze_structure, structure_similarity


@dataclass(frozen=True)
class FitnessWeights:
    """Weights of the four fitness components."""

    target: float = 0.9
    structure: float = 0.1
    input: float = 1.0
    syntax: float = 1.0



FITNESS_WEIGHT_PROFILES: dict[str, FitnessWeights] = {
    "balanced": FitnessWeights(target=0.9, structure=0.1, input=1.0, syntax=1.0),
    "target": FitnessWeights(target=1.4, structure=0.1, input=0.5, syntax=1.0),
    "syntax": FitnessWeights(target=0.45, structure=0.05, input=1.0, syntax=1.5),
    "input": FitnessWeights(target=0.45, structure=0.05, input=2.0, syntax=0.5),
}


def get_fitness_weights(profile: str = "balanced") -> FitnessWeights:
    try:
        return FITNESS_WEIGHT_PROFILES[profile]
    except KeyError as exc:
        valid = ", ".join(sorted(FITNESS_WEIGHT_PROFILES))
        raise ValueError(f"Unknown fitness-weight profile {profile!r}; choose one of: {valid}") from exc





def process_input_data(data):
    """Flatten the time-grid representation into analysis features."""

    right = list(data["right_hand"])
    left = list(data["left_hand"])

    onsets = {"right_hand": [], "left_hand": []}
    pitches = {"right_hand": [], "left_hand": []}
    durations = {"right_hand": [], "left_hand": []}
    ornaments = {"right_hand": [], "left_hand": []}
    clean_notes = {"right_hand": [], "left_hand": []}

    length = max(len(right), len(left))
    right += ([[]] * (length - len(right)))
    left += ([[]] * (length - len(left)))
    hands = {"right_hand": right, "left_hand": left}

    for hand in ["right_hand", "left_hand"]:
        rest_flag = False
        last_notes = []
        hand_data = hands[hand]
        for time, notes in enumerate(hand_data):
            if not notes:
                rest_flag = True
                clean_notes[hand].append([])
            elif notes == [-3]:
                clean_notes[hand].append([])
            else:
                tuplet_flag = False
                if len(notes) > 2:
                    next_idx = time + 1
                    if next_idx < len(hand_data) and hand_data[next_idx] == [-3]:
                        tuplet_flag = True

                if tuplet_flag:
                    tuplet_duration = 1
                    clean_notes[hand].append([])
                    for i in range(time + 1, len(hand_data)):
                        if hand_data[i] == [-3]:
                            tuplet_duration += 1
                        else:
                            break

                    tuplet_cnt = len(notes)
                    for i, pitch in enumerate(notes):
                        rest_flag = False
                        pitches[hand].append(pitch % 128)
                        tuplet_time = tuplet_duration / tuplet_cnt
                        durations[hand].append(tuplet_time)
                        onsets[hand].append(time + i * tuplet_time)
                        for ornament in ORNAMENT_CLASSES:
                            if pitch >= ornament:
                                ornaments[hand].append(ORNAMENT_CLASSES[ornament])
                                break
                    continue

                if notes == [-1]:
                    if not rest_flag and durations[hand]:
                        durations[hand][-1] += 1
                        clean_notes[hand].append(last_notes)
                    else:
                        clean_notes[hand].append([])
                else:
                    onsets[hand].append(time)
                    base_notes = [pitch % 128 for pitch in notes]
                    clean_notes[hand].append(base_notes)
                    last_notes = base_notes
                    for pitch in notes:
                        rest_flag = False
                        pitches[hand].append(pitch % 128)
                        durations[hand].append(1)
                        for ornament in ORNAMENT_CLASSES:
                            if pitch >= ornament:
                                ornaments[hand].append(ORNAMENT_CLASSES[ornament])
                                break

    return {
        "cleaned_notes": clean_notes,
        "onsets": onsets,
        "pitches": pitches,
        "chords": process_chord_data(clean_notes),
        "durations": durations,
        "ornaments": ornaments,
        "measure_boundaries": list(data.get("measure_boundaries", [])),
    }





def similarity_function(data, target, hand):
    """Feature-based similarity across pitch, rhythm, harmony, and ornament descriptors."""
    score = 0.0
    total_weight = 0.0

    for category, features in SIMILARITY_WEIGHTS[hand].items():
        if category not in target:
            continue

        for feature_name, weight in features.items():
            if weight <= 0:
                continue
            d1 = data.get(category, {}).get(feature_name, {})
            d2 = target.get(category, {}).get(feature_name, {})
            if d1 in ({}, None) or d2 in ({}, None):
                continue
            if "chord" not in feature_name:
                d1 = d1.get(hand, {}) if isinstance(d1, dict) else d1
                d2 = d2.get(hand, {}) if isinstance(d2, dict) else d2
                if d1 in ({}, None) or d2 in ({}, None):
                    continue

            if "histogram" in feature_name or "distribution" in feature_name:
                dist = normalized_js(d1, d2)
            else:
                dist = abs(d1 - d2) / max(abs(d1), abs(d2)) if (d1 or d2) else 0.0

            scale = get_scale(category, feature_name)
            score += weight * math.exp(-dist * scale)
            total_weight += weight

    return score / total_weight if total_weight > 0 else 0.0


def syntax_function(data, hand):
    """Normalized syntax score from melodic, tonal, and chord consonance."""
    score = 0.0
    total_weight = 0.0

    for feature_name, weight in SYNTAX_WEIGHTS[hand].items():
        feature_score = data.get("syntax", {}).get(feature_name, {})
        if feature_score in ({}, None):
            continue
        if "chord" not in feature_name:
            feature_score = feature_score.get(hand, {}) if isinstance(feature_score, dict) else feature_score
            if feature_score in ({}, None):
                continue
        scale = get_scale("syntax", feature_name)
        score += weight * feature_score * scale
        total_weight += weight

    return score / total_weight if total_weight > 0 else 0.0


def fitness_components(data, target, hand, origin=None, *, include_input=True):
    """Return target, structure, input, and syntax component scores."""
    if hand not in ["right", "left"]:
        raise ValueError("Hand must be 'right' or 'left'.")
    hand_key = "right_hand" if hand == "right" else "left_hand"

    target_similarity = similarity_function(data, target, hand_key)
    input_similarity = similarity_function(data, origin, hand_key) if (origin and include_input) else 0.0
    syntax_score = syntax_function(data, hand_key)
    structure_score = 0.0
    if "structure" in data and "structure" in target:
        structure_score = structure_similarity(data["structure"], target["structure"], alignment_weight=0.7)

    return {
        "target_similarity": float(target_similarity),
        "structure_score": float(structure_score),
        "input_similarity": float(input_similarity),
        "syntax_score": float(syntax_score),
    }


def combine_fitness_components(components, weights: FitnessWeights) -> float:
    """Weighted linear combination of the four fitness components."""
    return (
        weights.target * components["target_similarity"]
        + weights.structure * components["structure_score"]
        + weights.input * components["input_similarity"]
        + weights.syntax * components["syntax_score"]
    )


def fitness_function(
    data,
    target,
    hand,
    origin,
    gen=0,
    *,
    threshold=400,
    weights: FitnessWeights | None = None,
    return_components=False,
):
    """Two-phase multi-objective fitness.

    Before ``threshold``, input preservation is disabled. From ``threshold`` onward,
    the input-similarity term is added to preserve the source material.
    """
    weights = weights or FITNESS_WEIGHT_PROFILES["balanced"]
    input_active = origin is not None and gen >= threshold
    components = fitness_components(data, target, hand, origin, include_input=input_active)
    total = combine_fitness_components(components, weights)
    if return_components:
        return total, components
    return total


def analyze_function(data, ignore_syntax=False, include_structure=True):
    """Analyze a chromosome using feature, structural, and musical-syntax descriptors."""
    processed_data = process_input_data(data)
    analyzed_data = {
        "pitch": analyze_pitch(processed_data),
        "rhythm": analyze_rhythm(processed_data),
        "harmony": analyze_harmony(processed_data),
    }
    if include_structure:

        analyzed_data["structure"] = analyze_structure(processed_data, novelty_mode="ssm")
    if not ignore_syntax:
        analyzed_data["syntax"] = analyze_syntax(processed_data, analyzed_data["harmony"])
    return analyzed_data
