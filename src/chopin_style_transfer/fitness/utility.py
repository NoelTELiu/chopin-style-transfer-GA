


import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy




ORNAMENT_CLASSES = {
    768: 'lower_mordent',
    640: 'upper_mordent',
    512: 'trill',
    384: 'inverted_turn',
    256: 'turn',
    128: 'acciaccatura',
}

PITCH_CLASSES = {
    0: 'C',
    1: 'C#/Db',
    2: 'D',
    3: 'D#/Eb',
    4: 'E',
    5: 'F',
    6: 'F#/Gb',
    7: 'G',
    8: 'G#/Ab',
    9: 'A',
    10: 'A#/Bb',
    11: 'B'
}

CHORD_VALUES = {
    "Major": 1.0,
    "Minor": 1.0,
    "Diminished": 0.7,
    "Augmented": 0.3,
    "Major 7th": 0.8,
    "Dominant 7th": 0.85,
    "Minor 7th": 0.8,
    "Half-diminished 7th": 0.4,
    "Diminished 7th": 0.5,
    "Minor Major 7th": 0.3,
    "Dominant 9th": 0.4,
    "Dominant 11th": 0.3,
    "Dominant 13th": 0.3,
    "Sus2": 0.2,
    "Sus4": 0.2,
    "Add9": 0.4,
    "Add2": 0.4,
    "Unknown": -1.0
}

SIMILARITY_WEIGHTS = {
    "right_hand": {
        "pitch": {
            "pitch_histogram":          1.0,
            "pitch_class_distribution": 1.0,
            "ambitus":                  0.0,
            "interval_histogram":       1.0,
            "interval_class_histogram": 1.0,
        },
        "rhythm": {
            "note_duration_histogram":  0.8,
            "onset_entropy":            0.8,
            "ornament_density":         0.5,
        },
        "harmony": {
            "chord_distribution":       1.0,
            "chord_type_distribution":  1.0,
        },
    },
    "left_hand": {
        "pitch": {
            "pitch_histogram":          1.0,
            "pitch_class_distribution": 1.0,
            "ambitus":                  0.0,
            "interval_histogram":       1.0,
            "interval_class_histogram": 1.0,
        },
        "rhythm": {
            "note_duration_histogram":  0.8,
            "onset_entropy":            0.8,
            "ornament_density":         0.5,
        },
        "harmony": {
            "chord_distribution":       1.0,
            "chord_type_distribution":  1.0,
        },
    }
}

SYNTAX_WEIGHTS = {
    "right_hand": {
        "melodic_consonance":       1.0,
        "chord_consonance":         1.0,
        "tonal_consistency":        1.0,
    },
    "left_hand": {
        "melodic_consonance":       1.0,
        "chord_consonance":         1.0,
        "tonal_consistency":        1.0,
    }
}

SCALES = {
    "pitch": {
        "pitch_histogram":          10,
        "pitch_class_distribution": 10,
        "ambitus":                  2,
        "interval_histogram":       10,
        "interval_class_histogram": 10,
    },
    "rhythm": {
        "note_duration_histogram":  10,
        "onset_entropy":            20,
        "ornament_density":         1,
    },
    "harmony": {
        "chord_distribution":       10,
        "chord_type_distribution":  10,
    },
    "syntax": {
        "melodic_consonance":       1,
        "chord_consonance":         1,
        "tonal_consistency":        1,
    },
}


chord_table_path = Path(__file__).resolve().parents[1] / 'resources' / 'standard_chord_table.csv'
chord_table = pd.read_csv(chord_table_path)
chord_table['Pitch Classes'] = chord_table['Pitch Classes'].apply(json.loads)

CHORD_TABLE = dict(zip(chord_table['Chord Name'], chord_table['Pitch Classes']))
CHORD_TYPES = dict(zip(chord_table['Chord Name'], chord_table['Chord Type']))
CHORD_SET_LOOKUP = set(frozenset(pc) for pc in CHORD_TABLE.values())


CHORD_CACHE = {}
def process_chord_data(data):
    notes = []

    for right_notes, left_notes in zip(data["right_hand"], data["left_hand"]):
        right_notes = [note % 128 % 12 for note in right_notes]
        left_notes = [note % 128 % 12 for note in left_notes]
        notes.append(right_notes + left_notes)

    chord_names = {name: 0 for name in chord_table['Chord Name']}
    chord_types = {name: 0 for name in chord_table['Chord Type']}
    chord_names["Unknown"] = 0
    chord_types["Unknown"] = 0

    threshold = 0.6

    for time, note in enumerate(notes):
        mod_set = set(note)
        if len(mod_set) < 2:
            continue

        key = frozenset(mod_set)
        if key in CHORD_CACHE:
            predicted_chord_name, max_similarity = CHORD_CACHE[key]
        else:
            max_similarity = 0
            predicted_chord_name = "Unknown"
            for chord_name, pitch_classes in CHORD_TABLE.items():
                pitch_set = set(pitch_classes)
                similarity = len(mod_set & pitch_set) / len(mod_set | pitch_set)
                if similarity > max_similarity:
                    predicted_chord_name = chord_name
                    max_similarity = similarity
                if max_similarity >= 1.0:
                    break
            CHORD_CACHE[key] = (predicted_chord_name, max_similarity)

        if max_similarity >= threshold:
            chord_names[predicted_chord_name] += max_similarity
            chord_types[CHORD_TYPES[predicted_chord_name]] += max_similarity

        else:
            chord_names["Unknown"] += 1
            chord_types["Unknown"] += 1


    return {
        "chord_names": chord_names,
        "chord_types": chord_types,
    }





def js_divergence(p, q):
    keys = set(p) | set(q)
    p = np.array([p.get(k, 0) + 1e-12 for k in keys])
    q = np.array([q.get(k, 0) + 1e-12 for k in keys])
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

def normalized_js(p, q):
    return js_divergence(p, q) / np.log(2)

def l1_distance(d1, d2):
    keys = set(d1.keys()).union(set(d2.keys()))
    return sum(abs(d1.get(k, 0) - d2.get(k, 0)) for k in keys)

def normalize_distribution(counter):
    total = sum(counter.values())
    return {k: v / total for k, v in counter.items()} if total > 0 else {}

def get_pitch_class(pitch):
    """Get the pitch class of a MIDI note (0-11)."""
    return PITCH_CLASSES[pitch % 12]

def get_scale(category, feature_name):
    if category in SCALES:
        if feature_name in SCALES[category]:
            return SCALES[category][feature_name]
    return 1




def analyze_pitch(data):
    """Analyze pitch-related features."""
    from .feature import (
        pitch_histogram, pitch_class_distribution,
        ambitus, interval_histogram, interval_class_histogram
    )
    pitches = data["pitches"]
    durations = data["durations"]

    pitch_hist = {
        "right_hand": pitch_histogram(pitches["right_hand"], durations["right_hand"]),
        "left_hand": pitch_histogram(pitches["left_hand"], durations["left_hand"]),
    }

    pitch_class_dist = {
        "right_hand": pitch_class_distribution(pitches["right_hand"], durations["right_hand"]),
        "left_hand": pitch_class_distribution(pitches["left_hand"], durations["left_hand"]),
    }

    ambitus_values = {
        "right_hand": ambitus(pitches["right_hand"]),
        "left_hand": ambitus(pitches["left_hand"]),
    }

    interval_hist = {
        "right_hand": interval_histogram(pitches["right_hand"], durations["right_hand"]),
        "left_hand": interval_histogram(pitches["left_hand"], durations["left_hand"]),
    }

    interval_class_hist = {
        "right_hand": interval_class_histogram(pitches["right_hand"], durations["right_hand"]),
        "left_hand": interval_class_histogram(pitches["left_hand"], durations["left_hand"]),
    }

    return {
        "pitch_histogram": pitch_hist,
        "pitch_class_distribution": pitch_class_dist,
        "ambitus": ambitus_values,
        "interval_histogram": interval_hist,
        "interval_class_histogram": interval_class_hist,
    }

def analyze_rhythm(data):
    """Analyze rhythm-related features."""
    from .feature import note_duration_histogram, onset_entropy, ornament_density
    onsets = data["onsets"]
    pitches = data["pitches"]
    durations = data["durations"]
    ornaments = data["ornaments"]

    note_duration_hist = {
        "right_hand": note_duration_histogram(durations["right_hand"]),
        "left_hand": note_duration_histogram(durations["left_hand"]),
    }
    onset_entropy_value = {
        "right_hand": onset_entropy(onsets["right_hand"]),
        "left_hand": onset_entropy(onsets["left_hand"]),
    }

    ornament_density_value = {
        "right_hand": ornament_density(ornaments["right_hand"], pitches["right_hand"]),
        "left_hand": ornament_density(ornaments["left_hand"], pitches["left_hand"]),
    }

    return {
        "note_duration_histogram": note_duration_hist,
        "onset_entropy": onset_entropy_value,
        "ornament_density": ornament_density_value,
    }

def analyze_harmony(data):
    """Analyze harmony-related features."""
    from .feature import chord_distribution, chord_type_distribution
    chords = data["chords"]

    return {
        "chord_distribution": chord_distribution(chords["chord_names"]),
        "chord_type_distribution": chord_type_distribution(chords["chord_types"]),
    }

def analyze_syntax(data, harmony):
    """Analyze advanced syntax features."""
    from .feature import melodic_consonance, chord_consonance, tonal_consistency
    input_data = data["cleaned_notes"]
    pitches = data["pitches"]

    melodic_consonance_value = {
        "right_hand": melodic_consonance(pitches["right_hand"]),
        "left_hand": melodic_consonance(pitches["left_hand"]),
    }

    chord_consonance_value = chord_consonance(harmony)

    tonal_consistency_value = {
        "right_hand": tonal_consistency(input_data["right_hand"]),
        "left_hand": tonal_consistency(input_data["left_hand"]),
    }

    return {
        "melodic_consonance": melodic_consonance_value,
        "chord_consonance": chord_consonance_value,
        "tonal_consistency": tonal_consistency_value,
    }
