


from collections import Counter
from scipy.stats import entropy
from .utility import CHORD_VALUES, get_pitch_class, normalize_distribution
from .fast_key_detect import detect_key_ks, PITCH_CLASSES




def pitch_histogram(notes, durations):
    """Calculate the distribution of pitches weighted by durations."""
    weighted_counter = Counter()
    for note, duration in zip(notes, durations):
        weighted_counter[note] += duration
    return normalize_distribution(weighted_counter)

def pitch_class_distribution(notes, durations):
    """Calculate the distribution of pitch classes weighted by durations."""
    pitch_classes = [get_pitch_class(note) for note in notes]
    weighted_counter = Counter()
    for pitch_class, duration in zip(pitch_classes, durations):
        weighted_counter[pitch_class] += duration
    return normalize_distribution(weighted_counter)

def ambitus(notes):
    """Calculate the range between the highest and lowest note."""
    return max(notes) - min(notes) if notes else 0

def interval_histogram(notes, durations):
    """Absolute pitch-interval distribution over MIDI distances 0–127."""
    weighted_counter = Counter()
    for i in range(min(len(notes), len(durations)) - 1):
        interval = abs(notes[i] - notes[i + 1])
        weighted_counter[interval] += min(durations[i], durations[i + 1])
    return normalize_distribution(weighted_counter)

def interval_class_histogram(notes, durations):
    """Unsigned interval-class distribution over 0–11 semitones."""
    weighted_counter = Counter()
    for i in range(min(len(notes), len(durations)) - 1):
        interval = abs(notes[i] - notes[i + 1]) % 12
        weighted_counter[interval] += min(durations[i], durations[i + 1])
    return normalize_distribution(weighted_counter)




def note_duration_histogram(durations):
    """Calculate the histogram of note durations."""
    return normalize_distribution(Counter(durations))

def onset_entropy(onsets):
    """Calculate the entropy of note onsets."""
    if len(onsets) == 0:
        return 0
    onset_counts = Counter(onsets)
    probabilities = [count / sum(onset_counts.values()) for count in onset_counts.values()]
    return entropy(probabilities, base=2)

def ornament_density(ornament_notes, total_notes):
    """Calculate the ornament density."""
    return len(ornament_notes) / len(total_notes) if total_notes else 0





def chord_distribution(chords):
    """Calculate the distribution of chord names"""
    return normalize_distribution(Counter(chords))

def chord_type_distribution(chords):
    """Calculate the distribution of chord types"""
    return normalize_distribution(Counter(chords))




def melodic_consonance(notes):
    """Evaluate interval-class smoothness on a folded 0–6 interval-class scale."""
    if len(notes) < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(len(notes) - 1):
        semitones = abs(notes[i] - notes[i + 1]) % 12
        interval_class = min(semitones, 12 - semitones)
        if interval_class <= 2:
            score = 1.0
        elif interval_class <= 4:
            score = 0.6
        else:
            score = 0.2
        total += score
        count += 1
    return total / count if count else 0.0

def chord_consonance(data):
    """Calculate the consonance of chords based on their types."""
    chord_type_distribution = data["chord_type_distribution"]

    score = 0
    for type in chord_type_distribution:
        if type not in CHORD_VALUES:
            continue
        value = CHORD_VALUES[type]
        score += chord_type_distribution[type] * value

    return score

def tonal_consistency(timed_sequence, window_size=32, step_size=16):
    """
    Compute average % of notes in key in sliding windows of a normalized time-step sequence.
    Input: list of frames (e.g., [[60], [60, 64], [], ...]), already cleaned via normalize_representation().
    Assumes all notes are base MIDI pitches (0–127), no [-1], [-3], or ornament offsets.
    """
    if not timed_sequence:
        return 0

    scores = []

    for start in range(0, len(timed_sequence) - window_size + 1, step_size):
        window = timed_sequence[start:start + window_size]


        midi_notes = [n for frame in window for n in frame]

        if not midi_notes:
            continue


        result = detect_key_ks(midi_notes)
        if result is None:
            continue

        tonic, mode, _ = result


        scale_template = [0, 2, 4, 5, 7, 9, 11] if mode == 'major' else [0, 2, 3, 5, 7, 8, 10]
        tonic_index = PITCH_CLASSES.index(tonic)
        scale_pcs = [(tonic_index + pc) % 12 for pc in scale_template]


        pcs = [p % 12 for p in midi_notes]
        in_key = sum(1 for pc in pcs if pc in scale_pcs)
        scores.append(in_key / len(pcs))

    return sum(scores) / len(scores) if scores else 0
