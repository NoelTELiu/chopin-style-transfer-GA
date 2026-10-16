import json
from copy import deepcopy

from math import gcd

def _choose_tuplet_params_safe(k: int, U: int):
    if k <= 0 or U <= 0:
        raise ValueError(f"Invalid tuplet params: k={k}, U={U}")
    g = gcd(k, U)
    return '16th', k // g, U // g

def _apply_tuplet_safe(n_or_ch, k: int, U: int):
    from music21 import duration

    baseType, actual, normal = _choose_tuplet_params_safe(k, U)

    n_or_ch.duration.type = baseType
    tup = duration.Tuplet(actual, normal)

    n_or_ch.duration.tuplets = [tup]
    return n_or_ch

def read_chromosome_from_file(file_path):
    """Load a chromosome from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)
def write_output(chromosome, file_path='best_solution.json'):
    with open(file_path, 'w') as file:
        json.dump(chromosome, file)

MAJOR_PATTERN = [0, 2, 4, 5, 7, 9, 11]
MINOR_PATTERN = [0, 2, 3, 5, 7, 8, 10]

def get_neighbor(pitch_midi, direction, tonic_midi, mode='major'):
    """Return the scale-aware neighboring pitches around ``pitch_midi``."""
    pitch_pc = (pitch_midi - tonic_midi + 60) % 12
    pattern = MAJOR_PATTERN if mode == 'major' else MINOR_PATTERN
    if pitch_pc in pattern:
        idx = pattern.index(pitch_pc)
        if direction == 'up':
            next_pc = pattern[(idx + 1) % len(pattern)]
            octave_shift = 12 if idx + 1 >= len(pattern) else 0
            return pitch_midi + (next_pc - pitch_pc) + octave_shift
        elif direction == 'down':
            next_pc = pattern[(idx - 1) % len(pattern)]
            octave_shift = -12 if idx - 1 < 0 else 0
            return pitch_midi + (next_pc - pitch_pc) + octave_shift
        else:
            raise ValueError("Direction must be 'up' or 'down'")
    else:

        return pitch_midi + (1 if direction == 'up' else -1)

PITCH_CLASS_BY_NAME = {
    "C": 0, "B#": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "E#": 5, "F": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

def tonic_name_to_midi(tonic_name: str, octave: int = 4) -> int:
    """Convert a tonic spelling such as ``C``, ``F#`` or ``Bb`` to MIDI."""
    if tonic_name not in PITCH_CLASS_BY_NAME:
        raise ValueError(f"Unsupported tonic name: {tonic_name!r}")

    return 12 * (octave + 1) + PITCH_CLASS_BY_NAME[tonic_name]

ORNAMENT_OFFSETS = {
    'acciaccatura': 128,
    'turn': 256,
    'inverted_turn': 384,
    'trill': 512,
    'upper_mordent': 640,
    'lower_mordent': 768,
}

def get_ornament_base_pitch(p):
    """Decode an ornamented pitch into its base pitch and ornament type."""
    for name, base in ORNAMENT_OFFSETS.items():
        if base <= p < base + 128:
            return p - base, name
    return p, None

def apply_ornament(pitch, ornament_type, tonic_midi, mode='major', duration=0.25):
    """Expand an ornament type into a sequence of music21 note objects."""
    from music21 import note
    if ornament_type == 'acciaccatura':
        n = note.Note(pitch, quarterLength= duration / 8)
        n.stemDirection = 'up'
        n.tie = None
        return [n]
    elif ornament_type == 'turn':
        up = get_neighbor(pitch, 'up', tonic_midi, mode)
        down = get_neighbor(pitch, 'down', tonic_midi, mode)
        return [note.Note(up, quarterLength=duration / 8),
                note.Note(pitch, quarterLength=duration / 8),
                note.Note(down, quarterLength=duration / 8),
                note.Note(pitch, quarterLength=duration *5/8)]
    elif ornament_type == 'inverted_turn':
        up = get_neighbor(pitch, 'up', tonic_midi, mode)
        down = get_neighbor(pitch, 'down', tonic_midi, mode)
        return [note.Note(down, quarterLength=duration / 8),
                note.Note(pitch, quarterLength=duration / 8),
                note.Note(up, quarterLength=duration / 8),
                note.Note(pitch, quarterLength=duration *5/8)]
    elif ornament_type == 'trill':
        up = get_neighbor(pitch, 'up', tonic_midi, mode)
        unit = 0.0625
        num = max(1, int(duration / unit / 2))
        notes = []
        for _ in range(num):
            notes.append(note.Note(pitch, quarterLength=unit))
            notes.append(note.Note(up, quarterLength=unit))
        return notes
    elif ornament_type == 'upper_mordent':
        up = get_neighbor(pitch, 'up', tonic_midi, mode)
        return [note.Note(pitch, quarterLength=duration / 8),
                note.Note(up, quarterLength=duration / 8),
                note.Note(pitch, quarterLength=duration *3/4)]
    elif ornament_type == 'lower_mordent':
        down = get_neighbor(pitch, 'down', tonic_midi, mode)
        return [note.Note(pitch, quarterLength=duration / 8),
                note.Note(down, quarterLength=duration / 8),
                note.Note(pitch, quarterLength=duration *3/4)]
    else:
        return [note.Note(pitch, quarterLength=duration)]

def two_hand_chromosome_to_midi(right, left, filename='output.mid', bpm=120, tonic_midi=60, mode='major'):
    from music21 import stream, midi, tempo, note, chord

    score = stream.Score()
    score.append(tempo.MetronomeMark(number=bpm))
    right_part = stream.Part()
    left_part = stream.Part()

    for part, chromo in zip([right_part, left_part], [right, left]):
        idx = 0
        pending_grace = []
        while idx < len(chromo):
            current = chromo[idx]
            if current == []:
                duration_counter = 1
                idx += 1
                while idx < len(chromo) and chromo[idx] in ([], [-1]):
                    duration_counter += 1
                    idx += 1

                total_16ths = duration_counter
                while total_16ths > 0:
                    rest_len_16ths = min(total_16ths, 4)
                    part.append(note.Rest(quarterLength=rest_len_16ths / 4))
                    total_16ths -= rest_len_16ths
            elif current != [-1] and current != [-3]:
                is_tuplet_start = (
                    isinstance(current, list)
                    and len(current) > 1
                    and (idx + 1) < len(chromo)
                    and chromo[idx + 1] == [-3]
                )

                if is_tuplet_start:

                    j = idx + 1
                    while j < len(chromo) and chromo[j] == [-3]:
                        j += 1
                    U = j - idx
                    k = len(current)

                    if pending_grace:
                        U -= 1
                        grace_len = (1 / 8) / len(pending_grace)
                        for g in pending_grace:
                            if len(g) > 1:
                                part.append(chord.Chord(g, quarterLength=grace_len))
                            else:
                                part.append(note.Note(g[0], quarterLength=grace_len))
                        pending_grace = []


                    for p in current:
                        base, ornament_type = get_ornament_base_pitch(p)
                        base += tonic_midi - 60
                        base = max(0, min(127, base))

                        n = note.Note(base)
                        _apply_tuplet_safe(n, k, U)
                        part.append(n)

                    idx = j
                else:
                    duration_counter = 1
                    idx += 1
                    while idx < len(chromo) and chromo[idx] == [-1]:
                        duration_counter += 1
                        idx += 1

                    base_pitches = []
                    ornaments = []
                    for p in current:
                        base, ornament = get_ornament_base_pitch(p)
                        base += tonic_midi - 60
                        base = max(0, min(127, base))
                        if ornament:
                            ornaments.append((base, ornament))
                        else:
                            base_pitches.append(base)

                    quarter_length = duration_counter / 4

                    if ornaments:
                        if ornaments[0][1] == 'acciaccatura':
                            pending_grace.append([o[0] for o in ornaments if o[1] == 'acciaccatura'])
                        for base, ornament in ornaments:
                            if ornament == 'acciaccatura':
                                continue
                            for n in apply_ornament(base, ornament, tonic_midi, mode, quarter_length):
                                part.append(deepcopy(n))
                    else:
                        if pending_grace:
                            grace_len = quarter_length / 32
                            for g in pending_grace:
                                if len(g) > 1:
                                    part.append(chord.Chord(g, quarterLength=grace_len))
                                else:
                                    part.append(note.Note(g[0], quarterLength=grace_len))
                            quarter_length -= grace_len * len(pending_grace)
                            pending_grace = []

                        if len(base_pitches) > 1:
                            part.append(chord.Chord(base_pitches, quarterLength=quarter_length))
                        elif len(base_pitches) == 1:
                            part.append(note.Note(base_pitches[0], quarterLength=quarter_length))
                        else:
                            pass

            else:
                idx += 1

    score.append(right_part)
    score.append(left_part)
    mf = midi.translate.streamToMidiFile(score)
    mf.open(filename, 'wb')
    mf.write()
    mf.close()

def count_notes_in_chromosome(chromosome):
    """Count sounding note events in the chromosome representation."""
    count = 0
    for i in range(len(chromosome)):
        if chromosome[i] not in ([], [-1], [-3]):
            count += 1

    return count
