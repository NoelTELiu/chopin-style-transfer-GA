import xml.etree.ElementTree as ET
import zipfile
import os
import json
import shutil

ORNAMENT_OFFSETS = {
    'acciaccatura': 128,
    'turn': 256,
    'inverted_turn': 384,
    'trill': 512,
    'upper_mordent': 640,
    'lower_mordent': 768,
}

def extract_mxl(mxl_path, output_dir="temp_xml"):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(mxl_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    for filename in os.listdir(output_dir):
        if filename.endswith(".xml"):
            return os.path.join(output_dir, filename)

    raise FileNotFoundError("No XML file found in extracted MXL.")

def pitch_to_midi_from_xml(pitch_element):
    step = pitch_element.findtext("step")
    alter = pitch_element.findtext("alter")
    octave = int(pitch_element.findtext("octave"))
    semitone_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    midi = 12 + (octave * 12) + semitone_map[step]
    if alter:
        midi += int(alter)
    return midi

def duration_to_units(duration_val, divisions):
    return int(round(float(duration_val) * 4 / divisions))

def is_acciaccatura(note_el):
    grace = note_el.find('grace')
    if grace is not None and \
        (grace.attrib.get('slash') == 'yes' or 'slash' not in grace.attrib):
        return True
    return False

def detect_ornament(note_el):
    notations = note_el.find('notations')
    if notations is None:
        return None
    ornaments = notations.find('ornaments')
    if ornaments is None:
        return None
    if ornaments.find('turn') is not None:
        return 'turn'
    if ornaments.find('inverted-turn') is not None:
        return 'inverted_turn'
    if ornaments.find('trill-mark') is not None:
        return 'trill'
    if ornaments.find('mordent') is not None:
        return 'upper_mordent'
    if ornaments.find('inverted-mordent') is not None:
        return 'lower_mordent'
    return None

def encode_pitch(pitch, ornament=None):
    if ornament is None:
        return pitch
    offset = ORNAMENT_OFFSETS.get(ornament)
    return offset + pitch if offset is not None else pitch

def is_tie_start(note_el):
    for tie in note_el.findall('tie'):
        if tie.attrib.get('type') == 'start':
            return True
    return False

def is_tie_stop(note_el):
    for tie in note_el.findall('tie'):
        if tie.attrib.get('type') == 'stop':
            return True
    return False

def parse_musicxml_to_json(xml_path, output_path):
    from music21 import converter, note
    if xml_path.endswith(".mxl"):
        xml_path = extract_mxl(xml_path)


    score = converter.parse(xml_path)
    key_obj = score.analyze('key')
    tonic_class = key_obj.tonic.name
    forced_tonic = note.Note(tonic_class + '4')
    tonic_midi = forced_tonic.pitch.midi

    if xml_path.endswith(".mxl"):
        xml_path = extract_mxl(xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    bpm = 120
    for sound in root.findall('.//sound'):
        if 'tempo' in sound.attrib:
            bpm = float(sound.attrib['tempo'])
            break

    right_hand = []
    left_hand = []
    measure_boundaries = []

    current_divisions = 4

    ongoing_tie = {
        1: None,
        2: None
    }

    for part in root.findall('part'):
        for measure in part.findall('measure'):
            attributes = measure.find('attributes')
            if attributes is not None:
                div = attributes.find('divisions')
                if div is not None:
                    current_divisions = int(div.text)

            notes_in_measure = measure.findall('note')

            i = 0
            while i < len(notes_in_measure):
                note_el = notes_in_measure[i]
                is_rest = note_el.find('rest') is not None
                duration_el = note_el.find('duration')
                if duration_el is None:
                    if is_acciaccatura(note_el):
                        duration_val = 0
                    else:
                        i += 1
                        continue
                else:
                    duration_val = duration_el.text
                units = duration_to_units(duration_val, current_divisions)

                staff_el = note_el.find('staff')
                staff = int(staff_el.text) if staff_el is not None else 1

                time_mod = note_el.find('time-modification')
                if time_mod is not None:
                    actual_notes = int(time_mod.findtext('actual-notes', '3'))

                    tuplet_group = []
                    base_i = i
                    tuplet_units = 0

                    while i < base_i + actual_notes and i < len(notes_in_measure):
                        tuplet_el = notes_in_measure[i]
                        if tuplet_el.find('pitch') is not None:
                            pitch_el = tuplet_el.find('pitch')
                            midi = pitch_to_midi_from_xml(pitch_el)
                            midi = midi - tonic_midi + 60
                            ornament = detect_ornament(tuplet_el)
                            if is_acciaccatura(tuplet_el):
                                ornament = 'acciaccatura'
                            encoded_pitch = encode_pitch(midi, ornament)
                            tuplet_group.append(encoded_pitch)

                        dur_el = tuplet_el.find('duration')
                        if dur_el is not None:
                            tuplet_units += float(dur_el.text)

                        i += 1

                    real_units = int(round(tuplet_units * 4 / current_divisions))
                    out = [tuplet_group]
                    out_span = [[-3]] * (real_units - 1)

                    if staff == 1:
                        right_hand.extend(out + out_span)
                    else:
                        left_hand.extend(out + out_span)
                    continue

                if is_rest:
                    out = []
                elif note_el.find('pitch') is not None:
                    pitch_el = note_el.find('pitch')
                    midi = pitch_to_midi_from_xml(pitch_el)
                    midi = midi - tonic_midi + 60
                    ornament = detect_ornament(note_el)
                    if is_acciaccatura(note_el):
                        ornament = 'acciaccatura'
                    encoded_pitch = encode_pitch(midi, ornament)


                    chord_notes = [encoded_pitch]
                    j = i + 1
                    while j < len(notes_in_measure) and notes_in_measure[j].find('chord') is not None:
                        next_pitch_el = notes_in_measure[j].find('pitch')
                        if next_pitch_el is not None:
                            next_midi = pitch_to_midi_from_xml(next_pitch_el)
                            next_midi = next_midi - tonic_midi + 60
                            next_ornament = detect_ornament(notes_in_measure[j])
                            if is_acciaccatura(notes_in_measure[j]):
                                next_ornament = 'acciaccatura'
                            encoded_next_pitch = encode_pitch(next_midi, next_ornament)
                            chord_notes.append(encoded_next_pitch)
                        j += 1
                    out = chord_notes
                    i = j - 1


                    previous = ongoing_tie[staff]
                    if previous is not None and out == previous:
                        if staff == 1:
                            right_hand.append([-1])
                            right_hand.extend([[-1]] * (units - 1))
                        else:
                            left_hand.append([-1])
                            left_hand.extend([[-1]] * (units - 1))
                        if is_tie_stop(note_el):
                            ongoing_tie[staff] = None
                        i += 1
                        continue
                    else:
                        if is_tie_start(note_el):
                            ongoing_tie[staff] = out
                        else:
                            ongoing_tie[staff] = None
                else:
                    out = []

                if staff == 1:
                    right_hand.append(out)
                    right_hand.extend([[-1]] * (units - 1))
                elif staff == 2:
                    left_hand.append(out)
                    left_hand.extend([[-1]] * (units - 1))
                i += 1



            boundary = max(len(right_hand), len(left_hand))
            if boundary > 0 and (not measure_boundaries or boundary > measure_boundaries[-1]):
                measure_boundaries.append(boundary)

    max_len = max(len(right_hand), len(left_hand))
    right_hand += [[]] * (max_len - len(right_hand))
    left_hand += [[]] * (max_len - len(left_hand))

    output = {
        "right_hand": right_hand,
        "left_hand": left_hand,
        "key": {
            "tonic": key_obj.tonic.name,
            "mode": key_obj.mode
        },
        "bpm": bpm,
        "measure_boundaries": measure_boundaries,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

