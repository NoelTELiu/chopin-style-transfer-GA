from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.stats import entropy

from .feature import (
    pitch_histogram, pitch_class_distribution, ambitus, interval_class_histogram,
    note_duration_histogram, onset_entropy,
    chord_distribution, chord_type_distribution,
)
from .utility import process_chord_data, normalized_js


def _measure_windows(n_frames, measure_len=16):
    """Return sixteenth-note windows; keep a final partial measure instead of dropping it."""
    if n_frames <= 0:
        return [], 0
    windows = [(a, min(a + measure_len, n_frames)) for a in range(0, n_frames, measure_len)]
    return windows, len(windows)



def _windows_for_processed(processed, n_frames, measure_len=16):
    boundaries = [int(x) for x in processed.get("measure_boundaries", []) if 0 < int(x) <= n_frames]
    if boundaries:
        windows = []
        start = 0
        for end in boundaries:
            if end > start:
                windows.append((start, end))
                start = end
        if start < n_frames:
            windows.append((start, n_frames))
        return windows, len(windows)
    return _measure_windows(n_frames, measure_len)


def _cosine_ssm(vectors):
    if not len(vectors):
        return np.zeros((0, 0), dtype=float)
    P = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Pn = P / norms
    return Pn @ Pn.T


def _gaussian_checkerboard_kernel(k, sigma=None):
    L = 2 * k + 1
    sigma = sigma if sigma is not None else max(k / 2.0, 1e-6)
    K = np.zeros((L, L))
    K[:k, :k] = 1
    K[:k, k + 1:] = -1
    K[k + 1:, :k] = -1
    K[k + 1:, k + 1:] = 1
    x = np.arange(-k, k + 1)
    X, Y = np.meshgrid(x, x)
    G = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return K * G


def _novelty_from_ssm(S, kernel_radius):
    L = S.shape[0]
    if L == 0:
        return np.zeros(0)
    if 2 * kernel_radius + 1 > L:

        return np.zeros(L)
    K = _gaussian_checkerboard_kernel(kernel_radius)
    nov = np.zeros(L)
    for t in range(kernel_radius, L - kernel_radius):
        block = S[t-kernel_radius:t+kernel_radius+1, t-kernel_radius:t+kernel_radius+1]
        nov[t] = np.sum(block * K)
    if nov.max() > nov.min():
        nov = (nov - nov.min()) / (nov.max() - nov.min())
    return nov


def extract_per_measure_features(processed, measure_len=16):
    """Extract per-measure descriptors used to build the structural SSM."""
    rh = processed["cleaned_notes"]["right_hand"]
    lh = processed["cleaned_notes"]["left_hand"]
    L = max(len(rh), len(lh))
    rh = list(rh) + [[]] * (L - len(rh))
    lh = list(lh) + [[]] * (L - len(lh))

    ms_bounds, _ = _windows_for_processed(processed, L, measure_len)
    per_measure_features = []
    for a, b in ms_bounds:
        notes_r = [n % 128 for frame in rh[a:b] for n in frame]
        notes_l = [n % 128 for frame in lh[a:b] for n in frame]
        notes = notes_r + notes_l
        durations = [1] * len(notes)
        feat = {
            "pitch_histogram": pitch_histogram(notes, durations),
            "pitch_class_distribution": pitch_class_distribution(notes, durations),
            "ambitus": ambitus(notes),
            "interval_class_histogram": interval_class_histogram(notes, durations),
        }
        chords = process_chord_data({"right_hand": rh[a:b], "left_hand": lh[a:b]})
        feat["chord_distribution"] = chord_distribution(chords["chord_names"])
        feat["chord_type_distribution"] = chord_type_distribution(chords["chord_types"])
        feat["note_duration_histogram"] = note_duration_histogram(durations)
        onset_positions = [i for i, frame in enumerate(rh[a:b]) if frame] + [i for i, frame in enumerate(lh[a:b]) if frame]
        feat["onset_entropy"] = onset_entropy(onset_positions)
        per_measure_features.append(feat)
    return per_measure_features



def extract_measure_pitch_class_vectors(processed, measure_len=16):
    """Return normalized 12-D pitch-class vectors for the SSM."""
    rh = processed["cleaned_notes"]["right_hand"]
    lh = processed["cleaned_notes"]["left_hand"]
    L = max(len(rh), len(lh))
    rh = list(rh) + [[]] * (L - len(rh))
    lh = list(lh) + [[]] * (L - len(lh))
    windows, _ = _windows_for_processed(processed, L, measure_len)
    vectors = []
    for a, b in windows:
        pc = np.zeros(12, dtype=float)
        for t in range(a, b):
            for pitch in rh[t] + lh[t]:
                pc[(pitch % 128) % 12] += 1.0
        if pc.sum() > 0:
            pc /= pc.sum()
        vectors.append(pc)
    return np.asarray(vectors, dtype=float)

def flux_novelty_from_existing_features(per_measure_features, feature_config=None, normalize=True):
    """Alternative feature-flux novelty mode for diagnostic comparison."""
    if feature_config is None:
        feature_config = {
            "pitch_histogram": ("js", 1.0),
            "pitch_class_distribution": ("js", 1.0),
            "interval_class_histogram": ("js", 1.0),
            "chord_distribution": ("js", 1.0),
            "chord_type_distribution": ("js", 1.0),
            "ambitus": ("euclidean", 0.5),
            "onset_entropy": ("euclidean", 0.5),
        }
    L = len(per_measure_features)
    nov = np.zeros(L)
    for t in range(1, L):
        total, weight_sum = 0.0, 0.0
        for fname, (metric, weight) in feature_config.items():
            v1 = per_measure_features[t - 1].get(fname)
            v2 = per_measure_features[t].get(fname)
            if v1 is None or v2 is None:
                continue
            if metric == "js" or isinstance(v1, dict):
                dist = normalized_js(v1, v2)
            else:
                dist = abs(v1 - v2) / max(abs(v1), abs(v2), 1e-6)
            total += weight * dist
            weight_sum += weight
        nov[t] = total / weight_sum if weight_sum else 0.0
    if normalize and nov.max() > nov.min():
        nov = (nov - nov.min()) / (nov.max() - nov.min())
    return nov


def section_boundaries_global(processed, measure_len=16, min_gap=2, prominence=0.2, novelty_mode="ssm"):
    """Detect section boundaries using SSM or feature-flux novelty."""
    if novelty_mode == "ssm":
        vecs = extract_measure_pitch_class_vectors(processed, measure_len)
        if len(vecs) == 0:
            return [(0, 0)], np.zeros((0, 0)), np.array([])
        S = _cosine_ssm(vecs)
        if len(vecs) < 5:
            k = 1
        else:
            k = max(2, min(len(vecs) // 16, (len(vecs) - 1) // 2))
        nov = _novelty_from_ssm(S, kernel_radius=k)
        M = len(vecs)
    elif novelty_mode == "flux":
        per_measure_features = extract_per_measure_features(processed, measure_len)
        if not per_measure_features:
            return [(0, 0)], np.zeros((0, 0)), np.array([])
        nov = flux_novelty_from_existing_features(per_measure_features)
        S = np.zeros((0, 0))
        M = len(per_measure_features)
    else:
        raise ValueError(f"Unsupported novelty_mode: {novelty_mode}")

    peaks, _ = find_peaks(nov, prominence=prominence, distance=min_gap)
    cuts = [int(p) for p in peaks if 0 < int(p) < M]
    sections, prev = [], 0
    for cut in cuts:
        if cut > prev:
            sections.append((prev, cut))
            prev = cut
    if prev < M:
        sections.append((prev, M))
    if not sections:
        sections = [(0, M)]
    return sections, S, nov


def label_sections_by_content(processed, sections, measure_len=16):
    """Assign A/B/C... labels, reusing a label when section pitch-class content recurs."""
    rh = processed["cleaned_notes"]["right_hand"]
    lh = processed["cleaned_notes"]["left_hand"]
    L = max(len(rh), len(lh))
    rh = list(rh) + [[]] * (L - len(rh))
    lh = list(lh) + [[]] * (L - len(lh))
    ms_bounds, _ = _windows_for_processed(processed, L, measure_len)

    def section_pcp(s, e):
        if not ms_bounds or e <= s:
            return np.zeros(12)
        a = ms_bounds[s][0]
        b = ms_bounds[min(e - 1, len(ms_bounds) - 1)][1]
        pc = np.zeros(12)
        for t in range(a, b):
            for n in rh[t] + lh[t]:
                pc[(n % 128) % 12] += 1
        return pc / pc.sum() if pc.sum() > 0 else pc

    centroids = []
    centroid_labels = []
    labels = []
    next_label_ord = ord("A")
    for s, e in sections:
        v = section_pcp(s, e)
        best, idx = -1.0, -1
        for j, u in enumerate(centroids):
            den = (np.linalg.norm(v) * np.linalg.norm(u)) or 1.0
            cos = float(np.dot(v, u)) / den
            if cos > best:
                best, idx = cos, j
        if best >= 0.85 and idx >= 0:
            labels.append(centroid_labels[idx])
        else:
            lbl = chr(next_label_ord)
            labels.append(lbl)
            centroids.append(v)
            centroid_labels.append(lbl)
            next_label_ord += 1
    return labels


def analyze_structure(processed, novelty_mode="ssm"):
    """SSM -> novelty curve -> section boundaries -> recurrence labels."""
    measure_len = 16
    sections, S, nov = section_boundaries_global(
        processed,
        measure_len=measure_len,
        min_gap=2,
        prominence=0.2,
        novelty_mode=novelty_mode,
    )
    labels = label_sections_by_content(processed, sections, measure_len=measure_len)
    return {
        "sections": [list(x) for x in sections],
        "section_labels": labels,
        "S": S.tolist() if hasattr(S, "tolist") else S,
        "nov": nov.tolist() if hasattr(nov, "tolist") else nov,
    }


def _normalized_intervals(sections):
    if not sections:
        return []
    total = max(float(e) for _, e in sections)
    if total <= 0:
        return []
    return [(float(s) / total, float(e) / total) for s, e in sections if e > s]


def _interval_iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def section_alignment_similarity(sections_a, sections_b):
    """Compare where sections occur using optimal IoU matching on normalized timelines."""
    A = _normalized_intervals(sections_a)
    B = _normalized_intervals(sections_b)
    if not A or not B:
        return 0.0
    matrix = np.array([[_interval_iou(a, b) for b in B] for a in A], dtype=float)
    rows, cols = linear_sum_assignment(-matrix)
    matched = float(matrix[rows, cols].sum()) if len(rows) else 0.0
    return matched / max(len(A), len(B))


def _js_similarity_from_multisets(counter_a, counter_b):
    keys = set(counter_a) | set(counter_b)
    if not keys:
        return 0.0
    p = np.array([counter_a.get(k, 0) + 1e-12 for k in keys], dtype=float)
    q = np.array([counter_b.get(k, 0) + 1e-12 for k in keys], dtype=float)
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
    return 1.0 - min(1.0, float(js / np.log(2)))


def recurrence_similarity(labels_a, labels_b):
    """Compare how often section labels recur while ignoring their ordering."""
    mult_a = {lbl: labels_a.count(lbl) for lbl in set(labels_a)}
    mult_b = {lbl: labels_b.count(lbl) for lbl in set(labels_b)}
    return _js_similarity_from_multisets(mult_a, mult_b) if (mult_a or mult_b) else 0.0


def structure_similarity(struct_a, struct_b, alignment_weight=0.7):
    """Combine section alignment and recurrence similarity into a score in [0, 1]."""
    alignment = section_alignment_similarity(struct_a.get("sections", []), struct_b.get("sections", []))
    recurrence = recurrence_similarity(
        struct_a.get("section_labels", []),
        struct_b.get("section_labels", []),
    )
    return alignment_weight * alignment + (1.0 - alignment_weight) * recurrence
