import numpy as np


KS_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                 'F#', 'G', 'G#', 'A', 'A#', 'B']

def build_histogram(pitches):
    """Build normalized pitch class histogram from list of MIDI notes."""
    if len(pitches) == 0:
        return np.zeros(12, dtype=np.float32)

    arr = np.asarray(pitches, dtype=np.int32)


    arr = arr[(arr >= 0) & (arr < 128)]
    if arr.size == 0:
        return np.zeros(12, dtype=np.float32)

    pcs = arr % 12
    counts = np.bincount(pcs, minlength=12).astype(np.float32)

    total = counts.sum()
    return counts / total if total > 0 else counts


_MAJOR_ROTS = np.vstack([np.roll(KS_MAJOR_PROFILE, i) for i in range(12)])
_MINOR_ROTS = np.vstack([np.roll(KS_MINOR_PROFILE, i) for i in range(12)])
_PROFILES = np.vstack([_MAJOR_ROTS, _MINOR_ROTS])

_MEAN = _PROFILES.mean(axis=1, keepdims=True)
_STD = _PROFILES.std(axis=1, keepdims=True).clip(min=1e-8)
_ZPROFILES = (_PROFILES - _MEAN) / _STD

def detect_key_ks(pitches):
    """Detect key using precomputed KS profiles with robust normalization."""
    if len(pitches) < 3:
        return None

    histogram = build_histogram(pitches)
    if histogram.sum() == 0:
        return None

    h_mean = histogram.mean().astype(np.float32)
    h_std = histogram.std()
    if h_std < 1e-6:
        scores = _PROFILES @ histogram
    else:
        z_hist = (histogram - h_mean) / h_std
        scores = _ZPROFILES @ z_hist

    best_idx = int(np.argmax(scores))
    tonic = PITCH_CLASSES[best_idx % 12]
    mode = 'major' if best_idx < 12 else 'minor'
    best_score = (scores[best_idx] - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return tonic, mode, best_score
