import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

PITCH_CLASSES = {
    'C':        0,
    'C#/Db':    1,
    'D':        2,
    'D#/Eb':    3,
    'E':        4,
    'F':        5,
    'F#/Gb':    6,
    'G':        7,
    'G#/Ab':    8,
    'A':        9,
    'A#/Bb':    10,
    'B':        11
}

def merge_symmetric_intervals(histogram):
    """Merge complementary interval classes ``i`` and ``12 - i`` into one histogram bin."""
    merged = {}
    for i in range(12):
        j = (12 - i) % 12
        if i > j:
            break
        merged[i] = histogram.get(i, 0) + histogram.get(j, 0) if i != j else histogram.get(i, 0)
    return merged

def plot_histogram(data, title, xlabel, ylabel, output_path):
    """Plot a histogram and a smoothed line plot."""
    plt.figure(figsize=(10, 6))
    keys = list(data.keys())
    x = np.arange(len(keys))
    x_labels = list(range(min(keys), max(keys) + 1))
    values = list(data.values())


    plt.bar(x, values, color='skyblue', alpha=0.5, label='Histogram')


    if len(values) > 2:
        smooth_values = gaussian_filter1d(values, sigma=1, truncate=1)
        plt.plot(x, smooth_values, color='blue', linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks=x, labels=x_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_histogram_comparison(data1, data2, data3, title, xlabel, ylabel, output_path):
    """Plot a comparison histogram for three datasets with smoothed line plots."""
    plt.figure(figsize=(10, 6))
    keys = list(data1.keys())
    x = np.arange(len(keys))
    x_labels = list(range(min(keys), max(keys) + 1))
    width = 0.25

    values1 = list(data1.values())
    values2 = list(data2.values())
    values3 = list(data3.values())

    plt.bar(x - width, values1, width=width, label='Input', color='skyblue', alpha=0.3)
    plt.bar(x, values2, width=width, label='Output', color='green', alpha=0.3)
    plt.bar(x + width, values3, width=width, label='Target', color='salmon', alpha=0.3)


    if len(values1) > 2:
        plt.plot(x, values1, color='blue', linewidth=1, linestyle=':')
        plt.plot(x, values2, color='darkgreen', linewidth=1)
        plt.plot(x, values3, color='red', linewidth=1, linestyle='--')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(ticks=x, labels=x_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def get_full_histogram(histogram, min_value, max_value):
    """Get a full histogram with all keys from min_value to max_value."""
    full_histogram = {i: histogram.get(i, 0) for i in range(min_value, max_value + 1)}
    return full_histogram

def plot_pitch_analysis(output_dir, input_data, output_data=None, target_data=None):
    """Plot pitch-related analysis results."""
    if target_data:
        for hand, histogram in input_data["pitch_histogram"].items():
            min_key = min(min(histogram.keys(), default=0),
                          min(output_data["pitch_histogram"][hand].keys(), default=0),
                          min(target_data["pitch_histogram"][hand].keys(), default=0))
            max_key = max(max(histogram.keys(), default=127),
                          max(output_data["pitch_histogram"][hand].keys(), default=127),
                          max(target_data["pitch_histogram"][hand].keys(), default=127))
            full_histogram = get_full_histogram(histogram, min_key, max_key)
            output_histogram = get_full_histogram(
                output_data["pitch_histogram"][hand], min_key, max_key
            )
            target_histogram = get_full_histogram(
                target_data["pitch_histogram"][hand], min_key, max_key
            )

            plot_histogram_comparison(
                full_histogram,
                output_histogram,
                target_histogram,
                title=f"Pitch Histogram Comparison ({hand})",
                xlabel="Pitch (MIDI 0-127)",
                ylabel="Frequency",
                output_path=f"{output_dir}/pitch_histogram_comparison_{hand}.png"
            )

        for hand, distribution in input_data["pitch_class_distribution"].items():
            full_distribution = get_full_histogram(
                {PITCH_CLASSES[key]: value for key, value in distribution.items()}, 0, 11
            )
            output_distribution = get_full_histogram(
                {PITCH_CLASSES[key]: value \
                    for key, value in output_data["pitch_class_distribution"][hand].items()}, 0, 11
            )
            target_distribution = get_full_histogram(
                {PITCH_CLASSES[key]: value \
                    for key, value in target_data["pitch_class_distribution"][hand].items()}, 0, 11
            )
            plot_histogram_comparison(
                full_distribution,
                output_distribution,
                target_distribution,
                title=f"Pitch Class Distribution Comparison ({hand})",
                xlabel="Pitch Class (0-11)",
                ylabel="Frequency",
                output_path=f"{output_dir}/pitch_class_distribution_comparison_{hand}.png"
            )

        for hand, histogram in input_data["interval_histogram"].items():
            min_key = min(min(histogram.keys(), default=-127),
                          min(output_data["interval_histogram"][hand].keys(), default=-127),
                          min(target_data["interval_histogram"][hand].keys(), default=-127))
            max_key = max(max(histogram.keys(), default=127),
                          max(output_data["interval_histogram"][hand].keys(), default=127),
                          max(target_data["interval_histogram"][hand].keys(), default=127))
            full_histogram = get_full_histogram(histogram, min_key, max_key)
            output_histogram = get_full_histogram(
                output_data["interval_histogram"][hand], min_key, max_key
            )
            target_histogram = get_full_histogram(
                target_data["interval_histogram"][hand], min_key, max_key
            )

            plot_histogram_comparison(
                full_histogram,
                output_histogram,
                target_histogram,
                title=f"Interval Histogram Comparison ({hand})",
                xlabel="Interval (-127~127)",
                ylabel="Frequency",
                output_path=f"{output_dir}/interval_histogram_comparison_{hand}.png"
            )

        for hand, histogram in input_data["interval_class_histogram"].items():
            full_histogram = get_full_histogram(histogram, -11, 11)
            output_histogram = get_full_histogram(output_data["interval_class_histogram"][hand], -11, 11)
            target_histogram = get_full_histogram(target_data["interval_class_histogram"][hand], -11, 11)
            full_histogram = merge_symmetric_intervals(full_histogram)
            output_histogram = merge_symmetric_intervals(output_histogram)
            target_histogram = merge_symmetric_intervals(target_histogram)

            plot_histogram_comparison(
                full_histogram,
                output_histogram,
                target_histogram,
                title=f"Interval Class Histogram Comparison ({hand})",
                xlabel="Interval Class (-11~11)",
                ylabel="Frequency",
                output_path=f"{output_dir}/interval_class_histogram_comparison_{hand}.png"
            )
    else:
        for hand, histogram in input_data["pitch_histogram"].items():
            min_key = min(histogram.keys(), default=0)
            max_key = max(histogram.keys(), default=127)
            full_histogram = get_full_histogram(histogram, min_key, max_key)

            plot_histogram(
                full_histogram,
                title=f"Pitch Histogram ({hand})",
                xlabel="Pitch (MIDI 0-127)",
                ylabel="Frequency",
                output_path=f"{output_dir}/pitch_histogram_{hand}.png"
            )

        for hand, distribution in input_data["pitch_class_distribution"].items():
            full_distribution = get_full_histogram(
                {PITCH_CLASSES[key]: value for key, value in distribution.items()}, 0, 11
            )
            plot_histogram(
                full_distribution,
                title=f"Pitch Class Distribution ({hand})",
                xlabel="Pitch Class (0-11)",
                ylabel="Frequency",
                output_path=f"{output_dir}/pitch_class_distribution_{hand}.png"
            )

        for hand, histogram in input_data["interval_histogram"].items():
            min_key = min(histogram.keys(), default=-127)
            max_key = max(histogram.keys(), default=127)
            full_histogram = get_full_histogram(histogram, min_key, max_key)
            plot_histogram(
                full_histogram,
                title=f"Interval Histogram ({hand})",
                xlabel="Interval (-127~127)",
                ylabel="Frequency",
                output_path=f"{output_dir}/interval_histogram_{hand}.png"
            )

        for hand, histogram in input_data["interval_class_histogram"].items():
            full_histogram = get_full_histogram(histogram, -11, 11)
            full_histogram = merge_symmetric_intervals(full_histogram)
            plot_histogram(
                full_histogram,
                title=f"Interval Class Histogram ({hand})",
                xlabel="Interval Class (-11~11)",
                ylabel="Frequency",
                output_path=f"{output_dir}/interval_class_histogram_{hand}.png"
            )

def plot_rhythm_analysis(output_dir, input_data, output_data=None, target_data=None):
    """Plot rhythm-related analysis results."""
    if target_data:
        for hand, histogram in input_data["note_duration_histogram"].items():
            max_duration = max(
                max(histogram.keys(), default=0),
                max(output_data["note_duration_histogram"][hand].keys(), default=0),
                max(target_data["note_duration_histogram"][hand].keys(), default=0)
            )
            full_histogram = get_full_histogram(histogram, 0, max_duration)
            output_histogram = get_full_histogram(output_data["note_duration_histogram"][hand], 0, max_duration)
            target_histogram = get_full_histogram(target_data["note_duration_histogram"][hand], 0, max_duration)
            plot_histogram_comparison(
                full_histogram,
                output_histogram,
                target_histogram,
                title=f"Note Duration Histogram Comparison ({hand})",
                xlabel="Duration",
                ylabel="Frequency",
                output_path=f"{output_dir}/note_duration_histogram_comparison_{hand}.png"
            )
    else:
        for hand, histogram in input_data["note_duration_histogram"].items():
            max_duration = max(histogram.keys(), default=0)
            full_histogram = get_full_histogram(histogram, 0, max_duration)
            plot_histogram(
                full_histogram,
                title=f"Note Duration Histogram ({hand})",
                xlabel="Duration",
                ylabel="Frequency",
                output_path=f"{output_dir}/note_duration_histogram_{hand}.png"
            )

def plot_analysis(output_dir, input_data, output_data=None,
                  target_data=None):
    """Plot all analysis results (pitch/rhythm) and structure debug if available."""
    if target_data:
        plot_pitch_analysis(output_dir, input_data["pitch"], output_data["pitch"], target_data["pitch"])
        plot_rhythm_analysis(output_dir, input_data["rhythm"], output_data["rhythm"], target_data["rhythm"])

        try:
            plot_structure_analysis(output_dir, input_data, name="input")
            plot_structure_analysis(output_dir, output_data, name="output")
            plot_structure_analysis(output_dir, target_data, name="target")
        except Exception as e:
            pass
    else:
        plot_pitch_analysis(output_dir, input_data["pitch"])
        plot_rhythm_analysis(output_dir, input_data["rhythm"])
        try:
            plot_structure_analysis(output_dir, input_data, name="input")
        except Exception as e:
            pass







from matplotlib.patches import Rectangle
from typing import List, Tuple, Dict, Optional

def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    kernel = np.ones(win, dtype=float) / win
    y = np.convolve(xp, kernel, mode='valid')
    return y

def find_repetitions_from_ssm(
    S: np.ndarray,
    sim_thresh: float = 0.8,
    min_run: int = 4,
    max_offset: Optional[int] = None,
    smooth_win: int = 5
) -> List[Dict[str, Tuple[int,int]]]:
    n = S.shape[0]
    if n == 0:
        return []
    if max_offset is None:
        max_offset = n // 2
    reps: List[Dict[str, Tuple[int,int]]] = []
    for d in range(1, max_offset+1):
        diag = np.diag(S, k=d)
        if diag.size < min_run:
            continue
        sm = _moving_average(diag, smooth_win)
        on = sm >= sim_thresh
        i = 0
        while i < len(on):
            if not on[i]:
                i += 1; continue
            j = i + 1
            while j < len(on) and on[j]:
                j += 1
            if (j - i) >= min_run:
                reps.append({"a": (i, j), "b": (i+d, j+d), "offset": d, "mean_sim": float(diag[i:j].mean())})
            i = j
    reps.sort(key=lambda r: (r["offset"], r["a"][0], -(r["a"][1]-r["a"][0])))
    merged: List[Dict[str, Tuple[int,int]]] = []
    def _ovl(x0,x1,y0,y1): return max(0, min(x1,y1)-max(x0,y0))
    for r in reps:
        if not merged:
            merged.append(r); continue
        last = merged[-1]
        if r["offset"] == last["offset"]:
            a0,a1 = last["a"]; b0,b1 = last["b"]
            a0n,a1n = r["a"];  b0n,b1n = r["b"]
            oa = _ovl(a0,a1,a0n,a1n)
            ob = _ovl(b0,b1,b0n,b1n)
            if oa >= 0.5*min(a1-a0, a1n-a0n) and ob >= 0.5*min(b1-b0, b1n-b0n):
                merged[-1] = {
                    "a": (min(a0,a0n), max(a1,a1n)),
                    "b": (min(b0,b0n), max(b1,b1n)),
                    "offset": r["offset"],
                    "mean_sim": max(last["mean_sim"], r["mean_sim"])
                }
            else:
                merged.append(r)
        else:
            merged.append(r)
    return merged

def plot_ssm_with_annotations(S: np.ndarray,
                              cuts: Optional[List[int]] = None,
                              reps: Optional[List[Dict[str, Tuple[int,int]]]] = None,
                              title: str = "Self-Similarity Matrix (SSM)",
                              output_path: Optional[str] = None) -> None:
    plt.figure(figsize=(6,6))
    plt.imshow(S, interpolation="nearest")
    n = S.shape[0]
    plt.xlim(0, n)
    plt.ylim(n, 0)
    if cuts:
        for c in cuts:
            plt.axvline(c, linewidth=1)
            plt.axhline(c, linewidth=1)
    if reps:
        ax = plt.gca()
        for r in reps:
            (a0,a1) = r["a"]
            (b0,b1) = r["b"]
            ax.add_patch(Rectangle((a0, b0), a1-a0, b1-b0, fill=False, linewidth=1.5))
            ax.add_patch(Rectangle((b0, a0), b1-b0, a1-a0, fill=False, linewidth=1.0))
    plt.title(title)
    plt.xlabel("Position (measures)")
    plt.ylabel("Position (measures)")
    plt.colorbar(label="Similarity")
    if output_path:
        plt.tight_layout(); plt.savefig(output_path); plt.close()
    else:
        plt.show()

def plot_novelty_with_peaks(nov: np.ndarray,
                            thresh_val: Optional[float] = None,
                            peaks: Optional[List[int]] = None,
                            title: str = "Novelty Function",
                            output_path: Optional[str] = None) -> None:
    x = np.arange(len(nov))
    plt.figure(figsize=(7,3))
    plt.plot(x, nov, linewidth=1.5)
    if thresh_val is not None:
        plt.axhline(thresh_val, linestyle="--", linewidth=1)
    if peaks:
        for p in peaks:
            plt.axvline(p, linestyle=":", linewidth=1)
    plt.title(title)
    plt.xlabel("Position (downsampled units)")
    plt.ylabel("Novelty")
    plt.ylim(0, max(1.0, float(np.max(nov)) + 0.05))
    if output_path:
        plt.tight_layout(); plt.savefig(output_path); plt.close()
    else:
        plt.show()

def plot_structure_analysis(output_dir: str, analyzed: dict, name: str = "input") -> None:
    """Given analyzed_data (which includes structure with S & nov), save SSM & novelty debug plots."""
    struct = analyzed.get("structure", {})
    S = np.array(struct.get("S", []), dtype=float)
    nov = np.array(struct.get("nov", []), dtype=float)
    sections = struct.get("sections", [])
    if S.size == 0 or nov.size == 0 or not sections:
        return
    cuts = [e for (s,e) in sections[:-1]] if len(sections) > 1 else []
    reps = find_repetitions_from_ssm(S, sim_thresh=0.8, min_run=4, smooth_win=5)
    mu, sigma = float(nov.mean()), float(nov.std())
    thresh_val = max(0.0, min(1.0, mu + 0.5*sigma))
    peaks = cuts
    ssm_path = f"{output_dir}/structure_ssm_{name}.png"
    nov_path = f"{output_dir}/structure_novelty_{name}.png"
    plot_ssm_with_annotations(S, cuts=cuts, reps=reps, title=f"SSM with Cuts & Repeats ({name})", output_path=ssm_path)
    plot_novelty_with_peaks(nov, thresh_val=thresh_val, peaks=peaks, title=f"Novelty & Cuts ({name})", output_path=nov_path)
