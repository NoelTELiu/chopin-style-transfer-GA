"""End-to-end Chopin-style transfer pipeline."""

from __future__ import annotations

import csv
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .fitness.main import (
    analyze_function, similarity_function, fitness_function, get_fitness_weights,
)
from .fitness.structure import structure_similarity
from .ga import GA
from .music import count_notes_in_chromosome, read_chromosome_from_file, tonic_name_to_midi, two_hand_chromosome_to_midi
from .ngram import read_ngram_model_from_file
from .representation import parse_musicxml_to_json


@dataclass(frozen=True)
class RunConfig:
    input_score: Path
    data_dir: Path
    output_root: Path
    generations: int = 1000
    population_size: int = 100
    crossover_rate: float = 0.9
    origin_threshold: int = 400
    trials: int = 15
    tournament_size: int = 3
    ngram_order: int = 3
    seed: int | None = None
    target: Path | None = None
    weight_profile: str = "balanced"


def _json_dump(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _resolve_data_paths(data_dir: Path) -> dict[str, Path]:
    paths = {
        "features": data_dir / "chopin_nocturnes" / "features",
        "chromosomes": data_dir / "chopin_nocturnes" / "chromosomes",
        "musicxml": data_dir / "chopin_nocturnes" / "musicxml",
        "models": data_dir / "models",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required data directories: " + ", ".join(missing))
    return paths


def select_target(original_chromosome: dict, data_dir: Path) -> tuple[Path, float]:
    """Select the Chopin reference most similar to the whole input piece.

    Select a reference using whole-piece musical features from both hands together
    with structural similarity.
    """
    paths = _resolve_data_paths(data_dir)
    origin_features = analyze_function(original_chromosome, ignore_syntax=True)
    best_similarity = float("-inf")
    best_path: Path | None = None

    for chromosome_path in sorted(paths["chromosomes"].glob("*.json")):


        target_chromosome = read_chromosome_from_file(chromosome_path)
        target_features = analyze_function(target_chromosome, ignore_syntax=True)

        right = similarity_function(origin_features, target_features, "right_hand")
        left = similarity_function(origin_features, target_features, "left_hand")
        feature_similarity = 0.5 * (right + left)
        struct = structure_similarity(origin_features["structure"], target_features["structure"])
        similarity = 0.9 * feature_similarity + 0.1 * struct

        if similarity > best_similarity:
            best_similarity = similarity
            best_path = chromosome_path

    if best_path is None:
        raise RuntimeError("No usable reference nocturne was found in the data directory.")
    return best_path, best_similarity


def _write_fitness_csv(path: Path, melody: list[float], accompaniment: list[float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "melody_fitness", "accompaniment_fitness"])
        for gen in range(max(len(melody), len(accompaniment))):
            writer.writerow([
                gen,
                melody[gen] if gen < len(melody) else "",
                accompaniment[gen] if gen < len(accompaniment) else "",
            ])


def run_transfer(config: RunConfig) -> Path:
    """Run the two-stage GA and return the timestamped run directory."""
    if config.generations < 1:
        raise ValueError("generations must be >= 1")
    if config.population_size < config.tournament_size:
        raise ValueError("population_size must be >= tournament_size")
    if config.trials < 1:
        raise ValueError("trials must be >= 1")
    if not config.input_score.exists():
        raise FileNotFoundError(config.input_score)

    fitness_weights = get_fitness_weights(config.weight_profile)
    paths = _resolve_data_paths(config.data_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = config.output_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    input_json = run_dir / "input_chromosome.json"
    parse_musicxml_to_json(str(config.input_score), str(input_json))
    original = read_chromosome_from_file(input_json)

    bpm = original.get("bpm", 120)
    tonic_name = original.get("key", {}).get("tonic", "C")
    mode = original.get("key", {}).get("mode", "major")
    tonic_midi = tonic_name_to_midi(tonic_name, octave=4)
    main_melody = original["right_hand"]
    accompaniment = original["left_hand"]

    right_ngram = read_ngram_model_from_file(paths["models"] / "right_pitch_ngram.txt")
    left_ngram = read_ngram_model_from_file(paths["models"] / "left_pitch_ngram.txt")

    if config.target is None:
        target_path, target_similarity = select_target(original, config.data_dir)
    else:
        target_path = config.target
        if not target_path.is_absolute() and not target_path.exists():
            candidate = paths["chromosomes"] / target_path
            if candidate.exists():
                target_path = candidate
            elif candidate.with_suffix(".json").exists():
                target_path = candidate.with_suffix(".json")
        if not target_path.exists():
            raise FileNotFoundError(f"Target chromosome not found: {target_path}")
        target_similarity = float("nan")

    target = read_chromosome_from_file(target_path)

    melody_target = {
        "right_hand": target["right_hand"],
        "left_hand": [[] for _ in target["left_hand"]],
        "measure_boundaries": target.get("measure_boundaries", []),
    }
    melody_origin = {
        "right_hand": main_melody,
        "left_hand": [[] for _ in accompaniment],
        "measure_boundaries": original.get("measure_boundaries", []),
    }
    melody_target_features = analyze_function(melody_target)
    melody_origin_features = analyze_function(melody_origin)
    full_target_features = analyze_function(target)
    full_origin_features = analyze_function(original)

    melody_note_num = count_notes_in_chromosome(main_melody)
    accompaniment_note_num = count_notes_in_chromosome(accompaniment)
    if melody_note_num == 0 or accompaniment_note_num == 0:
        raise ValueError("Input must contain notes in both right- and left-hand parts.")
    melody_length = len(main_melody)
    accompaniment_length = len(accompaniment)
    measure_boundaries = original.get("measure_boundaries", [])

    trial_records: list[dict] = []
    all_melody_fitness: list[list[float]] = []
    all_accompaniment_fitness: list[list[float]] = []

    for idx in range(config.trials):
        if config.seed is not None:
            trial_seed = config.seed + idx
            random.seed(trial_seed)
            np.random.seed(trial_seed)
        else:
            trial_seed = None

        start = time.time()
        ga_melody = GA(
            num_generations=config.generations,
            population_size=config.population_size,
            chromosome_length=melody_length,
            crossover_rate=config.crossover_rate,
            mutation_rate=1 / melody_length,
            target_features=melody_target_features,
            origin_features=melody_origin_features,
            tonic_offset=tonic_midi - 60,
            mode=mode,
            hand="right",
            k_tournament_size=config.tournament_size,
            pitch_ngram_model=right_ngram,
            ngram_parameter=config.ngram_order,
            fitness_weights=fitness_weights,
            measure_boundaries=measure_boundaries,
        )
        best_melody, melody_fitness, _ = ga_melody.run(
            main_melody, origin_add=config.origin_threshold
        )
        middle = time.time()

        ga_accompaniment = GA(
            num_generations=config.generations,
            population_size=config.population_size,
            chromosome_length=accompaniment_length,
            crossover_rate=config.crossover_rate,
            mutation_rate=1 / accompaniment_length,
            target_features=full_target_features,
            origin_features=full_origin_features,
            tonic_offset=tonic_midi - 60,
            mode=mode,
            hand="left",
            k_tournament_size=config.tournament_size,
            pitch_ngram_model=left_ngram,
            ngram_parameter=config.ngram_order,
            fitness_weights=fitness_weights,
            measure_boundaries=measure_boundaries,
        )
        best_accompaniment, accompaniment_fitness, _ = ga_accompaniment.run(
            accompaniment,
            best_melody,
            origin_add=config.origin_threshold,
        )
        end = time.time()

        combined = {
            "right_hand": best_melody,
            "left_hand": best_accompaniment,
            "key": {"tonic": tonic_name, "mode": mode},
            "bpm": bpm,
            "measure_boundaries": measure_boundaries,
        }
        stem = f"trial_{idx:02d}"
        json_path = run_dir / f"{stem}.json"
        midi_path = run_dir / f"{stem}.mid"
        csv_path = run_dir / f"{stem}_fitness.csv"
        _json_dump(combined, json_path)
        _write_fitness_csv(csv_path, melody_fitness, accompaniment_fitness)
        two_hand_chromosome_to_midi(
            best_melody,
            best_accompaniment,
            str(midi_path),
            bpm=bpm,
            tonic_midi=tonic_midi,
            mode=mode,
        )

        final_generation = max(0, config.generations - 1)
        melody_eval_data = analyze_function({
            "right_hand": best_melody,
            "left_hand": [[] for _ in best_melody],
            "measure_boundaries": measure_boundaries,
        })
        melody_total, melody_components = fitness_function(
            melody_eval_data, melody_target_features, "right", melody_origin_features,
            gen=final_generation, threshold=config.origin_threshold,
            weights=fitness_weights, return_components=True,
        )
        full_eval_data = analyze_function(combined)
        accompaniment_total, accompaniment_components = fitness_function(
            full_eval_data, full_target_features, "left", full_origin_features,
            gen=final_generation, threshold=config.origin_threshold,
            weights=fitness_weights, return_components=True,
        )

        final_score = accompaniment_fitness[-1] if accompaniment_fitness else float("-inf")
        trial_records.append({
            "trial": idx,
            "seed": trial_seed,
            "final_accompaniment_fitness": final_score,
            "final_melody_components": {"total": melody_total, **melody_components},
            "final_accompaniment_components": {"total": accompaniment_total, **accompaniment_components},
            "melody_seconds": middle - start,
            "accompaniment_seconds": end - middle,
            "total_seconds": end - start,
            "json": json_path.name,
            "midi": midi_path.name,
            "fitness_csv": csv_path.name,
        })
        all_melody_fitness.append(melody_fitness)
        all_accompaniment_fitness.append(accompaniment_fitness)

    best = max(trial_records, key=lambda r: r["final_accompaniment_fitness"])
    shutil.copy2(run_dir / best["midi"], run_dir / "best.mid")
    shutil.copy2(run_dir / best["json"], run_dir / "best.json")

    plt.figure()
    plt.plot(np.mean(np.asarray(all_melody_fitness), axis=0), label="Melody Fitness")
    plt.plot(np.mean(np.asarray(all_accompaniment_fitness), axis=0), label="Accompaniment Fitness")
    plt.legend()
    plt.title(f"Mean Best Fitness ({config.trials} trial(s))")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.tight_layout()
    plt.savefig(run_dir / "fitness_plot.png", dpi=160)
    plt.close()

    config_dict = asdict(config)
    for key, value in list(config_dict.items()):
        if isinstance(value, Path):
            config_dict[key] = str(value)
    summary = {
        "config": config_dict,
        "selected_target": str(target_path),
        "target_selection_similarity": None if np.isnan(target_similarity) else target_similarity,
        "best_trial": best,
        "trials": trial_records,
    }
    _json_dump(summary, run_dir / "run_summary.json")
    return run_dir
