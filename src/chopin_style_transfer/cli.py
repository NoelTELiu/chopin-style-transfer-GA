"""Command-line interface for the style-transfer pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chopin-style-transfer",
        description="Transform a symbolic piano score toward Chopin nocturne style using a two-stage GA.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Input MusicXML/MXL piano score.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Reference-data directory (default: data).")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Root directory for timestamped runs.")
    parser.add_argument("--target", type=Path, default=None, help="Optional encoded target JSON (path or filename).")
    parser.add_argument("--generations", type=int, default=1000, help="Generations for each GA stage (default: 1000).")
    parser.add_argument("--population-size", type=int, default=100, help="Population size (default: 100).")
    parser.add_argument("--trials", type=int, default=15, help="Independent trials (default: 15).")
    parser.add_argument("--crossover-rate", type=float, default=0.9, help="Crossover probability (default: 0.9).")
    parser.add_argument("--origin-threshold", type=int, default=400, help="Generation at which origin similarity is enabled (default: 400).")
    parser.add_argument("--tournament-size", type=int, default=3, help="Tournament selection size (default: 3).")
    parser.add_argument("--ngram-order", type=int, default=3, help="Pitch n-gram order (default: 3).")
    parser.add_argument(
        "--weight-profile",
        choices=["balanced", "target", "syntax", "input"],
        default="balanced",
        help="Fitness weights: balanced, target-focused, syntax-focused, or input-focused.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for controlled runs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from .pipeline import RunConfig, run_transfer

    config = RunConfig(
        input_score=args.input,
        data_dir=args.data_dir,
        output_root=args.output_dir,
        generations=args.generations,
        population_size=args.population_size,
        crossover_rate=args.crossover_rate,
        origin_threshold=args.origin_threshold,
        trials=args.trials,
        tournament_size=args.tournament_size,
        ngram_order=args.ngram_order,
        seed=args.seed,
        target=args.target,
        weight_profile=args.weight_profile,
    )
    run_dir = run_transfer(config)
    print(f"Run complete: {run_dir}")
    print(f"Best MIDI: {run_dir / 'best.mid'}")


if __name__ == "__main__":
    main()
