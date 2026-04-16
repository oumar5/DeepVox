#!/usr/bin/env python3
"""Run all 4 experimental conditions and produce a comparison report.

Usage:
    python scripts/run_all_conditions.py \
        --data-dir data/prepared \
        --textgrid-dir data/mfa-output \
        --output-dir outputs/phase1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deepvox.data.dataset import Condition, PhonemeDataset
from deepvox.training.phase1 import evaluate, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONDITIONS = [
    ("A", Condition.CODEC2_RAW, "Codec2 raw (48 feat)"),
    ("B", Condition.CODEC2_DELTA, "Codec2 + delta (96 feat)"),
    ("C", Condition.MEL_SPEC, "Mel spectrogram (80 feat)"),
    ("D", Condition.PCM_RAW, "PCM brut (320 feat)"),
]


def find_pairs(audio_dir: Path, textgrid_dir: Path):
    audio_paths, tg_paths = [], []
    for wav in sorted(audio_dir.glob("*.wav")):
        tg = textgrid_dir / f"{wav.stem}.TextGrid"
        if tg.exists():
            audio_paths.append(wav)
            tg_paths.append(tg)
    return audio_paths, tg_paths


def split_data(audio_paths, tg_paths, train_ratio=0.8, dev_ratio=0.1):
    n = len(audio_paths)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    return (
        (audio_paths[:n_train], tg_paths[:n_train]),
        (audio_paths[n_train:n_train + n_dev], tg_paths[n_train:n_train + n_dev]),
        (audio_paths[n_train + n_dev:], tg_paths[n_train + n_dev:]),
    )


def main():
    parser = argparse.ArgumentParser(description="Run all Phase 1 conditions")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--textgrid-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--context-frames", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    audio_paths, tg_paths = find_pairs(args.data_dir, args.textgrid_dir)
    logger.info("Found %d pairs", len(audio_paths))
    train_split, dev_split, test_split = split_data(audio_paths, tg_paths)

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    for label, condition, desc in CONDITIONS:
        logger.info("=" * 60)
        logger.info("Condition %s: %s", label, desc)
        logger.info("=" * 60)

        train_ds = PhonemeDataset(*train_split, condition=condition, context_frames=args.context_frames)
        dev_ds = PhonemeDataset(*dev_split, condition=condition, context_frames=args.context_frames)
        test_ds = PhonemeDataset(*test_split, condition=condition, context_frames=args.context_frames)

        model = train(
            train_dataset=train_ds,
            dev_dataset=dev_ds,
            condition=condition,
            output_dir=args.output_dir,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
        )

        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        per, acc = evaluate(model, test_loader, device)

        results.append({
            "condition": label,
            "description": desc,
            "per": per,
            "accuracy": acc,
            "params": model.count_parameters(),
        })

        logger.info("Condition %s → PER=%.3f, Acc=%.3f", label, per, acc)

    # Write comparison report
    report_lines = [
        "# Phase 1 — Comparison Report (All Conditions)",
        "",
        "| Condition | Description | PER | Accuracy | Params |",
        "|-----------|-------------|-----|----------|--------|",
    ]

    for r in results:
        report_lines.append(
            f"| {r['condition']} | {r['description']} | "
            f"{r['per']:.4f} ({r['per']*100:.1f}%) | "
            f"{r['accuracy']:.4f} ({r['accuracy']*100:.1f}%) | "
            f"{r['params']:,} |"
        )

    # Decision
    best = min(results, key=lambda x: x["per"])
    report_lines += [
        "",
        "## Best Condition",
        "",
        f"**{best['condition']}** — {best['description']} — PER = {best['per']*100:.1f}%",
        "",
        "## Decision",
        "",
    ]

    if best["per"] <= 0.15:
        report_lines.append(f"**GO** — PER = {best['per']*100:.1f}% (≤15%). Phase 2 validated.")
    elif best["per"] <= 0.25:
        report_lines.append(f"**GO conditionnel** — PER = {best['per']*100:.1f}% (15-25%).")
    else:
        report_lines.append(f"**NO-GO** — PER = {best['per']*100:.1f}% (>25%). Investigate further.")

    report = "\n".join(report_lines)
    report_path = args.output_dir / "phase1_comparison.md"
    report_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    logger.info("Report saved → %s", report_path)


if __name__ == "__main__":
    main()
