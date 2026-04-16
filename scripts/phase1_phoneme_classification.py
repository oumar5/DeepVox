#!/usr/bin/env python3
"""Phase 1 — Phoneme classification pipeline (end-to-end).

Usage:
    python scripts/phase1_phoneme_classification.py \
        --data-dir data/common-voice-french \
        --textgrid-dir data/mfa-output \
        --condition A \
        --output-dir outputs/phase1

Conditions:
    A  Codec2 raw (48 features)
    B  Codec2 + delta (96 features)
    C  Mel spectrogram baseline (80 features)
    D  Raw PCM baseline (320 features)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deepvox.data.dataset import Condition, PhonemeDataset
from deepvox.training.phase1 import evaluate_and_report, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONDITION_MAP = {
    "A": Condition.CODEC2_RAW,
    "B": Condition.CODEC2_DELTA,
    "C": Condition.MEL_SPEC,
    "D": Condition.PCM_RAW,
}


def find_pairs(audio_dir: Path, textgrid_dir: Path) -> tuple[list[Path], list[Path]]:
    """Find matching (wav, TextGrid) pairs."""
    audio_paths = []
    tg_paths = []

    for wav in sorted(audio_dir.glob("*.wav")):
        tg = textgrid_dir / f"{wav.stem}.TextGrid"
        if tg.exists():
            audio_paths.append(wav)
            tg_paths.append(tg)

    return audio_paths, tg_paths


def split_data(
    audio_paths: list[Path],
    tg_paths: list[Path],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
) -> tuple[tuple, tuple, tuple]:
    """Split data into train/dev/test."""
    n = len(audio_paths)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    train_split = (audio_paths[:n_train], tg_paths[:n_train])
    dev_split = (audio_paths[n_train:n_train + n_dev], tg_paths[n_train:n_train + n_dev])
    test_split = (audio_paths[n_train + n_dev:], tg_paths[n_train + n_dev:])

    return train_split, dev_split, test_split


def main():
    parser = argparse.ArgumentParser(description="DeepVox Phase 1 — Phoneme Classification")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with 8 kHz WAV files")
    parser.add_argument("--textgrid-dir", type=Path, required=True, help="Directory with MFA TextGrid files")
    parser.add_argument("--condition", choices=["A", "B", "C", "D"], default="A")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--context-frames", type=int, default=5, help="Context frames on each side")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    condition = CONDITION_MAP[args.condition]
    logger.info("=== DeepVox Phase 1 — Condition %s ===", args.condition)

    # Find data pairs
    audio_paths, tg_paths = find_pairs(args.data_dir, args.textgrid_dir)
    logger.info("Found %d audio/TextGrid pairs", len(audio_paths))

    if len(audio_paths) == 0:
        logger.error("No data found. Run preprocessing first.")
        return

    # Split
    train_split, dev_split, test_split = split_data(audio_paths, tg_paths)
    logger.info("Split: train=%d, dev=%d, test=%d",
                len(train_split[0]), len(dev_split[0]), len(test_split[0]))

    # Build datasets
    logger.info("Building datasets (condition=%s, context=%d)...", args.condition, args.context_frames)
    train_ds = PhonemeDataset(*train_split, condition=condition, context_frames=args.context_frames)
    dev_ds = PhonemeDataset(*dev_split, condition=condition, context_frames=args.context_frames)
    test_ds = PhonemeDataset(*test_split, condition=condition, context_frames=args.context_frames)

    logger.info("Samples: train=%d, dev=%d, test=%d", len(train_ds), len(dev_ds), len(test_ds))

    # Train
    model = train(
        train_dataset=train_ds,
        dev_dataset=dev_ds,
        condition=condition,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=args.device,
    )

    # Evaluate on test set
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    device = args.device or ("mps" if torch.backends.mps.is_available()
                              else "cuda" if torch.cuda.is_available() else "cpu")

    report_path = args.output_dir / f"phase1_results_{args.condition}.md"
    report = evaluate_and_report(model, test_loader, device, report_path)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    logger.info("Done. Report saved to %s", report_path)


if __name__ == "__main__":
    main()
