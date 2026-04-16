#!/usr/bin/env python3
"""Phase 2 — ASR pipeline (Codec2 → French text) with CTC.

Usage:
    python scripts/phase2_asr.py \
        --data-dir data/prepared_20k \
        --output-dir outputs/phase2 \
        --max-epochs 30 \
        --batch-size 32
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deepvox.data.ctc_dataset import ASRDataset, ctc_collate_fn
from deepvox.training.phase2_asr import evaluate_and_report, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_wav_lab_pairs(data_dir: Path) -> tuple[list[Path], list[Path]]:
    """Find matching (WAV, .lab) pairs in data_dir."""
    audio_paths, lab_paths = [], []
    for wav in sorted(data_dir.glob("*.wav")):
        # Resolve symlinks if needed
        lab = data_dir / f"{wav.stem}.lab"
        if lab.exists():
            audio_paths.append(wav)
            lab_paths.append(lab)
    return audio_paths, lab_paths


def split_data(audio_paths, lab_paths, train_ratio=0.9, dev_ratio=0.05):
    n = len(audio_paths)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    return (
        (audio_paths[:n_train], lab_paths[:n_train]),
        (audio_paths[n_train : n_train + n_dev], lab_paths[n_train : n_train + n_dev]),
        (audio_paths[n_train + n_dev :], lab_paths[n_train + n_dev :]),
    )


def main():
    parser = argparse.ArgumentParser(description="DeepVox Phase 2 — ASR")
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with WAV + .lab")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-duration-s", type=float, default=15.0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    logger.info("=== DeepVox Phase 2 — ASR ===")

    audio_paths, lab_paths = find_wav_lab_pairs(args.data_dir)
    logger.info("Found %d (wav, lab) pairs", len(audio_paths))

    if len(audio_paths) == 0:
        logger.error("No data found in %s", args.data_dir)
        return

    train_split, dev_split, test_split = split_data(audio_paths, lab_paths)
    logger.info(
        "Split: train=%d, dev=%d, test=%d",
        len(train_split[0]), len(dev_split[0]), len(test_split[0]),
    )

    logger.info("Building datasets (max_duration=%.1fs)...", args.max_duration_s)
    train_ds = ASRDataset(*train_split, max_duration_s=args.max_duration_s)
    dev_ds = ASRDataset(*dev_split, max_duration_s=args.max_duration_s)
    test_ds = ASRDataset(*test_split, max_duration_s=args.max_duration_s)

    logger.info("Dataset stats (train): %s", train_ds.stats())
    logger.info("Dataset stats (dev): %s", dev_ds.stats())

    model = train(
        train_dataset=train_ds,
        dev_dataset=dev_ds,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        device=args.device,
    )

    # Test evaluation
    device = args.device or (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=args.num_workers,
    )
    report = evaluate_and_report(
        model, test_loader, device,
        output_path=args.output_dir / "phase2_results.md",
    )

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)


if __name__ == "__main__":
    main()
