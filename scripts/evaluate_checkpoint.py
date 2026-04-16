#!/usr/bin/env python3
"""Évalue un checkpoint Phase 1 existant sur le test set.

Utilisé pour récupérer les métriques finales d'un run interrompu.

Usage:
    python scripts/evaluate_checkpoint.py \
        --checkpoint outputs/phase1_run4/best_model_A.pt \
        --data-dir data/prepared_80k \
        --textgrid-dir data/mfa-output-80k \
        --condition A \
        --output-dir outputs/phase1_run4
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deepvox.data.dataset import Condition, PhonemeDataset
from deepvox.models.phoneme_classifier import PhonemeClassifier
from deepvox.training.phase1 import evaluate_and_report

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
        audio_paths[n_train + n_dev :],
        tg_paths[n_train + n_dev :],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--textgrid-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=["A", "B", "C", "D"], default="A")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--context-frames", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    condition = CONDITION_MAP[args.condition]

    audio_paths, tg_paths = find_pairs(args.data_dir, args.textgrid_dir)
    logger.info("Found %d pairs", len(audio_paths))
    test_audio, test_tg = split_data(audio_paths, tg_paths)
    logger.info("Test pairs: %d", len(test_audio))

    logger.info("Building test dataset...")
    test_ds = PhonemeDataset(
        test_audio, test_tg,
        condition=condition,
        context_frames=args.context_frames,
    )
    logger.info("Test samples: %d", len(test_ds))

    device = args.device
    model = PhonemeClassifier(input_dim=test_ds.feature_dim)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    logger.info("Checkpoint loaded: %s", args.checkpoint)

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    report_path = args.output_dir / f"phase1_results_{args.condition}.md"
    report = evaluate_and_report(model, test_loader, device, report_path)
    print("\n" + report)
    logger.info("Report → %s", report_path)


if __name__ == "__main__":
    main()
