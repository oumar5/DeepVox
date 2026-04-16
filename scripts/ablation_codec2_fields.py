#!/usr/bin/env python3
"""Ablation study: mask Codec2 fields to measure phonetic contribution.

Masks each field (LSP, pitch, energy) independently and measures PER delta.
This tells us which Codec2 fields carry the most phonetic information.

Usage:
    python scripts/ablation_codec2_fields.py \
        --data-dir data/prepared \
        --textgrid-dir data/mfa-output \
        --output-dir outputs/phase1/ablation
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from deepvox.codec2.encoder import SAMPLE_RATE, SAMPLES_PER_FRAME, encode_pcm, unpack_frames
from deepvox.data.preprocess import (
    PHONEME_TO_IDX,
    align_phonemes_to_codec2_grid,
    parse_textgrid,
)
from deepvox.training.phase1 import evaluate, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Codec2 1200 bps bit field layout
FIELDS = {
    "all":    (0, 48),   # No masking
    "no_lsp": (0, 36),   # Mask LSP bits
    "no_pitch": (36, 43), # Mask pitch bits
    "no_energy": (43, 48), # Mask energy bits
}


class AblationDataset(Dataset):
    """Dataset with selective field masking."""

    def __init__(self, audio_paths, tg_paths, mask_range=None, context_frames=5):
        self.features = []
        self.labels = []

        for audio_path, tg_path in zip(audio_paths, tg_paths, strict=True):
            try:
                audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
            except Exception:
                continue

            pcm = (audio * 32767).astype(np.int16)
            n_frames = len(pcm) // SAMPLES_PER_FRAME
            if n_frames == 0:
                continue

            frames = encode_pcm(pcm)
            feats = unpack_frames(frames)

            # Apply mask
            if mask_range is not None:
                start, end = mask_range
                feats[:, start:end] = 0.0

            intervals = parse_textgrid(tg_path)
            phonemes = align_phonemes_to_codec2_grid(intervals, n_frames)
            labs = [PHONEME_TO_IDX.get(p, -1) for p in phonemes]

            for i, label in enumerate(labs):
                if label == -1:
                    continue
                start_idx = max(0, i - context_frames)
                end_idx = min(len(feats), i + context_frames + 1)
                frame_feats = feats[start_idx:end_idx]

                expected = 2 * context_frames + 1
                if len(frame_feats) < expected:
                    pad = expected - len(frame_feats)
                    if start_idx == 0:
                        frame_feats = np.pad(frame_feats, ((pad, 0), (0, 0)))
                    else:
                        frame_feats = np.pad(frame_feats, ((0, pad), (0, 0)))

                self.features.append(frame_feats)
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]).float(), self.labels[idx]


def find_pairs(audio_dir, textgrid_dir):
    audio_paths, tg_paths = [], []
    for wav in sorted(audio_dir.glob("*.wav")):
        tg = textgrid_dir / f"{wav.stem}.TextGrid"
        if tg.exists():
            audio_paths.append(wav)
            tg_paths.append(tg)
    return audio_paths, tg_paths


def main():
    parser = argparse.ArgumentParser(description="Ablation study on Codec2 fields")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--textgrid-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1/ablation"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--context-frames", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    audio_paths, tg_paths = find_pairs(args.data_dir, args.textgrid_dir)
    n = len(audio_paths)
    n_train = int(n * 0.8)
    n_dev = int(n * 0.1)

    train_audio, train_tg = audio_paths[:n_train], tg_paths[:n_train]
    dev_audio, dev_tg = audio_paths[n_train:n_train+n_dev], tg_paths[n_train:n_train+n_dev]
    test_audio, test_tg = audio_paths[n_train+n_dev:], tg_paths[n_train+n_dev:]

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    for name, (start, end) in FIELDS.items():
        mask_range = (start, end) if name != "all" else None
        logger.info("=== Ablation: %s (mask bits %s) ===", name, f"{start}-{end}" if mask_range else "none")

        train_ds = AblationDataset(train_audio, train_tg, mask_range, args.context_frames)
        dev_ds = AblationDataset(dev_audio, dev_tg, mask_range, args.context_frames)
        test_ds = AblationDataset(test_audio, test_tg, mask_range, args.context_frames)

        from deepvox.data.dataset import Condition
        model = train(
            train_dataset=train_ds,
            dev_dataset=dev_ds,
            condition=Condition.CODEC2_RAW,
            output_dir=args.output_dir,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
        )

        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
        per, acc = evaluate(model, test_loader, device)

        results.append({"name": name, "per": per, "accuracy": acc})
        logger.info("%s → PER=%.3f, Acc=%.3f", name, per, acc)

    # Report
    baseline_per = next(r["per"] for r in results if r["name"] == "all")

    lines = [
        "# Ablation Study — Codec2 Field Contribution",
        "",
        "| Condition | Masked bits | PER | Accuracy | ΔPER vs baseline |",
        "|-----------|------------|-----|----------|-----------------|",
    ]

    for r in results:
        delta = r["per"] - baseline_per
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"| {r['name']} | {FIELDS[r['name']]} | "
            f"{r['per']:.4f} | {r['accuracy']:.4f} | {sign}{delta:.4f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "A larger ΔPER means the masked field carries more phonetic information.",
        "If ΔPER is close to 0, the field contributes little to phoneme discrimination.",
    ]

    report = "\n".join(lines)
    report_path = args.output_dir / "ablation_report.md"
    report_path.write_text(report, encoding="utf-8")

    print("\n" + report)
    logger.info("Report → %s", report_path)


if __name__ == "__main__":
    main()
