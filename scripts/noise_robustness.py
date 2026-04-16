#!/usr/bin/env python3
"""Noise robustness analysis: measure PER degradation under noise.

Adds white or pink noise at various SNR levels to PCM before Codec2 encoding,
then evaluates the trained classifier.

Usage:
    python scripts/noise_robustness.py \
        --model-path outputs/phase1/best_model_A.pt \
        --data-dir data/prepared \
        --textgrid-dir data/mfa-output \
        --output-dir outputs/phase1/noise
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from deepvox.codec2.encoder import (
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    encode_pcm,
    unpack_frames,
)
from deepvox.data.preprocess import PHONEME_TO_IDX, align_phonemes_to_codec2_grid, parse_textgrid
from deepvox.eval.metrics import phone_error_rate
from deepvox.models.phoneme_classifier import PhonemeClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SNR_LEVELS = [30, 20, 15, 10, 5, 0]  # dB


def add_noise(pcm_float: np.ndarray, snr_db: float, noise_type: str = "white") -> np.ndarray:
    """Add noise to audio signal at a given SNR."""
    signal_power = np.mean(pcm_float ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    if noise_type == "white":
        noise = np.random.randn(len(pcm_float)) * np.sqrt(noise_power)
    elif noise_type == "pink":
        # Generate pink noise via spectral shaping
        white = np.random.randn(len(pcm_float))
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(len(white))
        freqs[0] = 1  # avoid division by zero
        fft /= np.sqrt(freqs)
        pink = np.fft.irfft(fft, n=len(pcm_float))
        pink *= np.sqrt(noise_power / np.mean(pink ** 2))
        noise = pink
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    return pcm_float + noise


class NoisyDataset(Dataset):
    """Dataset that adds noise before Codec2 encoding."""

    def __init__(self, audio_paths, tg_paths, snr_db, noise_type="white", context_frames=5):
        self.features = []
        self.labels = []

        for audio_path, tg_path in zip(audio_paths, tg_paths, strict=True):
            try:
                audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
            except Exception:
                continue

            # Add noise before encoding
            noisy = add_noise(audio, snr_db, noise_type)
            pcm = (np.clip(noisy, -1, 1) * 32767).astype(np.int16)

            n_frames = len(pcm) // SAMPLES_PER_FRAME
            if n_frames == 0:
                continue

            frames = encode_pcm(pcm)
            feats = unpack_frames(frames)

            intervals = parse_textgrid(tg_path)
            phonemes = align_phonemes_to_codec2_grid(intervals, n_frames)
            labs = [PHONEME_TO_IDX.get(p, -1) for p in phonemes]

            for i, label in enumerate(labs):
                if label == -1:
                    continue
                s = max(0, i - context_frames)
                e = min(len(feats), i + context_frames + 1)
                ff = feats[s:e]
                expected = 2 * context_frames + 1
                if len(ff) < expected:
                    pad = expected - len(ff)
                    ff = np.pad(ff, ((pad, 0) if s == 0 else (0, pad), (0, 0)))
                self.features.append(ff)
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]).float(), self.labels[idx]


def main():
    parser = argparse.ArgumentParser(description="Noise robustness analysis")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--textgrid-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1/noise"))
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    model = PhonemeClassifier(input_dim=48)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # Find test files
    audio_paths, tg_paths = [], []
    for wav in sorted(args.data_dir.glob("*.wav")):
        tg = args.textgrid_dir / f"{wav.stem}.TextGrid"
        if tg.exists():
            audio_paths.append(wav)
            tg_paths.append(tg)

    # Use last 10% as test
    n_test = max(1, len(audio_paths) // 10)
    test_audio = audio_paths[-n_test:]
    test_tg = tg_paths[-n_test:]

    results = []

    for noise_type in ["white", "pink"]:
        for snr in SNR_LEVELS:
            logger.info("Evaluating: %s noise, SNR=%d dB", noise_type, snr)
            ds = NoisyDataset(test_audio, test_tg, snr, noise_type)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

            all_preds, all_targets = [], []
            with torch.no_grad():
                for feats, labels in loader:
                    feats = feats.to(device)
                    labels = torch.tensor(labels, dtype=torch.long, device=device)
                    logits = model(feats)
                    preds = logits.reshape(-1, logits.size(-1)).argmax(dim=-1)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(labels.reshape(-1).cpu().numpy())

            per = phone_error_rate(np.array(all_preds), np.array(all_targets))
            results.append({"noise": noise_type, "snr": snr, "per": per})
            logger.info("  PER = %.3f", per)

    # Report
    lines = [
        "# Noise Robustness Analysis",
        "",
        "| Noise Type | SNR (dB) | PER |",
        "|------------|----------|-----|",
    ]
    for r in results:
        lines.append(f"| {r['noise']} | {r['snr']} | {r['per']:.4f} ({r['per']*100:.1f}%) |")

    report = "\n".join(lines)
    report_path = args.output_dir / "noise_robustness.md"
    report_path.write_text(report, encoding="utf-8")
    print("\n" + report)
    logger.info("Report → %s", report_path)


if __name__ == "__main__":
    main()
