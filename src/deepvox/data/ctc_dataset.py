"""PyTorch Dataset for DeepVox Phase 2 ASR (CTC training).

Each sample: (codec2_frames, char_ids)
  - codec2_frames: (T, 48) — variable T
  - char_ids: (L,) — variable L, typically L < T

Uses CTC collation to pad variable-length sequences.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from deepvox.codec2.encoder import (
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    encode_pcm,
    unpack_frames,
)
from deepvox.data.text import encode, normalize_text


class ASRDataset(Dataset):
    """Dataset of (codec2_frames, char_ids) pairs for CTC training."""

    def __init__(
        self,
        audio_paths: list[Path],
        lab_paths: list[Path],
        max_duration_s: float = 15.0,
        min_duration_s: float = 0.5,
    ):
        """
        Args:
            audio_paths: list of 8 kHz WAV file paths.
            lab_paths: matching .lab transcription file paths.
            max_duration_s: skip clips longer than this (memory).
            min_duration_s: skip clips shorter than this (too short).
        """
        self.samples: list[tuple[np.ndarray, list[int]]] = []
        self.lengths: list[tuple[int, int]] = []  # (T_frames, L_chars)

        iterator = tqdm(
            zip(audio_paths, lab_paths, strict=True),
            total=len(audio_paths),
            desc="Building ASR dataset",
            unit="file",
            dynamic_ncols=True,
        )

        max_samples = int(max_duration_s * SAMPLE_RATE)
        min_samples = int(min_duration_s * SAMPLE_RATE)

        for audio_path, lab_path in iterator:
            try:
                audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
            except Exception:
                continue

            if len(audio) < min_samples or len(audio) > max_samples:
                continue

            text = lab_path.read_text(encoding="utf-8").strip()
            text = normalize_text(text)
            if not text:
                continue

            pcm = (audio * 32767).astype(np.int16)
            frames = encode_pcm(pcm)
            feats = unpack_frames(frames)  # (T, 48) float32

            char_ids = encode(text)

            # Filter: transcription must fit in frames for CTC
            # CTC requires T >= L (output length >= label length after collapsing)
            # In practice: need T > 2L roughly (blanks between chars)
            if len(feats) < len(char_ids):
                continue

            self.samples.append((feats, char_ids))
            self.lengths.append((len(feats), len(char_ids)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        feats, char_ids = self.samples[idx]
        return (
            torch.from_numpy(feats).float(),
            torch.tensor(char_ids, dtype=torch.long),
            len(feats),
            len(char_ids),
        )

    @property
    def feature_dim(self) -> int:
        return 48

    def stats(self) -> dict:
        """Dataset statistics."""
        frame_lens = [t for t, _ in self.lengths]
        char_lens = [c for _, c in self.lengths]
        return {
            "n_samples": len(self.samples),
            "frames_mean": float(np.mean(frame_lens)),
            "frames_max": int(np.max(frame_lens)),
            "frames_min": int(np.min(frame_lens)),
            "chars_mean": float(np.mean(char_lens)),
            "chars_max": int(np.max(char_lens)),
            "chars_min": int(np.min(char_lens)),
        }


def ctc_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for CTC — pads variable-length sequences.

    Args:
        batch: list of (feats, char_ids, T, L) tuples.

    Returns:
        Tuple:
          - feats_padded: (batch, max_T, 48)
          - chars_padded: (batch, max_L)
          - feat_lens: (batch,) — original T per sample
          - char_lens: (batch,) — original L per sample
    """
    feats_list, chars_list, feat_lens, char_lens = zip(*batch, strict=True)

    max_T = max(feat_lens)
    max_L = max(char_lens)
    batch_size = len(batch)
    feat_dim = feats_list[0].size(-1)

    feats_padded = torch.zeros(batch_size, max_T, feat_dim, dtype=torch.float32)
    chars_padded = torch.zeros(batch_size, max_L, dtype=torch.long)

    for i, (feats, chars) in enumerate(zip(feats_list, chars_list, strict=True)):
        feats_padded[i, : feats.size(0)] = feats
        chars_padded[i, : chars.size(0)] = chars

    return (
        feats_padded,
        chars_padded,
        torch.tensor(feat_lens, dtype=torch.long),
        torch.tensor(char_lens, dtype=torch.long),
    )
