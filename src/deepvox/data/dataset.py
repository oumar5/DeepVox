"""PyTorch Dataset for DeepVox Phase 1 phoneme classification.

Provides (codec2_features, phoneme_id) pairs for training the BiLSTM classifier.
Supports four experimental conditions:
  A — Codec2 raw (48 features)
  B — Codec2 + delta (96 features)
  C — Mel spectrogram baseline (80 features)
  D — Raw PCM baseline (320 features)
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from deepvox.codec2.encoder import (
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    add_delta_features,
    encode_pcm,
    unpack_frames,
)
from deepvox.data.preprocess import (
    NUM_PHONEMES,
    PHONEME_TO_IDX,
    align_phonemes_to_codec2_grid,
    parse_textgrid,
)


class Condition(Enum):
    """Experimental conditions for Phase 1."""
    CODEC2_RAW = "A"       # 48 features
    CODEC2_DELTA = "B"     # 96 features
    MEL_SPEC = "C"         # 80 features
    PCM_RAW = "D"          # 320 features


class PhonemeDataset(Dataset):
    """Dataset of (features, phoneme_id) pairs for phoneme classification.

    Each sample is a single Codec2 frame (40 ms) with its phoneme label.
    """

    def __init__(
        self,
        audio_paths: list[Path],
        textgrid_paths: list[Path],
        condition: Condition = Condition.CODEC2_RAW,
        context_frames: int = 0,
        exclude_silence: bool = True,
    ):
        """
        Args:
            audio_paths: list of 8 kHz WAV file paths.
            textgrid_paths: matching MFA TextGrid paths.
            condition: experimental condition (A, B, C, or D).
            context_frames: number of neighboring frames to include on each side.
            exclude_silence: if True, skip frames labeled as silence.
        """
        self.condition = condition
        self.context_frames = context_frames
        self.features: list[np.ndarray] = []
        self.labels: list[int] = []

        iterator = tqdm(
            zip(audio_paths, textgrid_paths, strict=True),
            total=len(audio_paths),
            desc=f"Building dataset [{condition.value}]",
            unit="file",
            dynamic_ncols=True,
        )
        for audio_path, tg_path in iterator:
            feats, labs = self._process_file(audio_path, tg_path)
            if feats is None:
                continue

            for i, label in enumerate(labs):
                if exclude_silence and label == -1:
                    continue

                # Extract frame with optional context
                start = max(0, i - context_frames)
                end = min(len(feats), i + context_frames + 1)
                frame_feats = feats[start:end]

                # Pad if at boundaries
                expected_len = 2 * context_frames + 1
                if len(frame_feats) < expected_len:
                    pad_width = expected_len - len(frame_feats)
                    if start == 0:
                        frame_feats = np.pad(frame_feats, ((pad_width, 0), (0, 0)))
                    else:
                        frame_feats = np.pad(frame_feats, ((0, pad_width), (0, 0)))

                self.features.append(frame_feats)
                self.labels.append(label)

    def _process_file(
        self, audio_path: Path, tg_path: Path
    ) -> tuple[np.ndarray | None, list[int] | None]:
        """Extract features and labels from one file."""
        try:
            audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        except Exception:
            return None, None

        pcm = (audio * 32767).astype(np.int16)
        n_frames = len(pcm) // SAMPLES_PER_FRAME

        if n_frames == 0:
            return None, None

        # Get phoneme labels
        intervals = parse_textgrid(tg_path)
        phonemes = align_phonemes_to_codec2_grid(intervals, n_frames)
        labels = [PHONEME_TO_IDX.get(p, -1) for p in phonemes]

        # Extract features based on condition
        if self.condition in (Condition.CODEC2_RAW, Condition.CODEC2_DELTA):
            frames = encode_pcm(pcm)
            feats = unpack_frames(frames)
            if self.condition == Condition.CODEC2_DELTA:
                feats = add_delta_features(feats)

        elif self.condition == Condition.MEL_SPEC:
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=SAMPLE_RATE,
                n_mels=80,
                hop_length=SAMPLES_PER_FRAME,
                n_fft=1024,
                win_length=SAMPLES_PER_FRAME,
            )
            feats = librosa.power_to_db(mel, ref=np.max).T  # (n_frames, 80)
            feats = feats[:n_frames]

        elif self.condition == Condition.PCM_RAW:
            pcm_float = audio[: n_frames * SAMPLES_PER_FRAME]
            feats = pcm_float.reshape(n_frames, SAMPLES_PER_FRAME)  # (n_frames, 320)

        else:
            raise ValueError(f"Unknown condition: {self.condition}")

        return feats, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        feats = torch.from_numpy(self.features[idx]).float()
        label = self.labels[idx]
        return feats, label

    @property
    def feature_dim(self) -> int:
        """Dimension of features per frame."""
        if self.condition == Condition.CODEC2_RAW:
            return 48
        elif self.condition == Condition.CODEC2_DELTA:
            return 96
        elif self.condition == Condition.MEL_SPEC:
            return 80
        elif self.condition == Condition.PCM_RAW:
            return SAMPLES_PER_FRAME  # 320
        raise ValueError(f"Unknown condition: {self.condition}")

    @property
    def num_classes(self) -> int:
        return NUM_PHONEMES
