"""Data preprocessing pipeline for DeepVox Phase 1.

Pipeline:
  1. Resample audio 16 kHz → 8 kHz mono
  2. Run Montreal Forced Aligner (MFA) to get phoneme alignments
  3. Encode audio with Codec2 1200 bps
  4. Map MFA phoneme boundaries to Codec2 40 ms grid
  5. Produce (frame, phoneme) pairs
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from deepvox.codec2.encoder import (
    FRAME_DURATION_MS,
    SAMPLE_RATE,
    encode_pcm,
)

# Phonèmes du français produits par Montreal Forced Aligner (modèle french_mfa)
# Alphabet : IPA (Unicode). 44 phonèmes + token spécial silence/bruit.
FRENCH_PHONEMES = [
    # Voyelles orales (12)
    "a", "ɑ", "e", "ɛ", "i", "o", "ɔ", "u", "y", "ø", "œ", "ə",
    # Voyelles nasales (4)
    "ɑ̃", "ɛ̃", "ɔ̃", "œ̃",
    # Consonnes occlusives (8)
    "p", "b", "t", "d", "k", "ɡ", "c", "ɟ",
    # Fricatives (6)
    "f", "v", "s", "z", "ʃ", "ʒ",
    # Affriquées (2)
    "tʃ", "dʒ",
    # Nasales + autres (6)
    "m", "mʲ", "n", "ɲ", "ŋ", "ʎ",
    # Liquides (2)
    "l", "ʁ",
    # Semi-voyelles (3)
    "j", "w", "ɥ",
    # Bruit / non-parole (1)
    "spn",
]

PHONEME_TO_IDX = {p: i for i, p in enumerate(FRENCH_PHONEMES)}
IDX_TO_PHONEME = {i: p for i, p in enumerate(FRENCH_PHONEMES)}
NUM_PHONEMES = len(FRENCH_PHONEMES)

# Special token for silence (empty intervals in MFA output)
SIL_TOKEN = ""


def resample_audio(input_path: str | Path, output_path: str | Path) -> Path:
    """Resample audio file to 8 kHz 16-bit mono WAV.

    Args:
        input_path: source audio file (MP3, WAV, FLAC, etc.)
        output_path: destination WAV path.

    Returns:
        Path to the resampled file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audio, _ = librosa.load(str(input_path), sr=SAMPLE_RATE, mono=True)
    pcm = (audio * 32767).astype(np.int16)
    sf.write(str(output_path), pcm, SAMPLE_RATE, subtype="PCM_16")
    return output_path


def parse_textgrid(textgrid_path: str | Path) -> list[tuple[float, float, str]]:
    """Parse a TextGrid file to extract phoneme intervals.

    Uses praatio for robust TextGrid parsing. Looks for a tier named 'phones'
    (MFA default). Empty intervals (silence) are skipped.

    Args:
        textgrid_path: path to the .TextGrid file.

    Returns:
        List of (start_sec, end_sec, phoneme_label).
    """
    from praatio import textgrid as tgio

    tg = tgio.openTextgrid(str(textgrid_path), includeEmptyIntervals=False)

    # Find the phones tier
    tier_name = None
    for name in tg.tierNames:
        if name.lower() in ("phones", "phone"):
            tier_name = name
            break

    if tier_name is None:
        return []

    tier = tg.getTier(tier_name)
    return [(entry.start, entry.end, entry.label) for entry in tier.entries if entry.label.strip()]


def align_phonemes_to_codec2_grid(
    intervals: list[tuple[float, float, str]],
    n_frames: int,
) -> list[str]:
    """Map phoneme intervals to the Codec2 40 ms frame grid.

    For each Codec2 frame, assigns the phoneme that covers the majority
    of that frame's time window.

    Args:
        intervals: list of (start_sec, end_sec, phoneme_label) from MFA.
        n_frames: number of Codec2 frames in the utterance.

    Returns:
        List of phoneme labels, one per frame (length = n_frames).
    """
    frame_dur = FRAME_DURATION_MS / 1000.0  # 0.04 s
    labels = []

    for frame_idx in range(n_frames):
        frame_start = frame_idx * frame_dur
        frame_end = frame_start + frame_dur
        frame_mid = (frame_start + frame_end) / 2.0

        best_label = SIL_TOKEN
        for start, end, label in intervals:
            if start <= frame_mid < end:
                best_label = label
                break

        labels.append(best_label)

    return labels


def run_mfa(
    corpus_dir: str | Path,
    output_dir: str | Path,
    dictionary: str = "french_mfa",
    acoustic_model: str = "french_mfa",
    num_jobs: int = 4,
) -> Path:
    """Run Montreal Forced Aligner on a corpus directory.

    Expects corpus_dir to contain WAV files and matching .lab or .txt files.

    Args:
        corpus_dir: directory with audio + transcription files.
        output_dir: where to write TextGrid files.
        dictionary: MFA dictionary name or path.
        acoustic_model: MFA acoustic model name or path.
        num_jobs: number of parallel workers.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "mfa", "align",
        str(corpus_dir),
        dictionary,
        acoustic_model,
        str(output_dir),
        "--num_jobs", str(num_jobs),
        "--clean",
        "--overwrite",
    ]

    subprocess.run(cmd, check=True)
    return output_dir


def process_utterance(
    audio_path: str | Path,
    textgrid_path: str | Path,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Process a single utterance: encode + align.

    Args:
        audio_path: path to 8 kHz WAV file.
        textgrid_path: path to MFA TextGrid file.

    Returns:
        Tuple of:
          - frames: uint8 array (n_frames, 6) — raw Codec2 frames
          - phonemes: list of phoneme labels (length n_frames)
          - phoneme_ids: list of phoneme indices (length n_frames), -1 for unknown/sil
    """
    audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    pcm = (audio * 32767).astype(np.int16)

    frames = encode_pcm(pcm)
    n_frames = len(frames)

    intervals = parse_textgrid(textgrid_path)
    phonemes = align_phonemes_to_codec2_grid(intervals, n_frames)

    phoneme_ids = [PHONEME_TO_IDX.get(p, -1) for p in phonemes]

    return frames, phonemes, phoneme_ids


def prepare_common_voice_corpus(
    cv_tsv_path: str | Path,
    clips_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Prepare Common Voice data for MFA alignment.

    Creates a directory structure with WAV files (resampled to 8 kHz)
    and .lab transcription files, ready for MFA.

    Args:
        cv_tsv_path: path to validated.tsv (Common Voice metadata).
        clips_dir: directory containing MP3 clips.
        output_dir: where to write WAV + .lab files.

    Returns:
        Path to the prepared corpus directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = Path(clips_dir)

    with open(cv_tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mp3_name = row["path"]
            sentence = row["sentence"]

            mp3_path = clips_dir / mp3_name
            if not mp3_path.exists():
                continue

            stem = mp3_path.stem
            wav_path = output_dir / f"{stem}.wav"
            lab_path = output_dir / f"{stem}.lab"

            resample_audio(mp3_path, wav_path)
            lab_path.write_text(sentence, encoding="utf-8")

    return output_dir
