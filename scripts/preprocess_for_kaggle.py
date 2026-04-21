#!/usr/bin/env python3
"""Preprocess Common Voice FR → pickle pour Kaggle.

Convertit les MP3 en features Codec2 + char_ids et sauvegarde
en pickle pour éviter de refaire le preprocessing sur Kaggle.

Usage:
    python scripts/preprocess_for_kaggle.py --max-samples 586000

Output:
    data/deepvox_586k.pkl  (~2-3 GB)
    data/deepvox_586k.zip  (compressé pour upload Kaggle)
"""

import argparse
import pickle
import sys
import zipfile
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepvox.codec2.encoder import SAMPLE_RATE, encode_pcm, unpack_frames
from deepvox.data.text import encode, normalize_text


def process_sample(mp3_path, sentence, max_duration_s=12.0):
    """MP3 → (Codec2 features, char_ids, text) ou None si filtré."""
    try:
        audio, _ = librosa.load(str(mp3_path), sr=SAMPLE_RATE, mono=True)
    except Exception:
        return None

    if len(audio) / SAMPLE_RATE > max_duration_s or len(audio) / SAMPLE_RATE < 0.5:
        return None

    text = normalize_text(sentence)
    if not text:
        return None

    pcm = (audio * 32767).astype(np.int16)
    frames = encode_pcm(pcm)
    feats = unpack_frames(frames)
    char_ids = encode(text)

    if len(feats) < len(char_ids):
        return None

    return feats, char_ids, text


def main():
    parser = argparse.ArgumentParser(description="Preprocess Common Voice FR for Kaggle")
    parser.add_argument("--max-samples", type=int, default=586_000)
    parser.add_argument("--data-dir", type=str, default="data/cv-fr")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Charger le TSV
    tsv_path = data_dir / "train.tsv"
    print(f"Loading {tsv_path}...")
    df = pd.read_csv(tsv_path, sep="\t", usecols=["path", "sentence"])
    print(f"Entrées train: {len(df):,}")

    # Indexer les MP3
    clips_dir = data_dir / "clips"
    print(f"Clips dir: {clips_dir}")

    # Preprocessing
    samples = []
    skipped = 0

    for _, row in tqdm(df.iterrows(), total=min(len(df), args.max_samples * 3), desc="Processing"):
        if len(samples) >= args.max_samples:
            break

        mp3_path = clips_dir / row["path"]
        if not mp3_path.exists():
            skipped += 1
            continue

        result = process_sample(mp3_path, row["sentence"])
        if result is None:
            skipped += 1
            continue

        samples.append(result)

    n = len(samples)
    print(f"\nSamples valides : {n:,}")
    print(f"Skipped : {skipped:,}")

    # Stats
    frame_lens = [len(s[0]) for s in samples]
    char_lens = [len(s[1]) for s in samples]
    print(f"Frames/sample : min={min(frame_lens)}, max={max(frame_lens)}, mean={np.mean(frame_lens):.0f}")
    print(f"Chars/sample : min={min(char_lens)}, max={max(char_lens)}, mean={np.mean(char_lens):.0f}")

    # Sauvegarder en pickle
    label = f"{n // 1000}k" if n >= 1000 else str(n)
    pkl_path = output_dir / f"deepvox_{label}.pkl"
    print(f"\nSaving to {pkl_path}...")
    with open(pkl_path, "wb") as f:
        pickle.dump(samples, f, protocol=4)

    pkl_size = pkl_path.stat().st_size / 1e9
    print(f"Pickle: {pkl_size:.2f} GB")

    # Compresser en zip
    zip_path = output_dir / f"deepvox_{label}.zip"
    print(f"Compressing to {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pkl_path, pkl_path.name)

    zip_size = zip_path.stat().st_size / 1e9
    print(f"Zip: {zip_size:.2f} GB")

    print(f"\n=== DONE ===")
    print(f"1. Upload {zip_path} comme dataset Kaggle 'deepvox-preprocessed'")
    print(f'2. Dans le notebook: PREPROCESSED_PATH = "/kaggle/input/deepvox-preprocessed/deepvox_{label}.pkl"')


if __name__ == "__main__":
    main()
