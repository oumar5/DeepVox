#!/usr/bin/env python3
"""Prepare Common Voice French data for DeepVox Phase 1.

Steps:
  1. Read Common Voice TSV metadata
  2. Resample MP3 → WAV 8 kHz mono
  3. Create .lab files for MFA
  4. Run MFA alignment
  5. Output: WAV + TextGrid files ready for training

Usage:
    python scripts/prepare_data.py \
        --cv-dir data/common-voice-french \
        --output-dir data/prepared \
        --mfa-output-dir data/mfa-output \
        --max-samples 5000
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from deepvox.data.preprocess import resample_audio, run_mfa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_cv_files(cv_dir: Path) -> tuple[Path, Path]:
    """Find the TSV and clips directory in a Common Voice download."""
    # Look for validated.tsv or train.tsv
    tsv_candidates = list(cv_dir.rglob("validated.tsv")) + list(cv_dir.rglob("train.tsv"))
    if not tsv_candidates:
        raise FileNotFoundError(f"No validated.tsv or train.tsv found in {cv_dir}")

    tsv_path = tsv_candidates[0]

    # Look for clips directory
    clips_candidates = list(cv_dir.rglob("clips"))
    if not clips_candidates:
        raise FileNotFoundError(f"No clips directory found in {cv_dir}")

    clips_dir = clips_candidates[0]

    return tsv_path, clips_dir


def prepare_subset(
    tsv_path: Path,
    clips_dir: Path,
    output_dir: Path,
    max_samples: int | None = None,
) -> int:
    """Resample and create .lab files for a subset of Common Voice."""
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if max_samples and count >= max_samples:
                break

            mp3_name = row.get("path", "")
            sentence = row.get("sentence", "")

            if not mp3_name or not sentence:
                continue

            mp3_path = clips_dir / mp3_name
            if not mp3_path.exists():
                continue

            stem = mp3_path.stem
            wav_path = output_dir / f"{stem}.wav"
            lab_path = output_dir / f"{stem}.lab"

            try:
                resample_audio(mp3_path, wav_path)
                lab_path.write_text(sentence, encoding="utf-8")
                count += 1

                if count % 500 == 0:
                    logger.info("Processed %d files...", count)
            except Exception as e:
                logger.warning("Failed to process %s: %s", mp3_name, e)
                continue

    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare Common Voice FR for DeepVox")
    parser.add_argument("--cv-dir", type=Path, required=True,
                        help="Common Voice download directory")
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared"),
                        help="Output directory for WAV + .lab files")
    parser.add_argument("--mfa-output-dir", type=Path, default=Path("data/mfa-output"),
                        help="Output directory for MFA TextGrid files")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--skip-mfa", action="store_true",
                        help="Skip MFA alignment step")
    parser.add_argument("--mfa-jobs", type=int, default=4,
                        help="Number of MFA parallel workers")
    args = parser.parse_args()

    # Step 1: Find CV files
    logger.info("Looking for Common Voice files in %s", args.cv_dir)
    tsv_path, clips_dir = find_cv_files(args.cv_dir)
    logger.info("TSV: %s", tsv_path)
    logger.info("Clips: %s", clips_dir)

    # Step 2: Resample + create .lab files
    logger.info("Resampling to 8 kHz and creating .lab files...")
    count = prepare_subset(tsv_path, clips_dir, args.output_dir, args.max_samples)
    logger.info("Prepared %d files → %s", count, args.output_dir)

    # Step 3: Run MFA
    if not args.skip_mfa:
        logger.info("Running MFA alignment...")
        logger.info("  corpus: %s", args.output_dir)
        logger.info("  output: %s", args.mfa_output_dir)
        logger.info("  jobs:   %d", args.mfa_jobs)

        try:
            run_mfa(
                corpus_dir=args.output_dir,
                output_dir=args.mfa_output_dir,
                num_jobs=args.mfa_jobs,
            )
            logger.info("MFA alignment complete → %s", args.mfa_output_dir)
        except Exception as e:
            logger.error("MFA failed: %s", e)
            logger.info("You can run MFA manually:")
            logger.info("  conda activate mfa")
            logger.info("  mfa align %s french_mfa french_mfa %s --num_jobs %d --clean",
                        args.output_dir, args.mfa_output_dir, args.mfa_jobs)
    else:
        logger.info("Skipping MFA (--skip-mfa). Run it manually:")
        logger.info("  conda activate mfa")
        logger.info("  mfa align %s french_mfa french_mfa %s --num_jobs %d --clean",
                    args.output_dir, args.mfa_output_dir, args.mfa_jobs)

    logger.info("Done. Next step: run phase1_phoneme_classification.py")


if __name__ == "__main__":
    main()
