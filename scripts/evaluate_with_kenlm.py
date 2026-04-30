#!/usr/bin/env python3
"""Évaluation locale du modèle DeepVox avec et sans KenLM.

Utilise les données préprocessées (deepvox_586k.pkl) ou un échantillon plus petit.

Usage:
    python scripts/evaluate_with_kenlm.py \\
        --model checkpoints/best_asr_run4.pt \\
        --data data/deepvox_586k.pkl \\
        --lm models/fr_kenlm.bin \\
        --max-test 1000

Sans LM (greedy seul) :
    python scripts/evaluate_with_kenlm.py --model ... --data ...
"""

import argparse
import pickle
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from deepvox.data.text import (
    BLANK_IDX,
    UNK_IDX,
    VOCAB,
    decode,
    decode_ctc,
)
from deepvox.eval.wer import cer, wer
from deepvox.models.ctc_asr import CTCASR


class ASRDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, char_ids, _ = self.samples[idx]
        return (
            torch.from_numpy(feats).float(),
            torch.tensor(char_ids, dtype=torch.long),
            len(feats),
            len(char_ids),
        )


def ctc_collate(batch):
    feats, chars, f_lens, c_lens = zip(*batch, strict=False)
    max_T = max(f_lens)
    max_L = max(c_lens)
    B = len(batch)
    feats_pad = torch.zeros(B, max_T, 48)
    chars_pad = torch.zeros(B, max_L, dtype=torch.long)
    for i in range(B):
        feats_pad[i, : feats[i].size(0)] = feats[i]
        chars_pad[i, : chars[i].size(0)] = chars[i]
    return feats_pad, chars_pad, torch.tensor(f_lens), torch.tensor(c_lens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Chemin vers best_asr.pt")
    parser.add_argument("--data", required=True, help="Pickle de samples")
    parser.add_argument("--lm", default=None, help="Chemin vers KenLM .bin (optionnel)")
    parser.add_argument("--max-test", type=int, default=1000, help="Nb samples test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--beam-width", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Charger samples
    print(f"Loading {args.data}...")
    with open(args.data, "rb") as f:
        samples = pickle.load(f)
    random.seed(42)
    random.shuffle(samples)
    n = len(samples)
    n_test = int(n * 0.05)
    test_samples = samples[n - n_test : n - n_test + args.max_test]
    print(f"Test samples: {len(test_samples)}")
    del samples

    test_ds = ASRDataset(test_samples)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=ctc_collate, num_workers=0,
    )

    # Charger le modèle
    print(f"Loading model {args.model}...")
    model = CTCASR().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # ========== GREEDY ==========
    print("\n=== Évaluation GREEDY ===")
    all_refs = []
    all_hyps_greedy = []
    all_logprobs = []

    t0 = time.time()
    with torch.no_grad():
        for feats, chars, f_lens, c_lens in test_loader:
            feats = feats.to(device)
            log_probs = model(feats)  # (B, T, V)
            preds = log_probs.argmax(dim=-1)
            for i in range(feats.size(0)):
                T = f_lens[i].item()
                L = c_lens[i].item()
                all_logprobs.append(log_probs[i, :T].cpu().numpy())
                all_hyps_greedy.append(decode_ctc(preds[i, :T].tolist()))
                all_refs.append(decode(chars[i, :L].tolist()))

    dt_greedy = time.time() - t0
    wer_g = wer(all_refs, all_hyps_greedy)
    cer_g = cer(all_refs, all_hyps_greedy)
    print(f"WER: {wer_g:.4f} ({wer_g*100:.1f}%)")
    print(f"CER: {cer_g:.4f} ({cer_g*100:.1f}%)")
    print(f"Time: {dt_greedy:.1f}s ({dt_greedy/len(test_samples)*1000:.1f} ms/sample)")

    # ========== BEAM SEARCH + KenLM ==========
    if args.lm:
        try:
            from pyctcdecode import build_ctcdecoder
        except ImportError:
            print("\npyctcdecode not installed. Run: pip install pyctcdecode pypi-kenlm")
            return

        print(f"\n=== Évaluation BEAM SEARCH + KenLM ({args.lm}) ===")
        labels = list(VOCAB)
        labels[BLANK_IDX] = ""           # blank → chaîne vide
        labels[UNK_IDX] = "⁇"            # UNK → caractère unique pour éviter doublon
        assert len(labels) == len(set(labels)), "Duplicate labels"

        decoder = build_ctcdecoder(
            labels=labels,
            kenlm_model_path=args.lm,
            alpha=args.alpha,
            beta=args.beta,
        )

        all_hyps_lm = []
        t0 = time.time()
        for logp in all_logprobs:
            hyp = decoder.decode(logp, beam_width=args.beam_width)
            all_hyps_lm.append(hyp)
        dt_lm = time.time() - t0

        wer_lm = wer(all_refs, all_hyps_lm)
        cer_lm = cer(all_refs, all_hyps_lm)
        print(f"WER: {wer_lm:.4f} ({wer_lm*100:.1f}%)")
        print(f"CER: {cer_lm:.4f} ({cer_lm*100:.1f}%)")
        print(f"Time: {dt_lm:.1f}s ({dt_lm/len(test_samples)*1000:.1f} ms/sample)")

        print("\n=== Comparaison ===")
        print(f"{'':12} | Greedy  | +KenLM  | Δ")
        print(f"{'WER':12} | {wer_g*100:5.1f}%  | {wer_lm*100:5.1f}%  | {(wer_lm-wer_g)*100:+.1f} pp")
        print(f"{'CER':12} | {cer_g*100:5.1f}%  | {cer_lm*100:5.1f}%  | {(cer_lm-cer_g)*100:+.1f} pp")
    else:
        print("\n(Pas de LM fourni, --lm pour ajouter beam search KenLM)")
        all_hyps_lm = None

    # Exemples
    print("\n=== Exemples ===")
    indices = random.sample(range(len(all_refs)), min(10, len(all_refs)))
    for idx in indices:
        print(f"REF:    {all_refs[idx]}")
        print(f"GREEDY: {all_hyps_greedy[idx]}")
        if all_hyps_lm:
            print(f"+LM:    {all_hyps_lm[idx]}")
        print()


if __name__ == "__main__":
    main()
