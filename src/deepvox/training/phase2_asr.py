"""Training script for DeepVox Phase 2 — ASR (Codec2 → French text).

Architecture: BiLSTM + CTC
Loss: CTCLoss (PyTorch)
Decoder: greedy (argmax + CTC collapse)
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deepvox.data.ctc_dataset import ASRDataset, ctc_collate_fn
from deepvox.data.text import BLANK_IDX, decode, decode_ctc
from deepvox.eval.wer import cer, format_asr_report, wer
from deepvox.models.ctc_asr import CTCASR

logger = logging.getLogger(__name__)


def train(
    train_dataset: ASRDataset,
    dev_dataset: ASRDataset,
    output_dir: str | Path = "outputs/phase2",
    max_epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    patience: int = 5,
    num_workers: int = 4,
    device: str | None = None,
) -> CTCASR:
    """Train the ASR model.

    Args:
        train_dataset: training dataset.
        dev_dataset: validation dataset.
        output_dir: where to save checkpoints.
        max_epochs: maximum training epochs.
        batch_size: batch size (lower than phonemes because variable length).
        lr: learning rate.
        weight_decay: AdamW weight decay.
        patience: early stopping patience.
        num_workers: DataLoader workers.
        device: torch device.

    Returns:
        Trained CTCASR model.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    logger.info("Training ASR on %s", device)
    logger.info("Train samples: %d, Dev samples: %d", len(train_dataset), len(dev_dataset))

    model = CTCASR()
    model = model.to(device)
    logger.info("Model parameters: %d", model.count_parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # CTC loss (blank=0, zero_infinity handles edge cases)
    criterion = nn.CTCLoss(blank=BLANK_IDX, reduction="mean", zero_infinity=True)

    pin_memory = device != "cpu"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    best_dev_cer = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )
        for feats, chars, feat_lens, char_lens in pbar:
            feats = feats.to(device)
            chars = chars.to(device)
            feat_lens = feat_lens.to(device)
            char_lens = char_lens.to(device)

            log_probs = model(feats)  # (B, T, V)
            # CTC expects (T, B, V)
            log_probs_t = log_probs.transpose(0, 1)

            loss = criterion(log_probs_t, chars, feat_lens, char_lens)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for RNN stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.3f}", avg=f"{train_loss / n_batches:.3f}")

        train_loss /= max(n_batches, 1)

        # --- Evaluate ---
        dev_wer, dev_cer = evaluate(model, dev_loader, device)
        scheduler.step(dev_cer)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %02d | train_loss=%.4f | dev_WER=%.3f dev_CER=%.3f | lr=%.1e",
            epoch, train_loss, dev_wer, dev_cer, current_lr,
        )

        # Early stopping (on CER, more stable than WER)
        if dev_cer < best_dev_cer:
            best_dev_cer = dev_cer
            epochs_without_improvement = 0
            ckpt_path = output_dir / "best_model_asr.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger.info("Saved best model (CER=%.4f) → %s", dev_cer, ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Load best
    model.load_state_dict(
        torch.load(output_dir / "best_model_asr.pt", map_location=device, weights_only=True)
    )
    return model


def evaluate(
    model: CTCASR,
    data_loader: DataLoader,
    device: str,
) -> tuple[float, float]:
    """Evaluate model on dataset.

    Returns:
        Tuple (WER, CER) in [0, inf).
    """
    model.eval()
    all_refs = []
    all_hyps = []

    with torch.no_grad():
        for feats, chars, feat_lens, char_lens in tqdm(
            data_loader, desc="Evaluating", unit="batch", dynamic_ncols=True, leave=False
        ):
            feats = feats.to(device)
            preds = model.greedy_decode(feats)  # list of list of int

            for i, pred_ids in enumerate(preds):
                # Trim to actual feat_len (padding)
                pred_ids = pred_ids[: feat_lens[i].item()]
                hyp_text = decode_ctc(pred_ids)

                # Reference: trim to actual char_len
                ref_ids = chars[i, : char_lens[i].item()].cpu().tolist()
                ref_text = decode(ref_ids)

                all_hyps.append(hyp_text)
                all_refs.append(ref_text)

    return wer(all_refs, all_hyps), cer(all_refs, all_hyps)


def evaluate_and_report(
    model: CTCASR,
    test_loader: DataLoader,
    device: str,
    output_path: str | Path = "outputs/phase2/phase2_results.md",
) -> str:
    """Run evaluation on test set and write a markdown report."""
    model.eval()
    all_refs = []
    all_hyps = []

    with torch.no_grad():
        for feats, chars, feat_lens, char_lens in tqdm(
            test_loader, desc="Test", unit="batch", dynamic_ncols=True, leave=False
        ):
            feats = feats.to(device)
            preds = model.greedy_decode(feats)

            for i, pred_ids in enumerate(preds):
                pred_ids = pred_ids[: feat_lens[i].item()]
                hyp_text = decode_ctc(pred_ids)

                ref_ids = chars[i, : char_lens[i].item()].cpu().tolist()
                ref_text = decode(ref_ids)

                all_hyps.append(hyp_text)
                all_refs.append(ref_text)

    report = format_asr_report(all_refs, all_hyps)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info("Report written to %s", output_path)

    return report
