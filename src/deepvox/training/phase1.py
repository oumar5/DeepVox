"""Training script for DeepVox Phase 1 — Phoneme classification.

Adam optimizer, lr=1e-3, batch=64, max 50 epochs, early stopping on dev set.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deepvox.data.dataset import Condition, PhonemeDataset
from deepvox.eval.metrics import format_report, phone_error_rate
from deepvox.models.phoneme_classifier import PhonemeClassifier

logger = logging.getLogger(__name__)


def train(
    train_dataset: PhonemeDataset,
    dev_dataset: PhonemeDataset,
    condition: Condition,
    output_dir: str | Path = "outputs/phase1",
    max_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    patience: int = 5,
    device: str | None = None,
) -> PhonemeClassifier:
    """Train the phoneme classifier.

    Args:
        train_dataset: training dataset.
        dev_dataset: validation dataset.
        condition: experimental condition (for logging).
        output_dir: directory to save checkpoints and results.
        max_epochs: maximum training epochs.
        batch_size: batch size.
        lr: learning rate.
        patience: early stopping patience (epochs without improvement).
        device: torch device string.

    Returns:
        Trained PhonemeClassifier.
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

    logger.info("Training condition %s on %s", condition.value, device)
    logger.info("Train samples: %d, Dev samples: %d", len(train_dataset), len(dev_dataset))

    model = PhonemeClassifier(input_dim=train_dataset.feature_dim)
    model = model.to(device)
    logger.info("Model parameters: %d", model.count_parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    best_dev_per = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", dynamic_ncols=True, leave=False)
        for feats, labels in pbar:
            feats = feats.to(device)
            labels = labels.to(device, dtype=torch.long)

            logits = model(feats)  # (batch, seq, num_classes)
            # Dataset returns (2*context+1)-frame windows with the center frame's label.
            # Use only the center frame's logits for classification.
            center = logits.size(1) // 2
            logits_center = logits[:, center, :]  # (batch, num_classes)

            loss = criterion(logits_center, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = logits_center.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.numel()
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{train_correct/train_total:.3f}")

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Evaluate ---
        dev_per, dev_acc = evaluate(model, dev_loader, device)
        scheduler.step(dev_per)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %02d | train_loss=%.4f train_acc=%.3f | dev_PER=%.3f dev_acc=%.3f | lr=%.1e",
            epoch, train_loss, train_acc, dev_per, dev_acc, current_lr,
        )

        # Early stopping
        if dev_per < best_dev_per:
            best_dev_per = dev_per
            epochs_without_improvement = 0
            # Save best checkpoint
            ckpt_path = output_dir / f"best_model_{condition.value}.pt"
            torch.save(model.state_dict(), ckpt_path)
            logger.info("Saved best model (PER=%.4f) → %s", dev_per, ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Load best model
    best_ckpt = output_dir / f"best_model_{condition.value}.pt"
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))

    return model


def evaluate(
    model: PhonemeClassifier,
    data_loader: DataLoader,
    device: str,
) -> tuple[float, float]:
    """Evaluate model on a dataset.

    Returns:
        Tuple of (PER, accuracy).
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for feats, labels in tqdm(data_loader, desc="Evaluating", unit="batch", dynamic_ncols=True, leave=False):
            feats = feats.to(device)
            labels = labels.to(device, dtype=torch.long)

            logits = model(feats)
            center = logits.size(1) // 2
            preds = logits[:, center, :].argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    import numpy as np
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    per = phone_error_rate(all_preds, all_targets)
    acc = (all_preds == all_targets).mean()
    return per, acc


def evaluate_and_report(
    model: PhonemeClassifier,
    test_loader: DataLoader,
    device: str,
    output_path: str | Path = "outputs/phase1/phase1_results.md",
) -> str:
    """Run evaluation on test set and write the report.

    Returns:
        The report as a string.
    """
    import numpy as np

    model.eval()
    all_preds = []
    all_topk = []
    all_targets = []

    k = 5

    with torch.no_grad():
        for feats, labels in tqdm(test_loader, desc="Test", unit="batch", dynamic_ncols=True, leave=False):
            feats = feats.to(device)
            labels = labels.to(device, dtype=torch.long)

            logits = model(feats)
            center = logits.size(1) // 2
            center_logits = logits[:, center, :]  # (batch, num_classes)

            preds = center_logits.argmax(dim=-1)
            topk = center_logits.topk(k=k, dim=-1).indices  # (batch, k)

            all_preds.extend(preds.cpu().numpy())
            all_topk.extend(topk.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_topk = np.array(all_topk)
    all_targets = np.array(all_targets)

    report = format_report(all_preds, all_targets, topk_preds=all_topk)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    logger.info("Report written to %s", output_path)

    return report
