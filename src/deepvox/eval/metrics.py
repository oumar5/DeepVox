"""Evaluation metrics for DeepVox Phase 1.

Metrics:
  - Phone Error Rate (PER)
  - Macro precision (average per phoneme)
  - Confusion matrix (36x36)
  - Top-K most confused pairs
"""

from __future__ import annotations

import numpy as np

from deepvox.data.preprocess import IDX_TO_PHONEME, NUM_PHONEMES


def phone_error_rate(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Phone Error Rate (frame-level).

    PER = 1 - (correctly classified frames / total frames)

    Args:
        predictions: predicted phoneme indices, shape (n,).
        targets: ground-truth phoneme indices, shape (n,).

    Returns:
        PER as a float in [0, 1].
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    correct = (predictions == targets).sum()
    return 1.0 - correct / len(targets)


def macro_precision(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute macro-averaged precision across all phonemes.

    Args:
        predictions: predicted phoneme indices.
        targets: ground-truth phoneme indices.

    Returns:
        Macro precision as a float in [0, 1].
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    precisions = []
    for c in range(NUM_PHONEMES):
        pred_mask = predictions == c
        if pred_mask.sum() == 0:
            continue
        true_positives = ((predictions == c) & (targets == c)).sum()
        precisions.append(true_positives / pred_mask.sum())

    return float(np.mean(precisions)) if precisions else 0.0


def confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        predictions: predicted phoneme indices.
        targets: ground-truth phoneme indices.

    Returns:
        int array of shape (NUM_PHONEMES, NUM_PHONEMES).
        Row = true class, column = predicted class.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    cm = np.zeros((NUM_PHONEMES, NUM_PHONEMES), dtype=np.int64)
    for t, p in zip(targets, predictions, strict=True):
        cm[t, p] += 1
    return cm


def top_confused_pairs(
    cm: np.ndarray, k: int = 10
) -> list[tuple[str, str, int]]:
    """Find the top-K most confused phoneme pairs from the confusion matrix.

    Args:
        cm: confusion matrix (NUM_PHONEMES x NUM_PHONEMES).
        k: number of top pairs to return.

    Returns:
        List of (true_phoneme, predicted_phoneme, count), sorted descending.
    """
    pairs = []
    for i in range(NUM_PHONEMES):
        for j in range(NUM_PHONEMES):
            if i != j and cm[i, j] > 0:
                pairs.append((IDX_TO_PHONEME[i], IDX_TO_PHONEME[j], int(cm[i, j])))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]


def per_phoneme_accuracy(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Compute accuracy for each phoneme class.

    Args:
        predictions: predicted phoneme indices.
        targets: ground-truth phoneme indices.

    Returns:
        Dict mapping phoneme label to accuracy.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    accuracies = {}
    for c in range(NUM_PHONEMES):
        mask = targets == c
        if mask.sum() == 0:
            continue
        acc = (predictions[mask] == c).sum() / mask.sum()
        accuracies[IDX_TO_PHONEME[c]] = float(acc)

    return accuracies


def format_report(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> str:
    """Generate a formatted evaluation report.

    Args:
        predictions: predicted phoneme indices.
        targets: ground-truth phoneme indices.

    Returns:
        Markdown-formatted report string.
    """
    per = phone_error_rate(predictions, targets)
    mp = macro_precision(predictions, targets)
    cm = confusion_matrix(predictions, targets)
    top_conf = top_confused_pairs(cm)
    per_phone = per_phoneme_accuracy(predictions, targets)

    lines = [
        "# Phase 1 — Evaluation Report",
        "",
        f"**Phone Error Rate (PER):** {per:.4f} ({per * 100:.1f}%)",
        f"**Macro Precision:** {mp:.4f} ({mp * 100:.1f}%)",
        "",
        "## Top 10 Confused Pairs",
        "",
        "| True | Predicted | Count |",
        "|------|-----------|-------|",
    ]

    for true_p, pred_p, count in top_conf:
        lines.append(f"| {true_p} | {pred_p} | {count} |")

    lines += [
        "",
        "## Per-Phoneme Accuracy",
        "",
        "| Phoneme | Accuracy |",
        "|---------|----------|",
    ]

    for phoneme, acc in sorted(per_phone.items()):
        lines.append(f"| {phoneme} | {acc:.3f} |")

    # GO / NO-GO decision
    lines += [
        "",
        "## Decision",
        "",
    ]
    if per <= 0.15:
        lines.append(f"**GO** — PER = {per*100:.1f}% (≤15%). Hypothèse validée → Phase 2.")
    elif per <= 0.25:
        lines.append(f"**GO conditionnel** — PER = {per*100:.1f}% (15-25%). Information dégradée mais utilisable.")
    else:
        lines.append(f"**NO-GO** — PER = {per*100:.1f}% (>25%). Tester delta features, Codec2 700C, MFCC hybride.")

    return "\n".join(lines)
