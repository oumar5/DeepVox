"""Evaluation metrics for DeepVox Phase 1.

Metrics:
  - Phone Error Rate (PER)
  - Macro / weighted precision, recall, F1
  - Top-K accuracy
  - Confusion matrix
  - Top-K most confused pairs
  - Accuracy by IPA group (manner, place, voicing)
"""

from __future__ import annotations

import numpy as np

from deepvox.data.preprocess import IDX_TO_PHONEME, NUM_PHONEMES, PHONEME_TO_IDX

# Regroupements phonétiques IPA
# Utile pour comprendre *quelle information* Codec2 préserve/perd.
IPA_GROUPS = {
    "voyelles_orales": ["a", "ɑ", "e", "ɛ", "i", "o", "ɔ", "u", "y", "ø", "œ", "ə"],
    "voyelles_nasales": ["ɑ̃", "ɛ̃", "ɔ̃", "œ̃"],
    "occlusives": ["p", "b", "t", "d", "k", "ɡ", "c", "ɟ"],
    "fricatives": ["f", "v", "s", "z", "ʃ", "ʒ"],
    "affriquees": ["tʃ", "dʒ"],
    "nasales_consonnes": ["m", "mʲ", "n", "ɲ", "ŋ"],
    "liquides_laterales": ["l", "ʎ", "ʁ"],
    "semi_voyelles": ["j", "w", "ɥ"],
    "bruit": ["spn"],
}

# Regroupement par voisement (trait primaire de Codec2 via pitch bits)
VOICING_GROUPS = {
    "voise": (
        # Voyelles (toujours voisées)
        ["a", "ɑ", "e", "ɛ", "i", "o", "ɔ", "u", "y", "ø", "œ", "ə", "ɑ̃", "ɛ̃", "ɔ̃", "œ̃"]
        # Consonnes voisées
        + ["b", "d", "ɡ", "ɟ", "v", "z", "ʒ", "dʒ", "m", "mʲ", "n", "ɲ", "ŋ", "ʎ", "l", "ʁ"]
        # Semi-voyelles
        + ["j", "w", "ɥ"]
    ),
    "non_voise": ["p", "t", "k", "c", "f", "s", "ʃ", "tʃ"],
}


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


def topk_accuracy(topk_preds: np.ndarray, targets: np.ndarray) -> float:
    """Compute Top-K accuracy.

    The true label is considered correctly predicted if it appears anywhere
    in the top-K predictions.

    Args:
        topk_preds: array of shape (n, k) with top-k predicted indices per sample.
        targets: ground-truth indices, shape (n,).

    Returns:
        Top-K accuracy in [0, 1].
    """
    topk_preds = np.asarray(topk_preds)
    targets = np.asarray(targets).reshape(-1, 1)
    correct = (topk_preds == targets).any(axis=1)
    return float(correct.mean())


def precision_recall_f1(
    predictions: np.ndarray,
    targets: np.ndarray,
    average: str = "macro",
) -> tuple[float, float, float]:
    """Compute precision, recall and F1 globally across all classes.

    Args:
        predictions: predicted indices.
        targets: ground-truth indices.
        average: "macro" (unweighted mean) or "weighted" (by class support).

    Returns:
        Tuple (precision, recall, f1) in [0, 1].
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    precisions, recalls, f1s, supports = [], [], [], []
    for c in range(NUM_PHONEMES):
        tp = ((predictions == c) & (targets == c)).sum()
        fp = ((predictions == c) & (targets != c)).sum()
        fn = ((predictions != c) & (targets == c)).sum()

        support = (targets == c).sum()
        if support == 0:
            continue

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(support)

    if not precisions:
        return 0.0, 0.0, 0.0

    if average == "macro":
        return (
            float(np.mean(precisions)),
            float(np.mean(recalls)),
            float(np.mean(f1s)),
        )
    if average == "weighted":
        weights = np.array(supports) / sum(supports)
        return (
            float(np.sum(np.array(precisions) * weights)),
            float(np.sum(np.array(recalls) * weights)),
            float(np.sum(np.array(f1s) * weights)),
        )
    raise ValueError(f"Unknown average mode: {average}")


def accuracy_by_group(
    predictions: np.ndarray,
    targets: np.ndarray,
    groups: dict[str, list[str]],
) -> dict[str, dict]:
    """Compute accuracy within each phonetic group.

    For each group, measures whether the predicted phoneme falls in the same
    group as the target (coarser than exact match).

    Args:
        predictions: predicted indices.
        targets: ground-truth indices.
        groups: dict mapping group name to list of phoneme labels (IPA).

    Returns:
        Dict mapping group name to stats dict:
            - accuracy_exact: fraction predicted exactly right within the group
            - accuracy_group: fraction predicted to same group (coarser)
            - support: number of frames with target in this group
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    results = {}
    for name, phones in groups.items():
        indices = {PHONEME_TO_IDX[p] for p in phones if p in PHONEME_TO_IDX}
        if not indices:
            continue

        target_mask = np.isin(targets, list(indices))
        if target_mask.sum() == 0:
            continue

        exact_correct = (predictions[target_mask] == targets[target_mask]).sum()
        group_correct = np.isin(predictions[target_mask], list(indices)).sum()

        results[name] = {
            "accuracy_exact": float(exact_correct / target_mask.sum()),
            "accuracy_group": float(group_correct / target_mask.sum()),
            "support": int(target_mask.sum()),
        }

    return results


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
    topk_preds: np.ndarray | None = None,
) -> str:
    """Generate a formatted evaluation report.

    Args:
        predictions: predicted phoneme indices (top-1).
        targets: ground-truth phoneme indices.
        topk_preds: optional (n, k) array of top-K predictions for top-K accuracy.

    Returns:
        Markdown-formatted report string.
    """
    per = phone_error_rate(predictions, targets)
    top1_acc = float((predictions == targets).mean())

    p_macro, r_macro, f1_macro = precision_recall_f1(predictions, targets, "macro")
    p_weighted, r_weighted, f1_weighted = precision_recall_f1(predictions, targets, "weighted")

    cm = confusion_matrix(predictions, targets)
    top_conf = top_confused_pairs(cm)
    per_phone = per_phoneme_accuracy(predictions, targets)

    ipa_group_stats = accuracy_by_group(predictions, targets, IPA_GROUPS)
    voicing_stats = accuracy_by_group(predictions, targets, VOICING_GROUPS)

    lines = [
        "# Phase 1 — Evaluation Report",
        "",
        "## Global metrics",
        "",
        f"- **Phone Error Rate (PER):** {per:.4f} ({per * 100:.1f}%)",
        f"- **Top-1 Accuracy:** {top1_acc:.4f} ({top1_acc * 100:.1f}%)",
    ]

    if topk_preds is not None:
        for k in (3, 5):
            if topk_preds.shape[1] >= k:
                acc_k = topk_accuracy(topk_preds[:, :k], targets)
                lines.append(f"- **Top-{k} Accuracy:** {acc_k:.4f} ({acc_k * 100:.1f}%)")

    lines += [
        "",
        "## Precision / Recall / F1",
        "",
        "| Average | Precision | Recall | F1 |",
        "|---------|-----------|--------|-----|",
        f"| Macro   | {p_macro:.3f} | {r_macro:.3f} | {f1_macro:.3f} |",
        f"| Weighted| {p_weighted:.3f} | {r_weighted:.3f} | {f1_weighted:.3f} |",
        "",
        "## Accuracy par groupe IPA",
        "",
        "`accuracy_exact` = phonème exact. `accuracy_group` = phonème du même groupe.",
        "L'écart entre les deux mesure si Codec2 distingue les phonèmes au sein d'un groupe.",
        "",
        "| Groupe | Support | Acc. exact | Acc. groupe |",
        "|--------|---------|------------|-------------|",
    ]

    for group, stats in sorted(ipa_group_stats.items(), key=lambda x: -x[1]["support"]):
        lines.append(
            f"| {group} | {stats['support']} | "
            f"{stats['accuracy_exact']:.3f} | {stats['accuracy_group']:.3f} |"
        )

    lines += [
        "",
        "## Accuracy par voisement",
        "",
        "| Groupe | Support | Acc. exact | Acc. groupe |",
        "|--------|---------|------------|-------------|",
    ]

    for group, stats in voicing_stats.items():
        lines.append(
            f"| {group} | {stats['support']} | "
            f"{stats['accuracy_exact']:.3f} | {stats['accuracy_group']:.3f} |"
        )

    lines += [
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
