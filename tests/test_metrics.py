"""Tests for deepvox.eval.metrics module."""

import numpy as np
import pytest

from deepvox.data.preprocess import NUM_PHONEMES
from deepvox.eval.metrics import (
    IPA_GROUPS,
    VOICING_GROUPS,
    accuracy_by_group,
    confusion_matrix,
    format_report,
    phone_error_rate,
    precision_recall_f1,
    top_confused_pairs,
    topk_accuracy,
)


class TestPER:
    def test_perfect(self):
        targets = np.array([0, 1, 2, 3, 4])
        per = phone_error_rate(targets, targets)
        assert per == 0.0

    def test_all_wrong(self):
        preds = np.array([1, 2, 3, 4, 0])
        targets = np.array([0, 1, 2, 3, 4])
        per = phone_error_rate(preds, targets)
        assert per == 1.0

    def test_partial(self):
        preds = np.array([0, 1, 2, 0, 0])
        targets = np.array([0, 1, 2, 3, 4])
        per = phone_error_rate(preds, targets)
        assert per == pytest.approx(0.4)


class TestConfusionMatrix:
    def test_shape(self):
        preds = np.array([0, 1, 2])
        targets = np.array([0, 1, 2])
        cm = confusion_matrix(preds, targets)
        assert cm.shape == (NUM_PHONEMES, NUM_PHONEMES)

    def test_diagonal(self):
        targets = np.array([0, 0, 1, 1, 2])
        cm = confusion_matrix(targets, targets)
        assert cm[0, 0] == 2
        assert cm[1, 1] == 2
        assert cm[2, 2] == 1


class TestTopConfused:
    def test_returns_pairs(self):
        preds = np.array([0, 0, 1, 1, 0])
        targets = np.array([0, 1, 1, 0, 0])
        cm = confusion_matrix(preds, targets)
        pairs = top_confused_pairs(cm, k=5)
        assert len(pairs) > 0
        assert len(pairs[0]) == 3  # (true, pred, count)


class TestTopK:
    def test_top1_matches_exact(self):
        topk = np.array([[0], [1], [2]])
        targets = np.array([0, 1, 2])
        assert topk_accuracy(topk, targets) == 1.0

    def test_top3_more_lenient(self):
        topk = np.array([[5, 0, 7], [5, 1, 7], [5, 2, 7]])
        targets = np.array([0, 1, 2])
        # Exact match (position 0) would fail, but target is in top-3
        assert topk_accuracy(topk, targets) == 1.0

    def test_top_miss(self):
        topk = np.array([[5, 6, 7]])
        targets = np.array([0])
        assert topk_accuracy(topk, targets) == 0.0


class TestPrecisionRecallF1:
    def test_perfect(self):
        preds = np.array([0, 1, 2])
        targets = np.array([0, 1, 2])
        p, r, f1 = precision_recall_f1(preds, targets, "macro")
        assert p == pytest.approx(1.0)
        assert r == pytest.approx(1.0)
        assert f1 == pytest.approx(1.0)

    def test_weighted_vs_macro_differ_on_imbalance(self):
        # 10 samples class 0 (perfect), 1 sample class 1 (wrong)
        preds = np.array([0] * 10 + [0])
        targets = np.array([0] * 10 + [1])
        _, _, f1_macro = precision_recall_f1(preds, targets, "macro")
        _, _, f1_weighted = precision_recall_f1(preds, targets, "weighted")
        # Weighted should be higher (class 0 dominates, perfect on it)
        assert f1_weighted > f1_macro


class TestAccuracyByGroup:
    def test_ipa_groups_defined(self):
        assert "voyelles_orales" in IPA_GROUPS
        assert "voyelles_nasales" in IPA_GROUPS
        assert "occlusives" in IPA_GROUPS

    def test_voicing_groups_defined(self):
        assert "voise" in VOICING_GROUPS
        assert "non_voise" in VOICING_GROUPS

    def test_group_accuracy_returns_exact_and_group(self):
        preds = np.random.randint(0, NUM_PHONEMES, size=200)
        targets = np.random.randint(0, NUM_PHONEMES, size=200)
        results = accuracy_by_group(preds, targets, IPA_GROUPS)
        for _name, stats in results.items():
            assert "accuracy_exact" in stats
            assert "accuracy_group" in stats
            assert "support" in stats
            # group >= exact always (group match is more lenient)
            assert stats["accuracy_group"] >= stats["accuracy_exact"]


class TestReport:
    def test_generates_markdown(self):
        preds = np.random.randint(0, NUM_PHONEMES, size=100)
        targets = np.random.randint(0, NUM_PHONEMES, size=100)
        report = format_report(preds, targets)
        assert "Phone Error Rate" in report
        assert "Precision / Recall / F1" in report
        assert "groupe IPA" in report
        assert "voisement" in report
        assert "Decision" in report

    def test_report_with_topk(self):
        n = 100
        preds = np.random.randint(0, NUM_PHONEMES, size=n)
        targets = np.random.randint(0, NUM_PHONEMES, size=n)
        topk = np.random.randint(0, NUM_PHONEMES, size=(n, 5))
        report = format_report(preds, targets, topk_preds=topk)
        assert "Top-3 Accuracy" in report
        assert "Top-5 Accuracy" in report
