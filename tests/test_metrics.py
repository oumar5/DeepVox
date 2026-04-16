"""Tests for deepvox.eval.metrics module."""

import numpy as np
import pytest

from deepvox.data.preprocess import NUM_PHONEMES
from deepvox.eval.metrics import (
    confusion_matrix,
    format_report,
    phone_error_rate,
    top_confused_pairs,
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


class TestReport:
    def test_generates_markdown(self):
        preds = np.random.randint(0, NUM_PHONEMES, size=100)
        targets = np.random.randint(0, NUM_PHONEMES, size=100)
        report = format_report(preds, targets)
        assert "Phone Error Rate" in report
        assert "Macro Precision" in report
        assert "Decision" in report
