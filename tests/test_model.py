"""Tests for deepvox.models.phoneme_classifier module."""

import pytest
import torch

from deepvox.data.preprocess import NUM_PHONEMES
from deepvox.models.phoneme_classifier import PhonemeClassifier


class TestPhonemeClassifier:
    @pytest.mark.parametrize("input_dim", [48, 96, 80, 320])
    def test_output_shape(self, input_dim):
        model = PhonemeClassifier(input_dim=input_dim)
        x = torch.randn(2, 10, input_dim)
        out = model(x)
        assert out.shape == (2, 10, NUM_PHONEMES)

    def test_predict_shape(self):
        model = PhonemeClassifier(input_dim=48)
        x = torch.randn(4, 15, 48)
        preds = model.predict(x)
        assert preds.shape == (4, 15)
        assert preds.dtype == torch.int64

    def test_predict_range(self):
        model = PhonemeClassifier(input_dim=48)
        x = torch.randn(2, 10, 48)
        preds = model.predict(x)
        assert preds.min() >= 0
        assert preds.max() < NUM_PHONEMES

    def test_parameter_count(self):
        model = PhonemeClassifier(input_dim=48)
        params = model.count_parameters()
        assert params > 0
        # Should be around 1-3M for this architecture
        assert 500_000 < params < 10_000_000
