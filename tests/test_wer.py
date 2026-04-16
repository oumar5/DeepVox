"""Tests for deepvox.eval.wer module."""

import pytest

from deepvox.eval.wer import cer, levenshtein, wer


class TestLevenshtein:
    def test_empty(self):
        assert levenshtein([], []) == 0

    def test_equal(self):
        assert levenshtein(["a", "b"], ["a", "b"]) == 0

    def test_substitution(self):
        assert levenshtein(["a", "b"], ["a", "c"]) == 1

    def test_insertion(self):
        assert levenshtein(["a"], ["a", "b"]) == 1

    def test_deletion(self):
        assert levenshtein(["a", "b"], ["a"]) == 1


class TestWER:
    def test_perfect(self):
        refs = ["hello world"]
        hyps = ["hello world"]
        assert wer(refs, hyps) == 0.0

    def test_all_wrong(self):
        refs = ["aaa bbb"]
        hyps = ["xxx yyy"]
        assert wer(refs, hyps) == 1.0

    def test_half_wrong(self):
        refs = ["a b c d"]
        hyps = ["a b X Y"]
        assert wer(refs, hyps) == pytest.approx(0.5)


class TestCER:
    def test_perfect(self):
        assert cer(["bonjour"], ["bonjour"]) == 0.0

    def test_one_sub(self):
        # "bonjour" vs "bonsour" = 1 sub / 7 chars
        assert cer(["bonjour"], ["bonsour"]) == pytest.approx(1 / 7)

    def test_empty_ref(self):
        # Defensive: empty ref shouldn't crash
        assert cer([""], [""]) == 0.0
