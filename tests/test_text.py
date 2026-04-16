"""Tests for deepvox.data.text module."""

from deepvox.data.text import (
    BLANK_IDX,
    UNK_IDX,
    VOCAB_SIZE,
    decode,
    decode_ctc,
    encode,
    normalize_text,
)


class TestNormalize:
    def test_lowercase(self):
        assert normalize_text("Bonjour") == "bonjour"

    def test_smart_quotes(self):
        assert normalize_text("l’ami") == "l'ami"

    def test_em_dash(self):
        assert normalize_text("a—b") == "a-b"

    def test_collapse_whitespace(self):
        assert normalize_text("a   b\t\tc") == "a b c"

    def test_strip(self):
        assert normalize_text("  hello  ") == "hello"

    def test_french_accents_preserved(self):
        assert normalize_text("Château côté") == "château côté"

    def test_unknown_chars_removed(self):
        # @, # should be stripped as punctuation
        result = normalize_text("hello@world")
        assert "@" not in result


class TestEncodeDecode:
    def test_roundtrip(self):
        text = "bonjour"
        ids = encode(text)
        assert len(ids) == len(text)
        assert decode(ids) == text

    def test_accented(self):
        text = "père"
        ids = encode(text)
        assert decode(ids) == text

    def test_unknown_uses_unk(self):
        # Unicode char not in vocab
        ids = encode("a★b")
        # Star not in vocab → UNK
        assert UNK_IDX in ids


class TestCTCDecode:
    def test_collapse_repeats(self):
        # Simulate "hhhheelllooo" with blanks → "hello"
        # indices: h=, e=, l=, o=
        # Using BLANK_IDX for blank
        h = encode("h")[0]
        e = encode("e")[0]
        ll = encode("l")[0]
        o = encode("o")[0]
        ids = [h, h, BLANK_IDX, e, e, ll, ll, BLANK_IDX, ll, o, o, o]
        assert decode_ctc(ids) == "hello"

    def test_no_repeats(self):
        ids = encode("cat")
        assert decode_ctc(ids) == "cat"

    def test_all_blanks(self):
        assert decode_ctc([BLANK_IDX] * 10) == ""


class TestVocab:
    def test_blank_at_zero(self):
        assert BLANK_IDX == 0

    def test_vocab_size_reasonable(self):
        # 26 base + 18 accented + 3 separators + 2 special = ~49
        assert 40 < VOCAB_SIZE < 60
