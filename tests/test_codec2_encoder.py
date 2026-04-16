"""Tests for deepvox.codec2.encoder module."""

import numpy as np

from deepvox.codec2.encoder import (
    BYTES_PER_FRAME,
    SAMPLES_PER_FRAME,
    add_delta_features,
    decode_frames,
    encode_pcm,
    unpack_frame,
    unpack_frames,
)


class TestEncodeDecode:
    def test_encode_shape(self):
        pcm = np.random.randint(-5000, 5000, size=SAMPLES_PER_FRAME * 5, dtype=np.int16)
        frames = encode_pcm(pcm)
        assert frames.shape == (5, BYTES_PER_FRAME)
        assert frames.dtype == np.uint8

    def test_encode_truncates_partial_frames(self):
        pcm = np.random.randint(-5000, 5000, size=SAMPLES_PER_FRAME * 3 + 100, dtype=np.int16)
        frames = encode_pcm(pcm)
        assert frames.shape[0] == 3

    def test_decode_output_length(self):
        pcm = np.random.randint(-5000, 5000, size=SAMPLES_PER_FRAME * 4, dtype=np.int16)
        frames = encode_pcm(pcm)
        reconstructed = decode_frames(frames)
        assert len(reconstructed) == SAMPLES_PER_FRAME * 4
        assert reconstructed.dtype == np.int16

    def test_roundtrip_nonzero(self):
        pcm = np.random.randint(-10000, 10000, size=SAMPLES_PER_FRAME * 3, dtype=np.int16)
        frames = encode_pcm(pcm)
        reconstructed = decode_frames(frames)
        # Codec2 is lossy, but output should not be all zeros for non-silent input
        assert np.any(reconstructed != 0)


class TestUnpack:
    def test_unpack_frame_shape(self):
        frame = np.zeros(BYTES_PER_FRAME, dtype=np.uint8)
        bits = unpack_frame(frame)
        assert bits.shape == (48,)
        assert bits.dtype == np.float32

    def test_unpack_frame_values(self):
        frame = np.array([0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00], dtype=np.uint8)
        bits = unpack_frame(frame)
        assert bits[0] == 1.0  # 0xFF → first bit is 1
        assert bits[8] == 0.0  # 0x00 → first bit is 0

    def test_unpack_frames_batch(self):
        frames = np.random.randint(0, 256, size=(10, BYTES_PER_FRAME), dtype=np.uint8)
        feats = unpack_frames(frames)
        assert feats.shape == (10, 48)


class TestDeltaFeatures:
    def test_delta_shape(self):
        feats = np.random.rand(10, 48).astype(np.float32)
        delta_feats = add_delta_features(feats)
        assert delta_feats.shape == (10, 96)

    def test_delta_first_frame_zero(self):
        feats = np.random.rand(5, 48).astype(np.float32)
        delta_feats = add_delta_features(feats)
        # First delta should be zeros
        np.testing.assert_array_equal(delta_feats[0, 48:], np.zeros(48))

    def test_delta_values(self):
        feats = np.array([[1.0] * 48, [3.0] * 48], dtype=np.float32)
        delta_feats = add_delta_features(feats)
        expected_delta = 2.0
        np.testing.assert_array_almost_equal(delta_feats[1, 48:], [expected_delta] * 48)
