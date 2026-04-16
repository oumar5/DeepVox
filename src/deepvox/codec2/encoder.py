"""Codec2 encoder/decoder wrapper for DeepVox.

Provides encoding of WAV audio (8 kHz, 16-bit mono) to Codec2 1200 bps frames
and decoding back to PCM. Each frame = 48 bits = 40 ms of audio.

Supports two backends:
  - pycodec2 (preferred): pure Python binding, no subprocess
  - CLI fallback: calls c2enc/c2dec binaries via subprocess
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

# Codec2 1200 bps constants
SAMPLE_RATE = 8000
MODE_1200 = 1200
FRAME_DURATION_MS = 40
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 320
BITS_PER_FRAME = 48
BYTES_PER_FRAME = BITS_PER_FRAME // 8  # 6


def _check_pycodec2():
    """Check if pycodec2 is available."""
    try:
        import pycodec2  # noqa: F401
        return True
    except ImportError:
        return False


def _check_cli():
    """Check if c2enc/c2dec CLI tools are available."""
    try:
        subprocess.run(["c2enc", "--help"], capture_output=True, check=False)
        return True
    except FileNotFoundError:
        return False


def encode_pcm(pcm: np.ndarray) -> np.ndarray:
    """Encode raw PCM samples to Codec2 1200 bps frames.

    Args:
        pcm: int16 PCM audio at 8 kHz, shape (n_samples,).

    Returns:
        Raw frame bytes as uint8 array, shape (n_frames, 6).
        Each row is one 48-bit Codec2 frame.
    """
    pcm = np.asarray(pcm, dtype=np.int16)

    if _check_pycodec2():
        return _encode_pycodec2(pcm)
    if _check_cli():
        return _encode_cli(pcm)

    raise RuntimeError(
        "No Codec2 backend found. Install pycodec2 (`pip install pycodec2`) "
        "or Codec2 CLI tools (`brew install codec2` / build from source)."
    )


def decode_frames(frames: np.ndarray) -> np.ndarray:
    """Decode Codec2 1200 bps frames back to PCM.

    Args:
        frames: uint8 array, shape (n_frames, 6).

    Returns:
        int16 PCM audio at 8 kHz.
    """
    frames = np.asarray(frames, dtype=np.uint8)

    if _check_pycodec2():
        return _decode_pycodec2(frames)
    if _check_cli():
        return _decode_cli(frames)

    raise RuntimeError(
        "No Codec2 backend found. Install pycodec2 (`pip install pycodec2`) "
        "or Codec2 CLI tools (`brew install codec2` / build from source)."
    )


def unpack_frame(frame_bytes: np.ndarray) -> np.ndarray:
    """Unpack a single 48-bit Codec2 frame into individual bit features.

    Args:
        frame_bytes: uint8 array of shape (6,) — one Codec2 frame.

    Returns:
        float32 array of shape (48,) — each element is 0.0 or 1.0.
        Bit layout (1200 bps mode): LSP (36 bits) + pitch (7 bits) + energy (5 bits).
    """
    frame_bytes = np.asarray(frame_bytes, dtype=np.uint8)
    bits = np.unpackbits(frame_bytes).astype(np.float32)
    return bits


def unpack_frames(frames: np.ndarray) -> np.ndarray:
    """Unpack multiple Codec2 frames into bit features.

    Args:
        frames: uint8 array, shape (n_frames, 6).

    Returns:
        float32 array, shape (n_frames, 48).
    """
    return np.stack([unpack_frame(f) for f in frames])


def add_delta_features(features: np.ndarray) -> np.ndarray:
    """Add delta (difference) features to frame features.

    Args:
        features: float32 array, shape (n_frames, 48).

    Returns:
        float32 array, shape (n_frames, 96) — original + delta.
    """
    deltas = np.zeros_like(features)
    deltas[1:] = features[1:] - features[:-1]
    return np.concatenate([features, deltas], axis=1)


# --- pycodec2 backend ---

def _encode_pycodec2(pcm: np.ndarray) -> np.ndarray:
    import pycodec2

    codec = pycodec2.Codec2(MODE_1200)
    n_samples = len(pcm)
    # Truncate to whole frames
    n_frames = n_samples // SAMPLES_PER_FRAME
    pcm = pcm[: n_frames * SAMPLES_PER_FRAME]

    frames = []
    for i in range(n_frames):
        chunk = pcm[i * SAMPLES_PER_FRAME : (i + 1) * SAMPLES_PER_FRAME]
        encoded = codec.encode(chunk)
        frame = np.frombuffer(encoded, dtype=np.uint8)
        frames.append(frame)

    return np.stack(frames)


def _decode_pycodec2(frames: np.ndarray) -> np.ndarray:
    import pycodec2

    codec = pycodec2.Codec2(MODE_1200)
    pcm_chunks = []
    for frame in frames:
        decoded = codec.decode(frame.tobytes())
        pcm_chunks.append(np.frombuffer(decoded, dtype=np.int16))

    return np.concatenate(pcm_chunks)


# --- CLI backend ---

def _encode_cli(pcm: np.ndarray) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as raw_f:
        raw_path = raw_f.name
        raw_f.write(pcm.tobytes())

    c2_path = raw_path + ".c2"
    try:
        subprocess.run(
            ["c2enc", "1200", raw_path, c2_path],
            check=True,
            capture_output=True,
        )
        c2_data = Path(c2_path).read_bytes()
        n_frames = len(c2_data) // BYTES_PER_FRAME
        frames = np.frombuffer(c2_data, dtype=np.uint8).reshape(n_frames, BYTES_PER_FRAME)
        return frames.copy()
    finally:
        Path(raw_path).unlink(missing_ok=True)
        Path(c2_path).unlink(missing_ok=True)


def _decode_cli(frames: np.ndarray) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".c2", delete=False) as c2_f:
        c2_path = c2_f.name
        c2_f.write(frames.tobytes())

    raw_path = c2_path + ".raw"
    try:
        subprocess.run(
            ["c2dec", "1200", c2_path, raw_path],
            check=True,
            capture_output=True,
        )
        raw_data = Path(raw_path).read_bytes()
        return np.frombuffer(raw_data, dtype=np.int16).copy()
    finally:
        Path(c2_path).unlink(missing_ok=True)
        Path(raw_path).unlink(missing_ok=True)


# --- Utility ---

def wav_to_frames(wav_path: str | Path) -> np.ndarray:
    """Load a WAV file (any sample rate) and encode to Codec2 frames.

    Automatically resamples to 8 kHz mono if needed.

    Args:
        wav_path: path to a WAV/MP3/FLAC file.

    Returns:
        uint8 array, shape (n_frames, 6).
    """
    import librosa

    audio, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    # librosa returns float32 in [-1, 1], convert to int16
    pcm = (audio * 32767).astype(np.int16)
    return encode_pcm(pcm)
