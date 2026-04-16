"""French character-level tokenizer for DeepVox ASR (Phase 2).

Character-level approach (not BPE / wordpiece) for:
  - Simplicity
  - Small vocab (~50 classes + blank)
  - CTC-friendly (blank token reserved for CTC)

Vocabulary:
  [blank], a-z, accented chars, space, apostrophe, hyphen
"""

from __future__ import annotations

import unicodedata

# French-specific characters (with diacritics)
BASE_LETTERS = list("abcdefghijklmnopqrstuvwxyz")
ACCENTED = list("àâäéèêëïîôöùûüÿçœæ")
SEPARATORS = [" ", "'", "-"]

# Special tokens
BLANK_TOKEN = "<blank>"  # CTC blank, always index 0
UNK_TOKEN = "<unk>"  # unknown character

VOCAB = [BLANK_TOKEN, UNK_TOKEN] + BASE_LETTERS + ACCENTED + SEPARATORS
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCAB)}
IDX_TO_CHAR = {i: c for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
BLANK_IDX = 0
UNK_IDX = 1


def normalize_text(text: str) -> str:
    """Normalize French text for character-level ASR.

    Steps:
      1. Unicode NFC normalization (canonical composition)
      2. Lowercase
      3. Replace smart quotes / typographic chars
      4. Collapse multiple spaces
      5. Strip

    Args:
        text: raw text.

    Returns:
        Cleaned lowercase text.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    # Typographic replacements
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "«": '"',
        "»": '"',
        "–": "-",
        "—": "-",
        "…": "...",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    # Remove characters not in vocab (keep alphanumeric + known separators)
    # Everything not in VOCAB becomes UNK_TOKEN
    # But first strip basic punctuation
    allowed = set(VOCAB[2:])  # exclude blank + unk
    cleaned = []
    for ch in text:
        if ch in allowed:
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append(" ")
        # else: skip punctuation and unknown chars

    text = "".join(cleaned)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def encode(text: str) -> list[int]:
    """Convert text to list of character indices.

    Args:
        text: normalized text.

    Returns:
        List of char indices (int).
    """
    ids = []
    for ch in text:
        ids.append(CHAR_TO_IDX.get(ch, UNK_IDX))
    return ids


def decode(ids: list[int]) -> str:
    """Convert list of indices back to text.

    Args:
        ids: list of char indices (no blank collapse — use decode_ctc for that).

    Returns:
        Decoded string.
    """
    chars = []
    for i in ids:
        if i == BLANK_IDX:
            continue  # skip CTC blanks
        ch = IDX_TO_CHAR.get(i, "")
        if ch == UNK_TOKEN:
            continue
        chars.append(ch)
    return "".join(chars)


def decode_ctc(ids: list[int]) -> str:
    """CTC decoding: collapse repeats then remove blanks.

    Args:
        ids: list of char indices (output of model argmax).

    Returns:
        Decoded text.
    """
    # Step 1: collapse consecutive repeats
    collapsed = []
    prev = -1
    for i in ids:
        if i != prev:
            collapsed.append(i)
            prev = i
    # Step 2: remove blanks
    return decode(collapsed)
