"""Word Error Rate (WER) and Character Error Rate (CER) for ASR evaluation.

WER = (substitutions + deletions + insertions) / reference word count
CER = same but at character level

Both computed via Levenshtein distance on tokenized sequences.
"""

from __future__ import annotations


def levenshtein(a: list, b: list) -> int:
    """Compute Levenshtein edit distance between two sequences.

    Args:
        a: reference sequence.
        b: hypothesis sequence.

    Returns:
        Minimum number of insertions, deletions, substitutions.
    """
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, curr = curr, prev

    return prev[n]


def wer(references: list[str], hypotheses: list[str]) -> float:
    """Compute Word Error Rate over a corpus.

    Args:
        references: list of ground-truth transcriptions.
        hypotheses: list of model predictions.

    Returns:
        WER in [0, inf). Can exceed 1.0 if insertions dominate.
    """
    assert len(references) == len(hypotheses)
    total_errors = 0
    total_words = 0

    for ref, hyp in zip(references, hypotheses, strict=True):
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_errors += levenshtein(ref_words, hyp_words)
        total_words += len(ref_words)

    return total_errors / max(total_words, 1)


def cer(references: list[str], hypotheses: list[str]) -> float:
    """Compute Character Error Rate over a corpus.

    Spaces count as characters. Normalized on reference character count.

    Args:
        references: list of ground-truth transcriptions.
        hypotheses: list of model predictions.

    Returns:
        CER in [0, inf).
    """
    assert len(references) == len(hypotheses)
    total_errors = 0
    total_chars = 0

    for ref, hyp in zip(references, hypotheses, strict=True):
        total_errors += levenshtein(list(ref), list(hyp))
        total_chars += len(ref)

    return total_errors / max(total_chars, 1)


def format_asr_report(
    references: list[str],
    hypotheses: list[str],
    examples: int = 10,
) -> str:
    """Generate a markdown report for ASR evaluation.

    Args:
        references: ground-truth texts.
        hypotheses: predicted texts.
        examples: number of qualitative examples to include.

    Returns:
        Markdown-formatted report.
    """
    corpus_wer = wer(references, hypotheses)
    corpus_cer = cer(references, hypotheses)

    lines = [
        "# Phase 2 — ASR Evaluation Report",
        "",
        "## Global metrics",
        "",
        f"- **Word Error Rate (WER):** {corpus_wer:.4f} ({corpus_wer * 100:.1f} %)",
        f"- **Character Error Rate (CER):** {corpus_cer:.4f} ({corpus_cer * 100:.1f} %)",
        f"- **Samples evaluated:** {len(references)}",
        "",
        "## Qualitative examples",
        "",
    ]

    for i in range(min(examples, len(references))):
        lines.append(f"### Example {i + 1}")
        lines.append("")
        lines.append(f"- **Ref:** `{references[i]}`")
        lines.append(f"- **Hyp:** `{hypotheses[i]}`")
        lines.append("")

    # GO / NO-GO decision
    lines += [
        "## Decision",
        "",
    ]
    if corpus_wer <= 0.25:
        lines.append(f"**GO** — WER = {corpus_wer*100:.1f} % (≤25 %). ASR utilisable.")
    elif corpus_wer <= 0.50:
        lines.append(f"**GO conditionnel** — WER = {corpus_wer*100:.1f} % (25-50 %). Utile avec LM.")
    else:
        lines.append(f"**NO-GO** — WER = {corpus_wer*100:.1f} % (>50 %). Plus de données / meilleure architecture.")

    return "\n".join(lines)
