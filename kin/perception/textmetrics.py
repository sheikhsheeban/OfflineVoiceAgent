"""Word error rate, for evaluating speech models."""

from __future__ import annotations


def _words(s: str) -> list[str]:
    return "".join(c if c.isalnum() or c.isspace() else " " for c in s.lower()).split()


def wer(ref: str, hyp: str) -> float:
    """Word-level edit distance divided by the reference length (punctuation and case ignored)."""
    r, h = _words(ref), _words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))
