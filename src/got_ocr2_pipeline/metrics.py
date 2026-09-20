"""Corpus-level text-recognition measures and two non-neural baselines, in plain Python.

``ocr_metrics`` scores one hypothesis string per record against ``record['text']``: the character error rate and the
word error rate as **micro** averages (total Levenshtein edits over total reference characters or words — the
corpus CER/WER of the handwriting-recognition literature) and as **macro** averages (mean per-line rate), the exact-
match rate, and the counts behind them. Neither rate is capped: a hypothesis longer than its reference can push a
rate above 1.0, which is the signal that the model is generating text the image does not carry. The baselines
answer without looking at the image — the empty string, or one constant training transcript — and are scored by
the same function.
"""
# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline import edit_distance, normalise_text

METRIC_DEFINITIONS = {
    "cer": "character error rate, micro: total character edits (insertions + deletions + substitutions) over total reference characters; 0 is perfect, values above 1 mean over-generation",
    "wer": "word error rate, micro: total word edits over total reference words after lower-casing and whitespace tokenisation; punctuation kept",
    "cer_macro": "mean of the per-line character error rates (each line weighted equally regardless of length)",
    "wer_macro": "mean of the per-line word error rates",
    "exact_match": "fraction of lines whose hypothesis equals the reference after whitespace normalisation",
}
MEDOID_POOL = 120  # training transcripts considered when choosing the constant baseline (quadratic cost)


def _words(text: str) -> list[str]:
    return text.lower().split()


def ocr_metrics(hypotheses: Sequence[str], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score one hypothesis per record against `record['text']`; per-line rows plus the corpus rates. Raises when
    the lengths differ or nothing is scored."""
    if len(hypotheses) != len(records) or not records:
        raise ValueError("hypotheses and records must be non-empty and the same length")
    rows = []
    char_edits = word_edits = ref_chars = ref_words = hyp_chars = exact = 0
    for hypothesis, record in zip(hypotheses, records, strict=True):
        reference = normalise_text(record["text"])
        hyp = normalise_text(str(hypothesis))
        if not reference:
            raise ValueError(f"record {record.get('id')!r} has an empty reference")
        ce = edit_distance(reference, hyp)
        we = edit_distance(_words(reference), _words(hyp))
        n_words = len(_words(reference))
        rows.append({"id": record.get("id"), "ref_chars": len(reference), "hyp_chars": len(hyp), "char_edits": ce, "cer": ce / len(reference), "word_edits": we, "wer": we / n_words, "exact": hyp == reference})
        char_edits += ce
        word_edits += we
        ref_chars += len(reference)
        ref_words += n_words
        hyp_chars += len(hyp)
        exact += hyp == reference
    n = len(rows)
    return {
        "n": n,
        "cer": char_edits / ref_chars,
        "wer": word_edits / ref_words,
        "cer_macro": sum(r["cer"] for r in rows) / n,
        "wer_macro": sum(r["wer"] for r in rows) / n,
        "exact_match": exact / n,
        "char_edits": char_edits,
        "ref_chars": ref_chars,
        "hyp_chars": hyp_chars,
        "word_edits": word_edits,
        "ref_words": ref_words,
        "rows": rows,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def empty_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Predict the empty string for every line: CER and WER are exactly 1.0 by construction (every reference
    character is a deletion). The floor any recogniser must beat to be doing better than silence."""
    result = ocr_metrics([""] * len(records), records)
    result["baseline"] = "empty string for every line"
    return result


def medoid_transcript(train: Sequence[Mapping[str, Any]], *, pool: int = MEDOID_POOL) -> str:
    """The training transcript (among the first `pool`) with the smallest summed character error rate to the others:
    the single constant string that best matches the corpus on average."""
    texts = [normalise_text(r["text"]) for r in train[:pool]]
    texts = [t for t in texts if t]
    if not texts:
        raise ValueError("no non-empty training transcripts")
    return min(texts, key=lambda candidate: sum(edit_distance(other, candidate) / len(other) for other in texts))


def constant_baseline(train: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]], *, pool: int = MEDOID_POOL) -> dict[str, Any]:
    """Predict one constant training transcript (the medoid) for every line: what corpus statistics alone buy
    without reading the image — usually a CER near 1.0, since edits are dominated by substitutions."""
    text = medoid_transcript(train, pool=pool)
    result = ocr_metrics([text] * len(records), records)
    result["baseline"] = f"constant training transcript {text!r}"
    result["transcript"] = text
    return result
