"""Offline tests for the public validation, metric and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest
from PIL import Image

from got_ocr2_pipeline import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODE,
    INPUT_SCHEMA,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    MODES,
    character_error_rate,
    evaluation_report,
    validate_inputs,
    word_error_rate,
)

TEXT = "NOTICE OF ANNUAL GENERAL MEETING\nThe annual general meeting will be held on Thursday."


def _image(width: int = 1000, height: int = 520) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def _result(text: str = TEXT, truncated: bool = False, mode: str = DEFAULT_MODE) -> dict:
    return {"text": text, "mode": mode, "truncated": truncated}


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_image(), names=["page.png"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["modes"] == list(MODES)
    assert manifest["inputs"] == [{"id": "page.png", "mode": "RGB", "size": [1000, 520]}]
    assert manifest["recognition_mode"] == DEFAULT_MODE
    assert manifest["generation"] == {
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "do_sample": False,
        "decoding": "greedy",
    }
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_explicit_request() -> None:
    manifest = validate_inputs(_image(), mode="format", max_new_tokens=512)
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["recognition_mode"] == "format" and manifest["generation"]["max_new_tokens"] == 512


def test_validate_inputs_rejects_like_recognize() -> None:
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        validate_inputs(_image(MAX_IMAGE_SIDE + 1, 64))
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs(_image(8, 8))
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        validate_inputs("not an image")
    with pytest.raises(ValueError, match="MODES"):
        validate_inputs(_image(), mode="latex")
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        validate_inputs(_image(), max_new_tokens=0)
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(_image(), names=["a", "b"])


def test_character_error_rate_counts_character_edits() -> None:
    assert character_error_rate("hello world", "hello world") == 0.0
    assert character_error_rate("hello world", "hell0 world") == pytest.approx(1 / 11)
    assert character_error_rate("a  b", "a b") == 0.0  # whitespace collapsed
    assert character_error_rate("Hello", "hello") == pytest.approx(0.2)  # case kept
    with pytest.raises(ValueError, match="at least one character"):
        character_error_rate("   ", "x")


def test_word_error_rate_counts_word_edits() -> None:
    assert word_error_rate("a b c", "a b c") == 0.0
    assert word_error_rate("a b c d", "a x c") == pytest.approx(0.5)
    assert word_error_rate("Hello World", "hello world") == 0.0
    with pytest.raises(ValueError, match="at least one word"):
        word_error_rate("   ", "x")


def test_evaluation_report_not_measurable_without_reference() -> None:
    report = evaluation_report(_result())
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["n_chars"] == len(TEXT) and report["recognition_mode"] == DEFAULT_MODE
    assert "character and word error rate" in report["needs"]
    assert report["baselines"] == []
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert "no score" in report["score_semantics"]


def test_evaluation_report_sample_sanity_with_reference() -> None:
    report = evaluation_report(_result(), TEXT, sample_kind="synthetic")
    assert report["verdict"] == "sample-sanity" and report["sample_kind"] == "synthetic"
    by_id = {m["id"]: m for m in report["metrics"]}
    assert by_id["character_error_rate"]["value"] == 0.0
    assert by_id["word_error_rate"]["value"] == 0.0
    noisy = evaluation_report(_result(TEXT.replace("Thursday", "Tuesday")), TEXT)
    by_id = {m["id"]: m for m in noisy["metrics"]}
    assert 0 < by_id["character_error_rate"]["value"] < by_id["word_error_rate"]["value"] < 0.2


def test_evaluation_report_carries_truncation_flag_and_mode() -> None:
    report = evaluation_report(_result(truncated=True, mode="format"))
    assert report["truncated"] is True and report["recognition_mode"] == "format"
