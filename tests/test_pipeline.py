import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from got_ocr2_pipeline import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODE,
    DEFAULT_WEIGHTS_DIR,
    MAX_IMAGE_SIDE,
    MAX_NEW_TOKENS,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    MODES,
    STOP_STRING,
    GotOcr2Pipeline,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "stepfun-ai/GOT-OCR-2.0-hf"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert DEFAULT_MODE == MODES[0] == "plain" and MODES == ("plain", "format")
    assert 1 <= DEFAULT_MAX_NEW_TOKENS <= MAX_NEW_TOKENS == 4096 and STOP_STRING == "<|im_end|>"
    manifest = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["modelId"] == MODEL_ID
        assert data["revision"] == MODEL_REVISION


def _write_snapshot(root: Path, content: bytes, sha: str | None = None, size: int | None = None) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    _write_snapshot(tmp_path, b'{"model_type": "got_ocr2"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == MODEL_REVISION and info["files"] == 1


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "got_ocr2"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_missing_file_and_revision(tmp_path):
    _write_snapshot(tmp_path, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["revision"] = "0" * 40
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


TEXT = "NOTICE OF ANNUAL GENERAL MEETING\nThe annual general meeting will be held on Thursday."


def _fake_pipeline(calls: list | None = None) -> GotOcr2Pipeline:
    def runner(images, mode, max_new_tokens):
        if calls is not None:
            calls.append((tuple(image.mode for image in images), mode, max_new_tokens))
        return [{"text": TEXT + STOP_STRING + "\n", "new_tokens": 40} for _ in images]

    return GotOcr2Pipeline(runner, "cpu", "float32", "injected")


def test_recognize_output_fields_and_defaults():
    calls: list = []
    pipe = _fake_pipeline(calls)
    result = pipe.recognize(Image.new("L", (400, 300)))
    assert result["text"] == TEXT  # stop string and trailing whitespace stripped
    assert result["mode"] == DEFAULT_MODE
    assert result["image_size"] == [400, 300]
    assert result["new_tokens"] == 40 and result["truncated"] is False
    assert result["generation"] == {
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "do_sample": False,
        "decoding": "greedy",
    }
    assert (result["model_id"], result["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert (result["device"], result["dtype"], result["source"]) == ("cpu", "float32", "injected")
    assert calls == [(("RGB",), DEFAULT_MODE, DEFAULT_MAX_NEW_TOKENS)]


def test_recognize_reports_truncation_and_format_mode():
    pipe = _fake_pipeline()
    result = pipe.recognize(Image.new("RGB", (64, 64)), mode="format", max_new_tokens=40)
    assert result["truncated"] is True and result["mode"] == "format"


def test_recognize_rejects_bad_inputs():
    pipe = _fake_pipeline()
    with pytest.raises(TypeError):
        pipe.recognize(np.zeros((30, 40, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        pipe.recognize(Image.new("RGB", (MIN_IMAGE_SIDE - 1, 64)))
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.recognize(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 64)))
    with pytest.raises(ValueError, match="MAX_IMAGE_PIXELS"):
        pipe.recognize(Image.new("RGB", (8192, 4096)))
    with pytest.raises(ValueError, match="MODES"):
        pipe.recognize(Image.new("RGB", (64, 64)), mode="markdown")
    with pytest.raises(TypeError, match="mode must be a str"):
        pipe.recognize(Image.new("RGB", (64, 64)), mode=None)
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        pipe.recognize(Image.new("RGB", (64, 64)), max_new_tokens=MAX_NEW_TOKENS + 1)
    with pytest.raises(TypeError, match="max_new_tokens"):
        pipe.recognize(Image.new("RGB", (64, 64)), max_new_tokens=True)


def test_recognize_rejects_malformed_runner_output():
    pipe = GotOcr2Pipeline(lambda *args: [{"tokens": 1}], "cpu")
    with pytest.raises(RuntimeError, match="text"):
        pipe.recognize(Image.new("RGB", (64, 64)))
    pipe = GotOcr2Pipeline(lambda images, *args: [{"text": "a"}] * (len(images) + 1), "cpu")
    with pytest.raises(RuntimeError, match="per image"):
        pipe.recognize(Image.new("RGB", (64, 64)))


def test_transcribe_batches_and_orders():
    calls: list = []
    pipe = _fake_pipeline(calls)
    items = pipe.transcribe([Image.new("RGB", (64, 64))] * 5, batch_size=2, max_new_tokens=64)
    assert [len(c[0]) for c in calls] == [2, 2, 1] and all(c[1] == DEFAULT_MODE and c[2] == 64 for c in calls)
    assert len(items) == 5
    assert all(i["text"] == TEXT and i["new_tokens"] == 40 and i["truncated"] is False for i in items)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.transcribe([Image.new("RGB", (64, 64))], batch_size=0)


def test_model_backed_methods_refuse_injected_runner(tmp_path):
    pipe = _fake_pipeline()
    with pytest.raises(RuntimeError, match="injected"):
        pipe.adapt([], None)
    with pytest.raises(RuntimeError, match="injected"):
        pipe.save_artifact(tmp_path)
    with pytest.raises(RuntimeError, match="injected"):
        pipe.load_artifact(tmp_path)
