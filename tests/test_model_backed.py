"""Model-backed checks that run only where the pinned snapshot is staged: the measured model facts, batched
recognition equal to single-image recognition, corpus evaluation on drawn printed lines, a one-epoch adaptation of
the last decoder layers, the artifact round trip with reload parity, the loader's scope check, the transactional
guarantee and — where CUDA is visible — the same path on the accelerator. Skipped when the weights are absent."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from got_ocr2_pipeline import (
    ADAPTER_PARAMETERS,
    DEFAULT_WEIGHTS_DIR,
    PARAMETER_COUNT,
    PLAIN_PROMPT_TOKENS,
    WEIGHT_FILE,
    GotOcr2Pipeline,
)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

LINES = ["The council met on Thursday 12 May.", "Minutes of the previous meeting were read.", "Approved without amendment by 14 votes.", "The treasurer reported a balance of 3000 francs.", "Repairs to the town hall roof were authorised.", "The session closed at nine in the evening.", "A petition from the market traders was received.", "Referred to the finance committee for report.", "Street lighting on the main road was discussed.", "The mayor thanked the members present.", "Next meeting fixed for the first Monday in June.", "Signed by the secretary and the mayor.", "Two new fountains were proposed for the square.", "The school inspector's letter was noted.", "Contracts for paving were put to tender.", "The motion was carried unanimously."]


def _record(i, size=(1024, 320)):
    image = Image.new("RGB", size, (250, 248, 240))
    draw = ImageDraw.Draw(image)
    draw.text((24, 130), LINES[i], fill=(15, 15, 40), font=ImageFont.load_default(size=44))
    image.putpixel((i, 0), (i % 256, 0, 0))
    return {"id": f"line{i:02d}", "image": image, "text": LINES[i]}


@pytest.fixture(scope="module")
def records():
    return [_record(i) for i in range(16)]


@pytest.fixture(scope="module")
def pipe():
    return GotOcr2Pipeline.from_pretrained(weights_dir=DEFAULT_WEIGHTS_DIR)


def test_model_facts_batched_recognition_and_frozen_evaluation(pipe, records):
    assert sum(p.numel() for p in pipe._model.parameters()) == PARAMETER_COUNT
    assert sum(p.numel() for n, p in pipe._model.named_parameters() if n.startswith(("model.language_model.layers.2", "model.language_model.norm."))) == ADAPTER_PARAMETERS
    assert pipe.weight_sha256 is not None and len(pipe.weight_sha256) == 64 and pipe.dtype == "float32"
    encoded = pipe._processor([records[0]["image"], records[1]["image"]], return_tensors="pt", padding=True)
    assert tuple(encoded["input_ids"].shape) == (2, PLAIN_PROMPT_TOKENS) and bool(encoded["attention_mask"].all())
    single = [pipe.recognize(r["image"], max_new_tokens=64) for r in records[:3]]
    batched = pipe.transcribe([r["image"] for r in records[:3]], max_new_tokens=64, batch_size=3)
    assert all(s["truncated"] is False and s["new_tokens"] > 0 for s in single)
    assert [s["text"] for s in single] == [b["text"] for b in batched] and [s["new_tokens"] for s in single] == [b["new_tokens"] for b in batched]
    metrics = pipe.evaluate(records[:8], max_new_tokens=64)
    assert metrics["n"] == 8 and metrics["cer"] < 0.5 and metrics["adapted"] is False and metrics["verdict"] == "measured-small-sample"
    assert len(metrics["hypotheses"]) == 8 and metrics["truncated"] == 0


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, records, tmp_path):
    result = pipe.adapt(records[:12], records[12:], epochs=1, batch_size=4, max_new_tokens=64)
    assert result["n_trainable"] == ADAPTER_PARAMETERS and result["n_total"] == PARAMETER_COUNT and result["first_trainable_layer"] == 20
    assert result["history"][0]["note"] == "frozen model" and result["history"][1]["train_loss"] > 0.0
    assert set(result["history"][1]["val"]) == {"cer", "wer", "cer_macro", "exact_match", "n"} and result["best_epoch"] in (0, 1)
    assert all(n.startswith(("model.language_model.layers.2", "model.language_model.norm.")) for n in result["trainable_names"])
    assert not any(n.startswith(("model.vision_tower", "model.multi_modal_projector", "model.language_model.embed_tokens", "lm_head")) for n in result["trainable_names"])
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"]) and manifest["base"]["weight_sha256"] == pipe.weight_sha256
    assert manifest["metadata"] == {"note": "test"} and manifest["adapter"]["selection"] == "lowest validation CER"
    reloaded = GotOcr2Pipeline.from_artifact(artifact, weights_dir=DEFAULT_WEIGHTS_DIR)
    assert pipe.transcribe([r["image"] for r in records[:3]], max_new_tokens=64) == reloaded.transcribe([r["image"] for r in records[:3]], max_new_tokens=64)
    assert reloaded.adapter["best_epoch"] == result["best_epoch"] and reloaded.evaluate(records[:4], max_new_tokens=64)["adapted"] is True
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, records, tmp_path):
    result = pipe.adapt(records[:12], None, epochs=2, batch_size=4)
    assert result["best_epoch"] == 2 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 3
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = GotOcr2Pipeline.from_artifact(artifact, weights_dir=DEFAULT_WEIGHTS_DIR)
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])


def test_adapt_refuses_bad_hyperparameters_and_datasets(pipe, records):
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(records[:12], None, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(records[:12], None, epochs=1, lr=0.5)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(records[:12], None, epochs=1, batch_size=0)
    with pytest.raises(ValueError, match="8..5000"):
        pipe.adapt(records[:4], None, epochs=1)
    with pytest.raises(ValueError, match="1..512"):
        pipe.adapt([*records[:11], {**records[11], "text": ""}], None, epochs=1)
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(pipe, records, tmp_path):
    from safetensors.torch import load_file, save_file

    pipe.adapt(records[:12], None, epochs=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        GotOcr2Pipeline.from_artifact(fewer, weights_dir=DEFAULT_WEIGHTS_DIR)
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["model.language_model.norm.zz_extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    files = [{**manifest["files"][0], "bytes": (extra / "adapter.safetensors").stat().st_size, "sha256": digest}]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        GotOcr2Pipeline.from_artifact(extra, weights_dir=DEFAULT_WEIGHTS_DIR)
    vision = tmp_path / "vision"
    shutil.copytree(artifact, vision)
    (vision / "manifest.json").write_text(json.dumps({**manifest, "tensors": [*manifest["tensors"], "model.vision_tower.layers.0.x"]}))
    with pytest.raises(ValueError, match="last 4 decoder layers"):
        GotOcr2Pipeline.from_artifact(vision, weights_dir=DEFAULT_WEIGHTS_DIR)


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe, records):
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}
    adapter_before = pipe.adapter

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(records[:12], None, epochs=2, batch_size=4, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert pipe.adapter is adapter_before
    assert not any(p.requires_grad for p in pipe._model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_the_default_device_is_cuda_when_visible(pipe):
    assert pipe.device == "cuda:0" and next(pipe._model.parameters()).device.type == "cuda"
