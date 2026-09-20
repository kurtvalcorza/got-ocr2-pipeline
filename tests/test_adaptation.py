"""Offline checks of the adaptation contract: the text-line record contract and its refusals, the pinned-corpus
refusals and the draw, splitting, the BYOD loader, the metrics and baselines, `evaluate` with an injected runner,
the artifact-manifest checks, and the model-free refusals of `adapt` / artifacts."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image, ImageDraw

from got_ocr2_pipeline import (
    ARTIFACT_FORMAT,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_SPLIT,
    GotOcr2Pipeline,
    build_sample_dataset,
    check_split_disjoint,
    constant_baseline,
    dataset_digest,
    edit_distance,
    empty_baseline,
    fetch_corpus,
    image_digest,
    load_byod_dataset,
    medoid_transcript,
    normalise_text,
    ocr_metrics,
    read_corpus,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)
from got_ocr2_pipeline import pipeline as pl
from got_ocr2_pipeline import samples as sm

LINES = ["les intérêts des 30000 francs", "Le Conseil municipal,", "séance du 12 mai", "adopté à l'unanimité", "rapport de la commission", "des finances", "Monsieur le Maire", "expose que", "la ville de Belfort", "considérant", "vu la délibération", "Art. 1er"]


def _line(i, size=(640, 64)):
    image = Image.new("RGB", size, (245, 240, 230))
    draw = ImageDraw.Draw(image)
    draw.text((8, 20), LINES[i % len(LINES)], fill=(20, 20, 60))
    image.putpixel((i % size[0], 0), (i % 256, 0, 0))
    return image


def _record(i, text=None):
    return {"id": f"r{i:03d}", "image": _line(i), "text": LINES[i % len(LINES)] if text is None else text}


def _records(n=12):
    return [_record(i) for i in range(n)]


def _echo_runner(images, mode, max_new_tokens):
    """An injected runner that returns the line it was shown (the harness keeps a side table by pixel digest)."""
    return [{"text": _echo_runner.table.get(image_digest(image), ""), "new_tokens": 5} for image in images]


_echo_runner.table = {}


# --- record contract -----------------------------------------------------------------------------------------------


def test_validate_dataset_accepts_records_and_reports_counts_and_digest():
    manifest = validate_dataset(_records())
    assert manifest["n_records"] == 12 and manifest["model_id"] == MODEL_ID
    assert manifest["text_chars"]["total"] == sum(len(LINES[i % len(LINES)]) for i in range(12))
    assert manifest["image_width"] == {"min": 640, "max": 640} and manifest["image_height"] == {"min": 64, "max": 64}
    assert manifest["digest"] == dataset_digest(_records()) and len(manifest["digest"]) == 64
    assert manifest["records"][0]["text"] == LINES[0]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.pop("text"), "missing 'text'"),
        (lambda r: r.update(id="bad id!"), "id must match"),
        (lambda r: r.update(image="nope.png"), "image file not found"),
        (lambda r: r.update(image=Image.new("RGB", (8, 8))), "MIN_IMAGE_SIDE"),
        (lambda r: r.update(text=""), "1..512"),
        (lambda r: r.update(text="   \n  "), "1..512"),
        (lambda r: r.update(text="x" * 513), "1..512"),
        (lambda r: r.update(text=123), "text must be a str"),
    ],
)
def test_validate_dataset_refuses_malformed_records(mutate, message):
    records = _records()
    mutate(records[3])
    with pytest.raises(ValueError, match=message):
        validate_dataset(records)


def test_validate_dataset_enforces_bounds_and_unique_ids():
    with pytest.raises(ValueError, match="8..5000"):
        validate_dataset(_records(7))
    with pytest.raises(ValueError, match="8..8 are required"):
        validate_dataset(_records(9), max_records=8)
    duplicated = [*_records(11), _record(0)]
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset(duplicated)
    with pytest.raises(ValueError, match="list of"):
        validate_dataset({"id": "x"})
    assert validate_dataset([{**_record(0), "text": "  a   b  "}], min_records=1)["records"][0]["text"] == "a b"


def test_validate_dataset_refuses_before_importing_model_libraries(forbid_model_imports):
    with pytest.raises(ValueError):
        validate_dataset(_records(3))
    validate_dataset(_records())


def test_digests_and_split_disjointness():
    records = _records(24)
    assert dataset_digest(records) == dataset_digest(list(reversed(records)))
    assert dataset_digest(records) != dataset_digest([{**records[0], "text": "other"}, *records[1:]])
    assert image_digest(records[0]["image"]) != image_digest(records[1]["image"])
    splits = split_dataset(records, seed=1)
    assert sum(len(v) for v in splits.values()) == 24 and all(splits.values())
    assert check_split_disjoint(splits) == {k: len(v) for k, v in splits.items()}
    leaked = {**splits, "test": [*splits["test"], splits["train"][0]]}
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint(leaked)
    duplicated = [*records, {**records[0], "id": "copy"}]
    assert sum(len(v) for v in split_dataset(duplicated, seed=1).values()) == 24
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.9)


# --- pinned corpus ----------------------------------------------------------------------------------------------


def test_pins_and_split_sizes():
    assert sorted(sm.ROW_GROUP_PINS) == list(range(sm.CORPUS_ROW_GROUPS)) == list(range(8))
    assert all(len(digest) == 64 and total > 4_500_000 for digest, total in sm.ROW_GROUP_PINS.values())
    assert sum(SAMPLE_SPLIT.values()) == 100 * sm.CORPUS_ROW_GROUPS == 800
    assert sm.CORPUS_URL.startswith(f"https://huggingface.co/datasets/{sm.CORPUS_REPO}/resolve/{sm.CORPUS_REVISION}/") and len(sm.CORPUS_REVISION) == 40
    assert "MIT" in sm.CORPUS_LICENSE and sm.CORPUS_LANGUAGE == "fr"


def _fake_rows(n=100):
    rows = []
    for i in range(n):
        buffer = io.BytesIO()
        _line(i, (320, 32)).save(buffer, format="JPEG")
        rows.append({"image": {"bytes": buffer.getvalue()}, "text": LINES[i % len(LINES)] if i % 17 else ""})
    return rows


def _fake_parquet(rows, path):
    table = pa.table({"image": pa.array([r["image"] for r in rows], type=pa.struct([("bytes", pa.binary())])), "text": pa.array([r["text"] for r in rows])})
    pq.write_table(table, path)


def test_fetch_corpus_refuses_a_row_group_that_does_not_match_its_pin(tmp_path, monkeypatch):
    shard = tmp_path / "shard.parquet"
    _fake_parquet(_fake_rows(), shard)
    with pytest.raises(ValueError, match="rows, pinned"):
        fetch_corpus(cache_dir=tmp_path / "cache", groups=[0], opener=lambda url: str(shard))
    with pytest.raises(ValueError, match="no pin"):
        fetch_corpus(cache_dir=tmp_path / "cache", groups=[99])
    monkeypatch.setattr(sm, "CORPUS_ROWS", 100)
    with pytest.raises(ValueError, match="sha256 .* != pinned"):
        fetch_corpus(cache_dir=tmp_path / "cache", groups=[0], opener=lambda url: str(shard))
    assert not list((tmp_path / "cache").iterdir())
    stale = tmp_path / "cache2"
    stale.mkdir()
    _fake_parquet(_fake_rows(), stale / "test-rg0.parquet")  # a cached file that fails its digest is refetched, not trusted
    with pytest.raises(ValueError, match="sha256 .* != pinned"):
        fetch_corpus(cache_dir=stale, groups=[0], opener=lambda url: str(shard))


def test_read_corpus_builds_records_from_verified_rows():
    rows = _fake_rows(50)
    records = read_corpus({3: [{"image": r["image"]["bytes"], "text": r["text"]} for r in rows]})
    assert len(records) == 50 - len([r for r in rows if not r["text"]]) and records[0]["id"] == "belfort-test-301" and records[0]["source_row_group"] == 3
    assert all(r["text"] == normalise_text(r["text"]) and r["image"].mode == "RGB" for r in records)
    splits = build_sample_dataset(records, sizes={"train": 30, "validation": 8, "test": 8})
    assert {k: len(v) for k, v in splits.items()} == {"train": 30, "validation": 8, "test": 8}
    assert build_sample_dataset(records, sizes={"train": 30, "validation": 8, "test": 8}) == splits
    assert check_split_disjoint(splits)
    with pytest.raises(ValueError, match="need"):
        build_sample_dataset(records)


def test_default_draw_matches_the_pinned_digest_when_the_row_groups_are_cached():
    cached = sm.DEFAULT_CACHE_DIR
    if not all((cached / f"test-rg{g}.parquet").is_file() for g in sm.ROW_GROUP_PINS):
        pytest.skip("Belfort row groups not cached")
    splits = build_sample_dataset(read_corpus(fetch_corpus(cache_dir=cached)))
    assert {k: len(v) for k, v in splits.items()} == SAMPLE_SPLIT
    assert check_split_disjoint(splits)
    assert dataset_digest([r for part in splits.values() for r in part]) == sm.SAMPLE_DIGEST


# --- BYOD ---------------------------------------------------------------------------------------------------------


def test_load_byod_dataset_reads_images_with_transcripts_from_a_zip_or_directory(tmp_path):
    rows = ["file,text,id"] + [f"line{i}.png,{LINES[i]},{'' if i % 2 else f'custom-{i}'}" for i in range(9)]
    folder = tmp_path / "byod"
    folder.mkdir()
    for i in range(9):
        _line(i).save(folder / f"line{i}.png")
    (folder / "transcripts.csv").write_text("\n".join(rows), encoding="utf-8")
    records = load_byod_dataset(folder)
    assert len(records) == 9 and records[0]["id"] == "custom-0" and records[1]["id"] == "line1" and records[2]["text"] == LINES[2]
    archive = tmp_path / "byod.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for file in folder.iterdir():
            zf.write(file, f"nested/{file.name}")
    assert [r["id"] for r in load_byod_dataset(archive)] == [r["id"] for r in records]
    assert validate_dataset(records)["n_records"] == 9
    (folder / "orphan.png").write_bytes((folder / "line0.png").read_bytes())
    with pytest.raises(ValueError, match="no transcripts.csv row"):
        load_byod_dataset(folder)
    (folder / "orphan.png").unlink()
    (folder / "transcripts.csv").write_text("file,text\nmissing.png,abc\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing image"):
        load_byod_dataset(folder)
    (folder / "transcripts.csv").unlink()
    with pytest.raises(ValueError, match="transcripts.csv"):
        load_byod_dataset(folder)
    with pytest.raises(ValueError, match="neither a directory nor a zip"):
        load_byod_dataset(tmp_path / "nothing.zip")
    out = write_dataset_csv(records, tmp_path / "out" / "train.csv")
    assert out.read_text(encoding="utf-8").startswith("id,file,width,height,chars,words,text,source_row_group")


# --- metrics and baselines ---------------------------------------------------------------------------------------


def test_edit_distance_and_ocr_metrics():
    assert edit_distance("kitten", "sitting") == 3 and edit_distance([], ["a"]) == 1 and edit_distance("abc", "abc") == 0
    records = [_record(0, "abcd"), _record(1, "hello world"), _record(2, "x")]
    result = ocr_metrics(["abcd", "hello there world", "yz"], records)
    assert result["n"] == 3 and result["char_edits"] == 0 + 6 + 2 and result["ref_chars"] == 16
    assert result["cer"] == pytest.approx(8 / 16) and result["cer_macro"] == pytest.approx((0 + 6 / 11 + 2) / 3)
    assert result["wer"] == pytest.approx(2 / 4) and result["exact_match"] == pytest.approx(1 / 3)
    assert result["rows"][2]["cer"] == 2.0  # over-generation: uncapped
    assert set(result["definitions"]) == {"cer", "wer", "cer_macro", "wer_macro", "exact_match"}
    with pytest.raises(ValueError, match="same length"):
        ocr_metrics(["a"], records)
    with pytest.raises(ValueError, match="empty reference"):
        ocr_metrics(["a"], [_record(0, " ")])


def test_baselines():
    records = _records(12)
    empty = empty_baseline(records)
    assert empty["cer"] == 1.0 and empty["wer"] == 1.0 and empty["exact_match"] == 0.0 and empty["baseline"].startswith("empty")
    train = [_record(i, t) for i, t in enumerate(["abc", "abd", "xyz", "abc"])]
    assert medoid_transcript(train) == "abc"
    constant = constant_baseline(train, records)
    assert constant["transcript"] == "abc" and constant["cer"] > 0.0 and constant["baseline"].endswith("'abc'")
    with pytest.raises(ValueError, match="no non-empty"):
        medoid_transcript([])


# --- evaluate with an injected runner ----------------------------------------------------------------------------


def test_evaluate_scores_hypotheses_from_the_runner_and_flags_truncation():
    records = _records(12)
    _echo_runner.table = {image_digest(r["image"]): r["text"] for r in records[:6]}
    pipe = GotOcr2Pipeline(_echo_runner, "cpu")
    result = pipe.evaluate(records, batch_size=5, max_new_tokens=5)
    assert result["n"] == 12 and result["exact_match"] == 0.5 and result["hypotheses"][:6] == [r["text"] for r in records[:6]]
    assert result["truncated"] == 12 and result["new_tokens"] == 60 and result["verdict"] == "measured-small-sample" and result["adapted"] is False
    assert result["model_revision"] == MODEL_REVISION and result["seconds"] >= 0.0
    with pytest.raises(ValueError, match="1..5000"):
        pipe.evaluate([])


def test_adapt_and_artifacts_require_a_loaded_model(forbid_model_imports):
    pipe = GotOcr2Pipeline(_echo_runner, "cpu")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.adapt(_records(), None)
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.save_artifact("x")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.load_artifact("x")


# --- artifact manifest checks --------------------------------------------------------------------------------------


def _manifest(tmp_path, **overrides):
    weights = tmp_path / "adapter.safetensors"
    weights.write_bytes(b"tensor-bytes")
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": "base-digest"},
        "adapter": {"best_epoch": 1},
        "tensors": ["model.language_model.layers.23.mlp.down_proj.weight", "model.language_model.norm.weight"],
        "files": [{"path": "adapter.safetensors", "bytes": weights.stat().st_size, "sha256": hashlib.sha256(b"tensor-bytes").hexdigest()}],
    }
    manifest.update(overrides)
    return manifest


def test_check_artifact_manifest_accepts_a_consistent_manifest_and_refuses_each_deviation(tmp_path):
    pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="format"):
        pl._check_artifact_manifest(_manifest(tmp_path, format="other"), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="trained on"):
        pl._check_artifact_manifest(_manifest(tmp_path, base={"model_id": "x", "revision": MODEL_REVISION, "weight_sha256": "base-digest"}), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="base weight digest"):
        pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "another-digest")
    bad = _manifest(tmp_path)
    bad["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        pl._check_artifact_manifest(bad, tmp_path, "base-digest")
    with pytest.raises(ValueError, match="exactly adapter.safetensors"):
        pl._check_artifact_manifest(_manifest(tmp_path, files=[]), tmp_path, "base-digest")
    for name in ("model.language_model.layers.19.mlp.down_proj.weight", "model.vision_tower.layers.0.x", "model.language_model.embed_tokens.weight", "lm_head.weight"):
        with pytest.raises(ValueError, match="last 4 decoder layers"):
            pl._check_artifact_manifest(_manifest(tmp_path, tensors=[name]), tmp_path, "base-digest")


def test_trainable_names_selects_the_last_layers_and_the_norm():
    class _Param:
        def numel(self):
            return 1

    class _Model:
        def named_parameters(self):
            names = ["model.vision_tower.layers.0.x", "model.multi_modal_projector.x", "model.language_model.embed_tokens.weight", "model.language_model.layers.0.x", "model.language_model.layers.19.x", "model.language_model.layers.20.x", "model.language_model.layers.23.mlp.up_proj.weight", "model.language_model.norm.weight", "lm_head.weight"]
            return [(n, _Param()) for n in names]

    assert pl._trainable_names(_Model()) == ["model.language_model.layers.20.x", "model.language_model.layers.23.mlp.up_proj.weight", "model.language_model.norm.weight"]
    assert pl._TRAINABLE_FIRST_LAYER == 20 and len(pl._TRAINABLE_PREFIXES) == 5


def test_manifest_json_round_trip(tmp_path):
    payload = {"epoch": 1, "train_loss": 3.1, "val": {"cer": 0.8, "wer": 1.1, "cer_macro": 0.9, "exact_match": 0.0, "n": 40}}
    (tmp_path / "h.json").write_text(json.dumps([payload]), encoding="utf-8")
    assert json.loads((tmp_path / "h.json").read_text(encoding="utf-8"))[0]["val"]["cer"] == 0.8
