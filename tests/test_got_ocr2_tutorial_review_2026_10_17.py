"""Regression tests for the row 30 tutorial review (2026-10-17): GOT2-M1 (pre-registered grid, fresh test), GOT2-M2
(the CER reading), GOT2-m1..m8. CPU only; nothing here loads the pinned checkpoint. The causal-mask parity test builds a
tiny randomly initialised Qwen2 decoder, which is interface evidence, not pretrained-model evidence."""
# ruff: noqa: E501  -- assertion messages and notebook source fragments are kept on one line

from __future__ import annotations

import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import got_ocr2_pipeline as got
from got_ocr2_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "got_ocr2_colab.ipynb"
PREREG = ROOT / "docs" / "recipe-grid-preregistration.md"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _markdown() -> str:
    return "\n".join(_src(c) for c in _nb()["cells"] if c["cell_type"] == "markdown")


def _code_after(heading: str) -> str:
    cells = _nb()["cells"]
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and heading in _src(cell):
            return next(_src(c) for c in cells[i + 1 :] if c["cell_type"] == "code")
    raise AssertionError(heading)


def _line(k: int) -> Image.Image:
    image = Image.new("RGB", (40 + k, 16), (255, 255 - k % 200, 255))
    ImageDraw.Draw(image).rectangle([2, 2, 4 + k % 30, 12], fill=(k % 255, 0, 0))
    return image


# --- GOT2-m4: the cached tail runs under an explicit causal mask ---------------------------------------------------


def _tiny_decoder(attn: str):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, attn_implementation=attn)
    torch.manual_seed(0)
    return torch, transformers.Qwen2Model(config).eval()


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_tail_hidden_states_match_the_full_decoder_up_to_float32_rounding(attn):
    torch, model = _tiny_decoder(attn)
    assert got.decoder_attention_implementation(model) == attn
    embeds = torch.randn(3, 11, 32)
    with torch.no_grad():
        full = model(inputs_embeds=embeds, output_hidden_states=True)
        tail = got.tail_hidden_states(model, 2, full.hidden_states[2])
        delta = (model.norm(tail) - full.last_hidden_state).abs().max().item()
    assert delta < 1e-5, f"{attn}: tail differs from the full decoder by {delta}"


def test_the_parity_test_detects_future_token_leakage():
    """The reviewer's failure mode: eager attention with attention_mask=None lets positions read later tokens."""
    torch, model = _tiny_decoder("eager")
    embeds = torch.randn(3, 11, 32)
    with torch.no_grad():
        full = model(inputs_embeds=embeds, output_hidden_states=True)
        hidden = full.hidden_states[2]
        positions = torch.arange(hidden.shape[1]).unsqueeze(0).expand(hidden.shape[0], -1)
        rope = model.rotary_emb(hidden, positions)
        for layer in model.layers[2:]:
            out = layer(hidden, attention_mask=None, position_ids=positions, position_embeddings=rope)
            hidden = out[0] if isinstance(out, tuple) else out
        leaked = (model.norm(hidden) - full.last_hidden_state).abs().max().item()
    assert leaked > 1e-3, "an unmasked eager tail should differ from the full decoder"


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_padded_batch_tail_loss_matches_the_full_model_masked_loss(attn):
    """Review S4: right-padded lines with -100 labels, as training batches them from the cache."""
    torch, model = _tiny_decoder(attn)
    head = torch.nn.Linear(32, 64, bias=False)
    lengths = [11, 7, 4]
    embeds = torch.randn(3, 11, 32)
    labels = torch.full((3, 11), -100)
    for i, n in enumerate(lengths):
        embeds[i, n:] = 0.0
        labels[i, 2:n] = torch.randint(0, 64, (n - 2,))
    with torch.no_grad():
        full_sum, total = 0.0, 0
        for i, n in enumerate(lengths):  # each line alone, unpadded, through the full decoder
            out = model(inputs_embeds=embeds[i : i + 1, :n])
            logits = head(out.last_hidden_state[0, :-1])
            target = labels[i, 1:n]
            keep = target != -100
            full_sum += torch.nn.functional.cross_entropy(logits[keep], target[keep], reduction="sum").item()
            total += int(keep.sum())
        hidden = model(inputs_embeds=embeds, output_hidden_states=True).hidden_states[2]
        tail = got.tail_hidden_states(model, 2, hidden)
        targets = labels[:, 1:]
        keep = targets != -100
        loss = torch.nn.functional.cross_entropy(head(model.norm(tail[:, :-1][keep])), targets[keep]).item()
    assert abs(loss - full_sum / total) < 1e-5


def test_an_unverified_attention_implementation_is_refused():
    class Config:
        _attn_implementation = "flash_attention_2"

    class Decoder:
        config = Config()

    with pytest.raises(RuntimeError, match="not one the cached-tail loss is verified for"):
        got.check_tail_attention(Decoder())
    source = (ROOT / "src" / "got_ocr2_pipeline" / "pipeline.py").read_text(encoding="utf-8")
    assert "attention_mask=None, position_ids" not in source and "loss equals the full model's loss exactly" not in source
    assert "up to float32 rounding" in source and "min_gain=" not in _code_after("## 6. Baselines")


# --- GOT2-M1: the pre-registered grid --------------------------------------------------------------------------------


def _result(curve, best, lr):
    return {"lr": lr, "best_epoch": best, "history": [{"epoch": e, "val": {"cer": c}} for e, c in enumerate(curve)]}


def test_select_recipe_keeps_g1_unless_beaten_by_more_than_the_margin():
    g1 = _result([1.55, 0.9, 0.70, 0.66], 3, 5e-5)
    near = _result([1.55, 0.8, 0.62, 0.7], 2, 1e-4)  # 0.04 better: not enough
    worse = _result([1.55, 1.0, 0.9, 0.8], 3, 2.5e-5)
    choice = got.select_recipe({"G1": g1, "G2": near, "G3": worse})
    assert choice["chosen"] == "G1" and "is kept" in choice["reason"]
    assert choice["validation_cer_by_epoch"]["G2"] == [1.55, 0.8, 0.62, 0.7]
    far = _result([1.55, 0.7, 0.55, 0.6], 2, 1e-4)  # 0.11 better
    farther = _result([1.55, 0.7, 0.52, 0.6], 2, 2.5e-5)
    assert got.select_recipe({"G1": g1, "G2": far, "G3": worse})["chosen"] == "G2"
    assert got.select_recipe({"G1": g1, "G2": far, "G3": farther})["chosen"] == "G3"


def test_select_recipe_returns_the_frozen_model_when_no_config_clears_epoch_0():
    flat = {name: _result([1.0, 0.98, 0.97], 0, lr) for name, lr in got.RECIPE_GRID.items()}
    choice = got.select_recipe(flat)
    assert choice["chosen"] == "frozen" and "frozen model" in choice["reason"]
    with pytest.raises(ValueError, match="validation only"):
        got.select_recipe({"G1": {"lr": 5e-5, "best_epoch": 0, "history": [{"epoch": 0, "val": None}]}})


def test_grid_constants_match_the_preregistration_committed_before_any_run():
    assert got.RECIPE_GRID == {"G1": 5e-5, "G2": 1e-4, "G3": 2.5e-5} and got.RECIPE_DEFAULT == "G1" and got.RECIPE_MARGIN == 0.05
    text = PREREG.read_text(encoding="utf-8")
    for needle in ("| G1 | 5e-5 |", "| G2 | 1e-4 |", "| G3 | 2.5e-5 |", "by more than 0.05", "before any grid run", sm.FRESH_TEST_DIGEST, "fix/got-ocr2-tutorial-review-2026-10-17-gridrun"):
        assert needle in text, needle
    sig = __import__("inspect").signature(got.GotOcr2Pipeline.adapt)
    assert sig.parameters["min_gain"].default == 0.0


def _records(prefix: str, n: int) -> list[dict]:
    return [{"id": f"{prefix}{k}", "text": f"ligne {prefix} {k % 7}"} for k in range(n)]


def _harness(tmp_path, monkeypatch, *, grid=False, form=None, data=None, ns=None, calls=None, sections=(7, 8)):
    """Execute Sections 7 and 8 verbatim with a stand-in pipeline that records which record sets it scores. `form`
    replaces `# @param` literals of Section 7 (e.g. {"EPOCHS": "10"}); `data` overrides Section 4/6 names."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir(exist_ok=True)
    curves = {5e-5: [1.5, 0.9, 0.7], 1e-4: [1.5, 0.8, 0.6], 2.5e-5: [1.5, 1.1, 0.95]}
    calls = [] if calls is None else calls
    fresh, dev, val, train = _records("f", 200), _records("d", 140), _records("v", 60), _records("t", 600)

    class FakePipe:
        adapter = None

        def adapt(self, train, val, *, epochs, lr, batch_size, max_new_tokens, progress, min_gain):
            calls.append(("adapt", lr, min_gain))
            curve = curves.get(lr, curves[5e-5])
            history = [{"epoch": e, "train_loss": None if e == 0 else 1.0, "val": {"cer": curve[e], "wer": 1.0, "cer_macro": 0.5, "exact_match": 0.0}} for e in range(3)]
            for entry in history:
                progress(entry)
            self.adapter = {"best_epoch": 2}
            return {"lr": lr, "best_epoch": 2, "history": history, "n_trainable": 1, "n_total": 2, "first_trainable_layer": 20, "seed": 0, "selection": "lowest validation CER", "attention": "sdpa", "loss": "x", "cache_seconds": 0.0, "min_gain": min_gain}

        def evaluate(self, records, *, max_new_tokens, batch_size):
            calls.append(("evaluate", records[0]["id"][0], len(records)))
            out = got.ocr_metrics(["ligne"] * len(records), records)
            out.update({"hypotheses": ["ligne"] * len(records), "truncated": 0, "adapted": self.adapter is not None, "verdict": "measured"})
            return out

        def save_artifact(self, path, metadata=None):
            calls.append(("save", Path(path).name))
            Path(path).mkdir(parents=True, exist_ok=True)

        def load_artifact(self, path):
            calls.append(("load", Path(path).name))

    if ns is None:
        ns = {name: getattr(got, name) for name in got.__all__}
        pipe = FakePipe()

        def reset():
            calls.append(("reset",))
            pipe.adapter = None

        frozen_dev = got.ocr_metrics(["zz"] * len(dev), dev)
        frozen_dev.update({"hypotheses": ["zz"] * len(dev), "truncated": 0, "adapted": False})
        ns.update({"os": os, "Path": Path, "json": json, "time": __import__("time"), "shutil": __import__("shutil"), "pipe": pipe, "METRICS": ("cer", "wer", "cer_macro", "exact_match"), "LINE_MAX_NEW_TOKENS": 128, "NOTEBOOK_SOURCE": {"repository_revision": "x"}, "require_valid_run": lambda section: None, "reset_to_pretrained": reset,
                   "data_label": "sample", "SPLIT_SEED": 42, "USE_BYOD": False, "train_records": train, "val_records": val, "test_records": fresh, "dev_test_records": dev, "dataset_manifests": {"fresh_test": {"digest": got.FRESH_TEST_DIGEST}, "train": {"digest": "t"}}, "data_source": "sample", "disjoint": {}, "fresh_report": {"n": 200},
                   "overlap": {"validation": [], "test": ["f3"], "development_test": ["d1"]}, "shown_records": dev, "shown_on": "exposed development test", "empty_shown": got.empty_baseline(dev), "constant_shown": got.constant_baseline(train, dev), "frozen_shown": frozen_dev})
        ns.update(data or {})
    for section in sections:
        source = _code_after({7: "## 7. Bounded fine-tuning", 8: "## 8. Held-out evaluation"}[section])
        if section == 7:
            assert "RUN_RECIPE_GRID = False  # @param" in source
            if grid:
                source = source.replace("RUN_RECIPE_GRID = False", "RUN_RECIPE_GRID = True", 1)
            for name, value in (form or {}).items():
                source, n = re.subn(rf"^{name} = [^ ]+  # @param", f"{name} = {value}  # @param", source, count=1, flags=re.M)
                assert n == 1, name
        exec(compile(source, f"<section {section}>", "exec"), ns)
    return ns, calls


def _scored_fresh(calls):
    return [c for c in calls if c[0] == "evaluate" and c[1] == "f"]


def test_section7_default_path_trains_g1_only(tmp_path, monkeypatch):
    ns, calls = _harness(tmp_path, monkeypatch, sections=(7,))
    assert [c for c in calls if c[0] == "adapt"] == [("adapt", 5e-5, 0.05)]
    assert ns["preregistered"] and ns["setting_differences"] == [] and ns["chosen_config"] == "G1" and ns["recipe_grid"] is None
    assert len(_scored_fresh(calls)) == 1  # the frozen model, before adapting
    assert not (tmp_path / "outputs" / "got_ocr2_recipe_grid.json").exists()


def test_section7_grid_trains_three_configs_from_pretrained_and_keeps_the_chosen_adapter(tmp_path, monkeypatch, capsys):
    ns, calls = _harness(tmp_path, monkeypatch, grid=True)
    assert [c[1] for c in calls if c[0] == "adapt"] == [5e-5, 1e-4, 2.5e-5]
    assert calls.count(("reset",)) == 4  # before each config and before loading the chosen adapter
    assert ns["chosen_config"] == "G2" and ("load", "G2") in calls and ns["adapt_result"]["lr"] == 1e-4
    table = json.loads((tmp_path / "outputs" / "got_ocr2_recipe_grid.json").read_text(encoding="utf-8"))
    assert table["chosen"] == "G2" and set(table["validation_cer_by_epoch"]) == {"G1", "G2", "G3"}
    assert table["fresh_test_scored"] is False and table["preregistered_fixed_settings"] is True
    # Score-once rule item 4: the selection-only grid run never scores the fresh test.
    assert _scored_fresh(calls) == [] and ns["test_look"] == "not scored (exploratory)"
    assert "G2" in capsys.readouterr().out


@pytest.mark.parametrize(("form", "data", "changed"), [
    ({"EPOCHS": "10"}, None, "epochs"),
    ({"BATCH_SIZE": "4"}, None, "batch_size"),
    ({"LEARNING_RATE": "1e-4"}, None, "learning_rate"),
    (None, {"SPLIT_SEED": 7}, "split_seed"),
    (None, {"data_label": "BYOD", "USE_BYOD": True}, "data"),
    (None, {"LINE_MAX_NEW_TOKENS": 64}, "line_max_new_tokens"),
])
def test_any_changed_setting_never_scores_the_fresh_test(tmp_path, monkeypatch, form, data, changed):
    ns, calls = _harness(tmp_path, monkeypatch, form=form, data=data)
    assert not ns["preregistered"] and any(d.startswith(changed) for d in ns["setting_differences"])
    assert _scored_fresh(calls) == [] and ns["test_look"] == "not scored (exploratory)"
    assert ns["chosen_config"].startswith("exploratory")
    assert (tmp_path / "outputs" / "got_ocr2_evaluation_report_exploratory_1.json").is_file()
    assert not (tmp_path / "outputs" / "got_ocr2_evaluation_report.json").exists()


@pytest.mark.parametrize("recorded", [None, got.RECORDED_EVALUATION])
def test_identical_rerun_is_labelled_a_reproduction_and_keeps_the_first_result(tmp_path, monkeypatch, recorded):
    # Rule item 8: without a recorded evaluation, the first pre-registered pass is "pre-registered evaluation (this
    # runtime)"; once RECORDED_EVALUATION names the evaluation of record, every later identical pass is a reproduction.
    ns, calls = _harness(tmp_path, monkeypatch, data={"RECORDED_EVALUATION": recorded})
    first_look = "pre-registered evaluation (this runtime)" if recorded is None else "reproduction"
    assert ns["test_look"] == first_look and ns["scored_on"].startswith("fresh test")
    first = (tmp_path / "outputs" / "got_ocr2_evaluation_report.json").read_text(encoding="utf-8")
    ns, calls = _harness(tmp_path, monkeypatch, ns=ns, calls=calls)  # same settings, same runtime
    assert ns["test_look"] == "reproduction"
    assert (tmp_path / "outputs" / "got_ocr2_evaluation_report.json").read_text(encoding="utf-8") == first
    assert (tmp_path / "outputs" / "got_ocr2_evaluation_report_reproduction_1.json").is_file()
    history = json.loads((tmp_path / "outputs" / "got_ocr2_run_history.json").read_text(encoding="utf-8"))
    assert [h["test_look"] for h in history] == [first_look, "reproduction"]
    report = json.loads(first)
    assert report["preregistered"] is True and report["settings_differ"] == [] and "bootstrap_95" in report
    assert {"adapted_unseen_transcripts", "adapted_seen_transcripts"} <= set(report["transcript_breakdown"])


def test_no_permanent_scored_once_claims():
    nb_text = NOTEBOOK.read_text(encoding="utf-8")
    assert "scored once)" not in nb_text and "reported result (scored once)" not in nb_text
    assert "a page-disjoint split is impossible" not in _markdown() and "That is the claim" not in _markdown()
    assert "test_examples" not in _code_after("## 4. Belfort lines")  # review m9
    # score-once rule item 6: no permanent "scored/reported once" prose anywhere a learner or a reader of the record sees
    once = re.compile(r"(scored|scores|score|reported|reports)( on it| the fresh test| the exposed development test)? once", re.IGNORECASE)
    root = NOTEBOOK.parents[1]
    for path in (NOTEBOOK, root / "README.md", root / "tutorials" / "README.md", root / "MODEL_CARD.md", root / "STATUS.md", *sorted((root / "src").rglob("*.py"))):
        hits = once.findall(path.read_text(encoding="utf-8"))
        assert not hits, (path.name, hits)


def test_bootstrap_interval_is_deterministic_and_paired():
    rows = [{"id": k, "char_edits": k % 3, "ref_chars": 4, "word_edits": 1, "ref_words": 2} for k in range(40)]
    other = [{**r, "char_edits": 2} for r in rows]
    a = got.bootstrap_interval(rows, seed=1)
    assert a == got.bootstrap_interval(rows, seed=1) and a["low"] <= a["estimate"] <= a["high"]
    d = got.bootstrap_interval(rows, other, measure="cer")
    assert d["paired"] and d["low"] <= d["estimate"] <= d["high"]
    with pytest.raises(ValueError, match="same lines"):
        got.bootstrap_interval(rows, other[::-1])


def test_preregistration_differences_and_folded_overlap():
    assert got.preregistration_differences(got.PREREGISTERED_SETTINGS) == []
    assert got.preregistration_differences({**got.PREREGISTERED_SETTINGS, "epochs": 10}) == ["epochs: 10 (pre-registered 6)"]
    assert got.preregistration_differences({**got.PREREGISTERED_SETTINGS, "learning_rate": 1e-4}, ignore=("learning_rate",)) == []
    train = [{"id": "a", "text": "Le Conseil municipal."}]
    other = [{"id": "x", "text": "le conseil Municipal"}]
    assert got.transcript_overlap(train, other) == [] and got.transcript_overlap(train, other, fold=True) == ["x"]
    assert got.normalise_text("été") == "été"  # NFC (review m3)


# --- GOT2-M1: the fresh test ----------------------------------------------------------------------------------------


def test_fresh_test_pins_are_rows_groups_8_and_9_of_the_same_shard():
    assert sm.FRESH_TEST_ROW_GROUPS == (8, 9) and set(sm.FRESH_TEST_ROW_GROUPS) <= set(sm.ROW_GROUP_PINS)
    assert not set(sm.FRESH_TEST_ROW_GROUPS) & set(sm.SAMPLE_ROW_GROUPS)
    assert sm.FRESH_TEST_IDS[0] == "belfort-test-800" and sm.FRESH_TEST_IDS[-1] == "belfort-test-999" and len(set(sm.FRESH_TEST_IDS)) == 200
    assert sm.ROW_GROUP_PINS[8] == ("f4036cc009855fb2e710fe55b5222b5a016b039a3b8aaf31d32126768ffc3b71", 5_000_587)
    assert sm.ROW_GROUP_PINS[9] == ("d26c4fa972ddcef0cb2152432ee6821f69cd9498cb140edff2baac4094013869", 4_683_815)


def _fresh_like(monkeypatch):
    records = [{"id": rid, "image": _line(k), "text": f"ligne {k}", "source_row_group": 8 + k // 100} for k, rid in enumerate(sm.FRESH_TEST_IDS)]
    monkeypatch.setattr(sm, "FRESH_TEST_DIGEST", sm.dataset_digest(records))
    return records


def test_build_fresh_test_refuses_wrong_ids_digest_and_any_overlap(monkeypatch):
    records = _fresh_like(monkeypatch)
    other = [{"id": f"belfort-test-{k}", "image": _line(300 + k), "text": "ligne 3", "source_row_group": 0} for k in range(5)]
    fresh, report = sm.build_fresh_test(records, {"train": other, "validation": [], "test": []})
    assert len(fresh) == 200 and report["train"]["shared_images"] == 0 and report["train"]["transcripts_also_in_train"] == 1
    with pytest.raises(ValueError, match="expected the 200 pinned ids"):
        sm.build_fresh_test(records[:-1], {"train": other})
    with pytest.raises(ValueError, match="overlaps the train split"):
        sm.build_fresh_test(records, {"train": [*other, {"id": "x", "image": records[5]["image"], "text": "t", "source_row_group": 0}]})
    with pytest.raises(ValueError, match="overlaps the test split"):
        sm.build_fresh_test(records, {"test": [{"id": "y", "image": _line(999), "text": "t", "source_row_group": 9}]})
    altered = [dict(r) for r in records]
    altered[0]["text"] = "changed"
    with pytest.raises(ValueError, match="dataset digest"):
        sm.build_fresh_test(altered, {"train": other})


def test_real_fresh_test_is_disjoint_when_the_row_groups_are_cached():
    cache = Path(os.environ.get("GOT_OCR2_BELFORT_CACHE", sm.DEFAULT_CACHE_DIR))
    if not all((cache / f"test-rg{g}.parquet").is_file() for g in sm.ROW_GROUP_PINS):
        pytest.skip("Belfort row groups 0-9 not cached (set GOT_OCR2_BELFORT_CACHE)")
    splits = sm.build_sample_dataset(sm.read_corpus(sm.fetch_corpus(cache_dir=cache)))
    fresh, report = sm.build_fresh_test(sm.read_corpus(sm.fetch_corpus(cache_dir=cache, groups=sm.FRESH_TEST_ROW_GROUPS)), splits)
    assert len(fresh) == 200 and all(report[name]["shared_images"] == 0 for name in ("train", "validation", "test"))


# --- GOT2-M2, m1, m2, m3, m5, m8: learner-facing text ------------------------------------------------------------


def test_cer_is_read_as_edits_per_reference_character():
    markdown = _markdown()
    assert not re.search(r"three characters in four|\b(two|three|four|\d+) (characters )?in (three|four|five|\d+)\b[^.]{0,40}right", markdown) and "**edits per reference character**" in markdown
    assert "4,680 character edits" in markdown and "6,169 reference" in markdown and "6,618 characters" in markdown
    assert "'char_edits': frozen_test['char_edits']" in _code_after("## 6. Baselines")


def test_selection_is_on_validation_and_the_fresh_test_is_scored_only_under_preregistered_settings():
    markdown = _markdown()
    for stale in ("moved the held-out number", "1.302", "on its own 80-line test split", "test CER still ended at 1.040", "build record"):
        assert stale not in markdown, stale
    assert re.search(r"(chosen|selected) (among|from) .{0,80}validation", markdown.lower())
    assert "with the fresh test scored after the choice and only under the pre-registered settings" in markdown and "**exposed development test**" in markdown
    assert "fix/got-ocr2-tutorial-review-2026-10-17-gridrun" in markdown
    # REL13 requote (2026-10-17): the grid table and the fresh-test numbers are the recorded runs, not "pending".
    assert "pending" not in markdown.replace("Owner decisions pending", "")
    assert "G1 0.634" in markdown and "G2 0.643" in markdown and "G3 0.653" in markdown and "`9ec5eb71`" in markdown
    assert "1.157 → **0.902**" in markdown and "-0.509 to -0.002" in markdown and "-0.761 to 0.135" in markdown
    assert "`59bb22a6`" in markdown and "1213.2 s" in markdown and "966 s in the recorded run" not in markdown


def test_recorded_evaluation_names_the_hosted_run_of_record():
    record = got.RECORDED_EVALUATION
    assert record["notebook_blob"].startswith("59bb22a6") and record["commit"].startswith("f564d9e")
    assert (ROOT / record["evidence"]).is_file()


def test_quotes_name_their_blob_and_overclaims_are_gone():
    markdown = _markdown()
    assert "previous notebook version" not in markdown and "reading cursive French" not in markdown
    assert "more than half the characters right" not in markdown
    assert markdown.count("blob `cbe04372`") >= 5
    assert re.search(r"overlap[^.]{0,160}cannot be ruled out", markdown)


def test_spec_kind_notice_references_and_environment_variables():
    nb = _nb()
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    header = "\n".join(_src(c) for c in nb["cells"][:first_code])
    assert "> **⚠️ Notice.** For research, training and evaluation only." in header and "**Kind:** `TUTORIAL`" in header
    markdown = _markdown()
    cited = set(re.findall(r"\[(\d)\]", markdown.split("## References")[0]))
    listed = set(re.findall(r"^\[(\d)\] ", markdown.split("## References")[1], re.M))
    assert cited == listed == {"1", "2", "3", "4"}
    assert "`DIMER_ISOLATED_ENV`" in markdown and "DIMER_NOTEBOOK_CI_PREINSTALLED" in markdown


# --- GOT2-m7: a refused BYOD input prints one message and invalidates the run -------------------------------------


def _zip(tmp_path: Path, n: int) -> Path:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        rows = ["file,text"]
        for i in range(n):
            png = io.BytesIO()
            _line(i).save(png, format="PNG")
            archive.writestr(f"l{i:03d}.png", png.getvalue())
            rows.append(f"l{i:03d}.png,ligne {i}")
        archive.writestr("transcripts.csv", "\n".join(rows) + "\n")
    path = tmp_path / f"byod{n}.zip"
    path.write_bytes(buffer.getvalue())
    return path


def _section4(tmp_path, monkeypatch, byod_path):
    monkeypatch.chdir(tmp_path)
    source = _code_after("## 4. Belfort lines").replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", f"BYOD_PATH = {str(byod_path)!r}", 1)
    ns = {name: getattr(got, name) for name in got.__all__}
    ns.update({"os": os, "Path": Path, "data_source": "the earlier sample run"})
    exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def test_refusal_keeps_earlier_outputs_marks_them_and_stops_later_sections(tmp_path, monkeypatch, capsys):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "got_ocr2_result.json").write_text("{}", encoding="utf-8")
    ns = _section4(tmp_path, monkeypatch, _zip(tmp_path, 5))
    out = capsys.readouterr().out
    assert out.count("BYOD refused:") == 1 and "Traceback" not in out and "Runtime -> Run after" in out
    assert (tmp_path / "outputs" / "got_ocr2_result.json").is_file()  # earlier outputs are kept ...
    marker = json.loads((tmp_path / "outputs" / "got_ocr2_run_invalidated.json").read_text(encoding="utf-8"))
    assert "got_ocr2_result.json" in marker["files_in_outputs"] and marker["files_come_from"] == "the earlier sample run"  # ... and marked
    with pytest.raises(ns["RunInvalidated"]) as stopped:
        ns["require_valid_run"]("Section 6")
    assert stopped.value._render_traceback_() == [str(stopped.value)] and stopped.value.dimer_stop is True
    assert "Section 6 did not run: Section 4 refused the BYOD input" in str(stopped.value)
    ok = _section4(tmp_path, monkeypatch, _zip(tmp_path, 12))  # a corrected input clears the invalidation
    assert ok["RUN_STATE"]["valid"] and not (tmp_path / "outputs" / "got_ocr2_run_invalidated.json").exists()


def test_an_unexpected_error_keeps_its_traceback(tmp_path, monkeypatch):
    def broken(path):
        raise OSError("disk failure")

    monkeypatch.setattr(got, "load_byod_dataset", broken)
    with pytest.raises(OSError, match="disk failure"):
        _section4(tmp_path, monkeypatch, _zip(tmp_path, 12))


@pytest.mark.skipif(os.name == "nt", reason="the isolated worker passes pipe fds (POSIX only)")
def test_the_router_shows_a_deliberate_stop_without_a_traceback(capfd):
    build = _load_tool("build_notebook")
    router = build._ISOLATED_ROUTER
    ns: dict = {"os": os, "subprocess": subprocess, "sys": sys}
    exec(compile(router[: router.index("if SKIP_INSTALL:")], "<router>", "exec"), ns)
    runtime = ns["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        with pytest.raises(ns["IsolatedCellStop"], match="refused here") as stopped:
            runtime.run("class Stop(RuntimeError):\n    dimer_stop = True\nraise Stop('refused here')\n")
        assert stopped.value._render_traceback_() == ["refused here"]
        assert "Traceback" not in capfd.readouterr().err
        with pytest.raises(ns["IsolatedCellError"]):
            runtime.run("raise ValueError('genuine')\n")
        assert "Traceback" in capfd.readouterr().err
    finally:
        runtime.close()


# --- GOT2-m6: the provenance stamp contains the carried code ---------------------------------------------------------


def test_generated_from_revision_contains_the_carried_modules():
    build = _load_tool("build_notebook")
    nb = _nb()
    revision = nb["metadata"]["dimer"]["generated_from"]["revision"]
    have = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{revision}^{{commit}}"], capture_output=True)
    if have.returncode != 0:
        pytest.skip(f"revision {revision[:12]} is not in this clone (shallow checkout)")
    template = _load_tool("notebook_template").TEMPLATE
    rels = nb["metadata"]["dimer"]["generated_from"]["modules"]
    texts = {Path(rel).name: subprocess.check_output(["git", "-C", str(ROOT), "show", f"{revision}:{rel}"]).decode("utf-8") for rel in rels}
    embedded = build.apply_rewrites(texts, template.get("rewrites"))
    cells = {c["metadata"]["dimer"]["embedded_module"]: build.strip_carried_title(_src(c)) for c in nb["cells"] if "dimer" in c.get("metadata", {})}
    for rel in rels:
        assert cells[rel] == embedded[Path(rel).name].rstrip("\n"), f"{rel} at {revision[:12]} differs from its carried cell"


def test_the_generator_refuses_uncommitted_carried_modules(tmp_path):
    build = _load_tool("build_notebook")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "m.py").write_text("X = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "m.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "m"], check=True)
    ctx = {"module_rels": ["m.py"], "modules": ["m.py"], "texts": {"m.py": "X = 1\n"}}
    assert build.uncommitted_modules(tmp_path, ctx) == []
    ctx["texts"]["m.py"] = "X = 2\n"
    assert build.uncommitted_modules(tmp_path, ctx) == ["m.py"]
