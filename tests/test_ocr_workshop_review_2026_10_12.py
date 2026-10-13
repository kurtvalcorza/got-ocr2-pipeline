"""Review round 2026-10-12 checks for the OCR workshop (OCR2-m1..m11).

The kernel's own run_stage drives real stage processes of the carried file with this interpreter; the other
stages run in-process with the generation stand-ins of the earlier review tests. Offline checks only: not
real-model or hosted-runtime evidence.
"""

import json
import re
import subprocess
import sys

import pytest
from PIL import Image

from test_ocr_workshop_uv import STAGE_FILE, _kernel, _notebook
from test_workshop_optional_paths import setup as byod_setup
from test_workshop_optional_paths import stage_ns
from test_workshop_reference_structure import model_ns

CELLS = {c["id"]: "".join(c["source"]) for c in _notebook()["cells"]}
MARKDOWN = [(c["id"], "".join(c["source"])) for c in _notebook()["cells"] if c["cell_type"] == "markdown"]


def two_frame_tiff(root):
    root.mkdir()
    frames = [Image.new("RGB", (64, 64), "white"), Image.new("RGB", (64, 64), "black")]
    frames[0].save(root / "two_pages.tif", save_all=True, append_images=frames[1:])
    return root, "2 pages/frames"


def duplicate_header_csv(root):
    root.mkdir()
    Image.new("RGB", (64, 64), "white").save(root / "page.png")
    (root / "transcripts.csv").write_text("file,text,text\npage.png,a,b\n", encoding="utf-8")
    return root, "duplicate column names"


# ---- OCR2-m1: a refusal is one line, no traceback, no raised error; genuine failures still raise -----------
@pytest.mark.parametrize("make", [two_frame_tiff, duplicate_header_csv])
def test_refused_byod_input_prints_one_line_and_does_not_raise(tmp_path, capsys, make):
    ns = _kernel(tmp_path, [])
    root, reason = make(tmp_path / "inputs")
    ns.update(USE_BYOD=True, BYOD_PATH=str(root))
    assert ns["run_stage"]("byod") is False
    out = capsys.readouterr().out
    assert "Traceback" not in out and out.count("Refused:") == 1 and out.count(reason) == 1
    assert "run the BYOD cell again" in out
    assert not (tmp_path / "work" / "state" / "byod.json").exists()
    assert not (tmp_path / "outputs" / "byod").exists()  # refused before any model or output


def test_empty_byod_path_is_refused(tmp_path, capsys):
    ns = _kernel(tmp_path, [])
    ns.update(USE_BYOD=True, BYOD_PATH="  ")
    assert ns["run_stage"]("byod") is False
    out = capsys.readouterr().out
    assert out.count("Refused: BYOD_PATH is empty") == 1 and "Traceback" not in out


def test_stage_file_exits_65_for_a_refusal(tmp_path):
    root, _ = two_frame_tiff(tmp_path / "inputs")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"USE_BYOD": True, "BYOD_PATH": str(root), "WORK_DIR": str(tmp_path / "w"),
                                  "OUTPUT_DIR": str(tmp_path / "o")}), encoding="utf-8")
    done = subprocess.run([sys.executable, str(STAGE_FILE), "--config", str(config), "--stage", "byod"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 65 and done.stderr == ""
    assert done.stdout.count("\n") == 1 and done.stdout.startswith("Refused: two_pages.tif: 2 pages/frames")


def test_a_successful_stage_returns_true(tmp_path):
    ns = _kernel(tmp_path, [])
    assert ns["run_stage"]("byod") is True  # USE_BYOD is off by default: "BYOD disabled."


def test_genuine_byod_failure_still_raises(tmp_path):
    ns, _ = byod_setup(tmp_path)
    root = tmp_path / "inputs"
    root.mkdir()
    Image.new("RGB", (64, 64), "white").save(root / "page.png")

    def broken(records):
        raise ValueError("injected model failure")

    ns.update(USE_BYOD=True, BYOD_PATH=str(root), infer_byod=broken)
    config = tmp_path / "config.json"
    settings = {"USE_BYOD": True, "BYOD_PATH": str(root), "WORK_DIR": str(tmp_path / "work"),
                "OUTPUT_DIR": str(tmp_path / "outputs")}
    config.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="injected model failure"):
        ns["main"](["--config", str(config), "--stage", "byod"])


def test_input_refused_outside_byod_is_not_a_refusal(tmp_path):
    ns = stage_ns(tmp_path)

    def refusing():
        raise ns["InputRefused"]("not a BYOD stage")

    ns["STAGES"]["pages"] = refusing
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"WORK_DIR": str(tmp_path / "work")}), encoding="utf-8")
    with pytest.raises(ns["InputRefused"], match="not a BYOD stage"):
        ns["main"](["--config", str(config), "--stage", "pages"])
    assert issubclass(ns["InputRefused"], ValueError)


def test_byod_cell_clears_an_earlier_report_after_a_refusal():
    ns = {"run_stage": lambda stage: False, "USE_BYOD": True, "byod_report": {"images": 3}, "byod_dir": "old",
          "state": lambda name: pytest.fail("a refused run must not read BYOD state"), "Path": str}
    exec(CELLS["63fd18ce"], ns)
    assert ns["byod_report"] is None and ns["byod_dir"] is None


def test_summary_after_a_refused_byod_covers_the_canonical_path(tmp_path, capsys):
    ns, _ = byod_setup(tmp_path)
    ns["USE_BYOD"] = True
    ns["save_state"]("compare.json", {"belfort_metric_rows": [], "page_text_rows": []})
    ns["save_state"]("smol_pages.json", {"summaries": {}})
    ns["save_state"]("belfort.json", {"splits": {"test": []}})
    ns["save_state"]("pages.json", {})
    out_root = tmp_path / "outputs"
    out_root.mkdir(exist_ok=True)
    for name in ("belfort_results.csv", "belfort_metrics.csv", "page_text_metrics.csv",
                 "smoldocling_structure.csv", "token_budget.csv", "shape_robustness.csv",
                 "resource_metrics.csv", "got_format_outputs.json", "smoldocling_summary.json",
                 "specialized_instruction_outputs.json", "metrics.json", "provenance.json"):
        (out_root / name).write_text("x", encoding="utf-8")
    ns["stage_summary"]()
    out = capsys.readouterr().out
    assert "BYOD: no completed BYOD run in this session" in out and "Outputs:" in out


# ---- OCR2-m2: residue is disclosed and counted for display; scoring is unchanged -----------------------
def test_smoldocling_residue_is_disclosed_and_counted_without_rescoring(capsys):
    assert "Scoring note" in CELLS["8299d664"] and "<loc_5" in CELLS["8299d664"]
    assert "markup fragments" in CELLS["aaaebfe3"]
    items = [{"text": "abc <loc_5"}, {"text": "fine"}, {"text": "x </te"}]
    ns = {"re": re, "state": lambda name: {"smol_belfort.json": {"items": items},
                                           "belfort.json": {"splits": {"test": [{"text": "abcd"}]}}}[name],
          "run_stage": lambda stage: True}
    exec(CELLS["f0d3d353"], ns)
    out = capsys.readouterr().out
    assert "2 of 3" in out and "add 10 characters" in out and "against 4 reference characters" in out


# ---- OCR2-m3, m6, m7, m9, m11: learner-facing text --------------------------------------------------------
def _md(cid):
    return next(text for i, text in MARKDOWN if i == cid)


def test_metric_normalisation_and_line_budgets_are_stated():
    """Wording check: learner-facing text only, not behaviour."""
    section7 = _md("c96f1b20")
    assert "case kept" in section7 and "lower-casing" in section7 and "128" in section7 and "160" in section7
    assert "case-insensitive" in _md("guided-05")


def test_reading_order_text_does_not_infer_order_from_the_wer_gap():
    """Wording check (CX1, CX4): the WER vs token-F1 gap prompts inspection; it is not an order statistic."""
    section21 = _md("bcd85ff8")
    assert "Reading order" in section21 and "not a reading-order statistic" in section21
    assert "`a b a b`" in section21 and "repetition" in section21 and "omissions" in section21
    assert "means many of the right words came out in a different order" not in section21
    assert "not the WER alignment" in section21 and "names exactly" not in section21


def test_wer_gap_example_in_the_text_holds():
    """The counter-example quoted in Section 21 is computed with the stage file's own metrics."""
    import ocr_workshop as stages

    m = stages.one_metrics("a b", "a b a b")
    _, _, f1 = stages.token_overlap("a b", "a b a b")
    assert m["wer"] == 1.0 and abs(f1 - 2 / 3) < 1e-12


def test_expected_count_rule_names_formula_and_key_value_cases():
    """Wording check: learner-facing text only, not behaviour."""
    section8 = _md("c7241d47")
    assert "formula" in section8 and "key_value_region" in section8 and "in addition" in section8


def test_belfort_provenance_and_references():
    """Wording check: learner-facing text only, not behaviour."""
    text = "\n".join(t for _, t in MARKDOWN)
    assert "https://huggingface.co/datasets/Teklia/Belfort-line" in text and "MIT licence" in text
    assert "live carrier" not in text.lower() and "# References" in _md("guided-05")


def test_use_notice_precedes_the_first_code_cell():
    """Wording check: learner-facing text only, not behaviour."""
    cells = _notebook()["cells"]
    first_code = next(i for i, c in enumerate(cells) if c["cell_type"] == "code")
    before = "\n".join("".join(c["source"]) for c in cells[:first_code])
    assert "> **⚠️ Notice.**" in before and "Not validated for production" in before and '"as is"' in before


# ---- OCR2-m4, m5: guidance for the stage-process design and an activity that addresses unchanged rows ------
def test_no_pre_uv_guidance_remains():
    """Wording check: learner-facing text only, not behaviour."""
    text = "\n".join(t for _, t in MARKDOWN).lower()
    for phrase in ("while their respective models are loaded", "do not rerun released-model cells",
                   "without reloading a released model", "already imported in the hosted kernel",
                   "follow explicit unload/release steps"):
        assert phrase not in text


def test_budget_activity_contrasts_smoldocling_and_explains_identical_rows():
    """Wording check: learner-facing text only, not behaviour."""
    activity = _md("guided-01")
    assert "SmolDocling's report-page budget from 256 to 512" in activity and "identical" in activity


# ---- OCR2-m8: probe stages show what was generated on images without text --------------------------------
def test_blank_and_noise_outputs_are_displayed(tmp_path, capsys):
    ns, _ = model_ns(tmp_path, budget=False, shape=False)
    ns["stage_got_probes"]()
    ns["stage_smol_probes"]()
    out = capsys.readouterr().out
    for title in ("GOT-OCR 2.0 · blank probe", "GOT-OCR 2.0 · noise probe",
                  "SmolDocling · blank probe: extracted", "SmolDocling · noise probe: native DocTags"):
        assert title in out
    assert "GOTMARK-plain" in out and "SMOLMARK" in out and "<doctag>" in out


# ---- OCR2-m10: metadata declaration ---------------------------------------------------------------------
def test_metadata_declares_profile_mode_kind_and_one_spec_version():
    dimer = _notebook()["metadata"]["dimer"]
    assert dimer["notebook_profile"] == "TASK-INFERENCE" and dimer["notebook_mode"] == "WORKSHOP"
    assert dimer["notebook_kind"] == "WORKSHOP" and "profile" not in dimer
    assert dimer["notebook_spec"] == "2.5" and "**Notebook Spec:** `2.5`" in _md("18cbaa81")
    assert '"notebook_spec":"2.5"' in STAGE_FILE.read_text(encoding="utf-8")
    latest = dimer["review_revisions"][-1]
    assert latest["date"] == "2026-10-12" and latest["hosted_execution"] == "pending"


# ---- Codex peer review follow-up (CX2..CX7) -------------------------------------------------------------
def test_configured_pipeline_labels_and_no_unmeasured_qualifier():
    """Wording check (CX2, CX6): results are labelled end to end; the residue effect is not called slight."""
    section7 = _md("c96f1b20")
    assert "Configured-pipeline results" in section7 and "do not show whether these line budgets" in section7
    assert "slightly" not in _md("8299d664") and "configured-pipeline" in _md("aaaebfe3")
    assert "the 60 are not used" in _md("4426885d") and "pretraining data, is unknown" in _md("4426885d")


def _full_run(tmp_path):
    from test_workshop_optional_paths import stage_ns as fresh_ns

    ns = fresh_ns(tmp_path)

    class Model:
        def parameters(self):
            return [type("P", (), {"numel": lambda self: 4})()]

    def generate(*args):
        return [dict(text="PUBLIC NOTICE", doctags="<doctag><text>PUBLIC NOTICE</text></doctag>",
                     new_tokens=3, truncated=False, inference_seconds=0.1) for _ in args[4]]

    ns.update(RUN_BELFORT=False, load_got=lambda: (Model(), None, 0, 1, 0.5),
              load_smoldoc=lambda: (Model(), None, 1, 0, 0.25), got_generate=generate, smol_generate=generate,
              RUNTIME={"device": "cpu"})
    ns["save_state"]("snapshots.json", {"got": {"dir": "g", "verify_seconds": 1.0},
                                        "smoldocling": {"dir": "s", "verify_seconds": 2.0}})
    ns["save_state"]("belfort.json", {"digest": "d", "image_digests": {}, "splits": {"test": []}})
    ns["save_state"]("baselines.json", {"empty_metrics": None, "constant_metrics": None, "constant_text": ""})
    for stage in ("pages", "got-load", "got-belfort", "got-pages", "got-probes", "smol-load", "smol-belfort",
                  "smol-pages", "smol-specialized", "smol-probes", "compare", "resources"):
        ns["STAGES"][stage]()
    return ns


def test_export_refuses_a_comparison_older_than_its_inputs(tmp_path):
    ns = _full_run(tmp_path)
    ns["STAGES"]["export"]()  # a straight run exports
    ns["STAGES"]["got-pages"]()  # a canonical step re-run with a different result
    rows = ns["load_state"]("got_pages.json")
    rows["rows"][0]["cer"] = 0.5
    ns["save_state"]("got_pages.json", rows)
    stale = r"comparison \(Section 20\) was computed before got_pages.json changed"
    with pytest.raises(RuntimeError, match=stale):
        ns["STAGES"]["export"]()
    ns["STAGES"]["compare"]()
    stale = r"resource summary \(Section 22\) was computed before got_pages.json"
    with pytest.raises(RuntimeError, match=stale):
        ns["STAGES"]["export"]()
    ns["STAGES"]["resources"]()
    ns["STAGES"]["export"]()  # both summaries refreshed: the export runs again


def test_probe_rerun_needs_only_the_export(tmp_path):
    ns = _full_run(tmp_path)
    ns["STAGES"]["got-probes"]()
    ns["STAGES"]["export"]()


def test_display_labels_for_structure_resources_and_word_diff(tmp_path, capsys):
    ns = _full_run(tmp_path)
    out = capsys.readouterr().out
    assert "authored count discrepancy" in out and "raw tag/token counts" in out and "abs err" not in out
    assert "amortised_s_per_line (batched Belfort run)" in out
    assert "peak_allocated_gpu_bytes (Belfort line inference only)" in out
    ns["stage_inspect"]("notice")
    assert "case-sensitive illustrative text diff; not the WER alignment" in capsys.readouterr().out
    rows = ns["load_state"]("resources.json")
    kept = {"mean_line_latency_s", "mean_page_latency_s", "peak_gpu_memory_bytes"}  # export keys unchanged
    assert kept <= set(rows[0])


# ---- Owner decision OCR2-CX2 = C: markup-exclusion sensitivity ------------------------------------------
@pytest.mark.parametrize(("text", "kept", "removed"), [
    ("abc <loc_5", "abc", "<loc_5"),
    ("x </te", "x", "</te"),
    ("abc <loc_", "abc", "<loc_"),
    ("word <", "word", "<"),
    ("<loc_12", "", "<loc_12"),
    ("clean text", "clean text", ""),
    ("a <b> c", "a <b> c", ""),
    ("2 < 3", "2 < 3", ""),
])
def test_strip_partial_markup_rule(text, kept, removed):
    import ocr_workshop as stages

    assert stages.strip_partial_markup(text) == (kept, removed)


@pytest.mark.parametrize("doctags", ["<text><loc_10><loc_20><loc_30><loc_5", "abc <loc_",
                                     "<doctag><text><loc_1><loc_2><loc_3><loc_4>word</te"])
def test_truncated_doctags_leave_no_markup_after_the_cleanup(doctags):
    """The OCR2-m2 cases, through the usual extraction and then the declared rule."""
    import ocr_workshop as stages

    cleaned, _ = stages.strip_partial_markup(stages.doctags_to_text(doctags))
    assert "<" not in cleaned and "loc_" not in cleaned


def _belfort_compare_ns(tmp_path):
    ns = stage_ns(tmp_path)
    test = [{"id": "l1", "text": "Le conseil"}, {"id": "l2", "text": "municipal"}]
    ns["save_state"]("belfort.json", {"splits": {"test": test}})
    empty = ns["corpus_metrics"](["", ""], test)
    baselines = {"empty_metrics": empty, "constant_metrics": empty, "constant_text": ""}
    ns["save_state"]("baselines.json", baselines)
    got = [dict(text="Le conseil", new_tokens=3, truncated=False, inference_seconds=0.1),
            dict(text="municipal <", new_tokens=128, truncated=True, inference_seconds=0.1)]
    box = "<loc_1><loc_2><loc_3><loc_4>"
    smol = [dict(text="Le conseil", doctags=f"<text>{box}Le conseil</text>", new_tokens=9,
                 truncated=False, inference_seconds=0.1),
            dict(text="municip <loc_5", doctags=f"<text>{box}municip<loc_5", new_tokens=160,
                 truncated=True, inference_seconds=0.1)]
    for name, items in (("got_belfort.json", got), ("smol_belfort.json", smol)):
        metrics = ns["corpus_metrics"]([x["text"] for x in items], test)
        metrics.update(truncated=sum(x["truncated"] for x in items), empty_hypotheses=0)
        ns["save_state"](name, {"items": items, "metrics": metrics, "seconds": 0.2, "peak_gpu": None})
    for name in ("got_pages.json", "smol_pages.json"):
        ns["save_state"](name, {"rows": []})
    return ns, test, got, smol


def test_sensitivity_table_leaves_the_configured_rows_unchanged(tmp_path, capsys):
    ns, test, got, smol = _belfort_compare_ns(tmp_path)
    ns["stage_compare"]()
    out = capsys.readouterr().out
    compare = ns["load_state"]("compare.json")
    by_system = {r["system"]: r for r in compare["belfort_metric_rows"]}
    for system, items in (("GOT-OCR 2.0", got), ("SmolDocling", smol)):
        expected = ns["corpus_metrics"]([x["text"] for x in items], test)  # raw extracted text, as before
        assert by_system[system]["cer"] == expected["cer"] and by_system[system]["wer"] == expected["wer"]
    rows = {r["model"]: r for r in compare["markup_sensitivity"]}
    for model in ("GOT-OCR 2.0", "SmolDocling"):
        assert rows[model]["configured_cer"] == by_system[model]["cer"]
    assert rows["SmolDocling"]["markup_excluded_cer"] < rows["SmolDocling"]["configured_cer"]
    assert rows["SmolDocling"]["lines_with_partial_tag"] == 1
    assert rows["SmolDocling"]["partial_tag_chars"] == 6
    assert rows["SmolDocling"]["lines_unbalanced_element_tags"] == 1
    assert rows["SmolDocling"]["lines_malformed_locations"] == 0
    assert rows["GOT-OCR 2.0"]["lines_with_partial_tag"] == 1  # the same rule, applied symmetrically
    assert rows["GOT-OCR 2.0"]["lines_unbalanced_element_tags"] is None
    assert "Markup-exclusion sensitivity" in out and "primary" in out
    assert [x["text"] for x in ns["load_state"]("smol_belfort.json")["items"]] == [x["text"] for x in smol]


def test_sensitivity_is_exported_next_to_the_configured_metrics(tmp_path):
    ns = _full_run(tmp_path)
    ns["STAGES"]["export"]()
    exported = tmp_path / "outputs" / "belfort_markup_sensitivity.csv"
    header = exported.read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",") == ns["MARKUP_SENSITIVITY_FIELDS"]


def test_sensitivity_wording():
    """Wording check (OCR2-CX2 = C): the table is described as a sensitivity next to the primary result."""
    assert "Markup-exclusion sensitivity (owner decision OCR2-CX2)" in _md("aaaebfe3")
    assert "never repaired" in _md("aaaebfe3") and "belfort_markup_sensitivity.csv" in _md("3ee52ee6")
