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
    section7 = _md("c96f1b20")
    assert "case kept" in section7 and "lower-casing" in section7 and "128" in section7 and "160" in section7
    assert "case-insensitive" in _md("guided-05")


def test_reading_order_is_explained_where_token_f1_is_shown():
    section21 = _md("bcd85ff8")
    assert "Reading order" in section21 and "token-F1" in section21 and "INSPECT_PAGE" in section21


def test_expected_count_rule_names_formula_and_key_value_cases():
    section8 = _md("c7241d47")
    assert "formula" in section8 and "key_value_region" in section8 and "in addition" in section8


def test_belfort_provenance_and_references():
    text = "\n".join(t for _, t in MARKDOWN)
    assert "https://huggingface.co/datasets/Teklia/Belfort-line" in text and "MIT licence" in text
    assert "live carrier" not in text.lower() and "# References" in _md("guided-05")


def test_use_notice_precedes_the_first_code_cell():
    cells = _notebook()["cells"]
    first_code = next(i for i, c in enumerate(cells) if c["cell_type"] == "code")
    before = "\n".join("".join(c["source"]) for c in cells[:first_code])
    assert "> **⚠️ Notice.**" in before and "Not validated for production" in before and '"as is"' in before


# ---- OCR2-m4, m5: guidance for the stage-process design and an activity that addresses unchanged rows ------
def test_no_pre_uv_guidance_remains():
    text = "\n".join(t for _, t in MARKDOWN).lower()
    for phrase in ("while their respective models are loaded", "do not rerun released-model cells",
                   "without reloading a released model", "already imported in the hosted kernel",
                   "follow explicit unload/release steps"):
        assert phrase not in text


def test_budget_activity_contrasts_smoldocling_and_explains_identical_rows():
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
