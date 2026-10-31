"""DAT20: a refused BYOD input in Section 4 shows exactly one line, ``Refused: <file>: <reason>``, with no traceback,
and the cell raises so Run all stops there (before this change it printed and carried on). The run is still
invalidated in the cell. A genuine failure keeps its traceback. Section 4 is executed from the committed notebook."""
# ruff: noqa: E501

from __future__ import annotations

import json

import pytest

import got_ocr2_pipeline as got
from test_got_ocr2_colab_review_fixes import _code_after, _section4, _zip

REFUSAL_CELL = "got_ocr2-17"


def _line(ns: dict) -> str:
    refusal = ns.get("_refusal")
    assert refusal is not None, "Section 4 did not raise a refusal"
    assert type(refusal).__name__ == "InputRefused" and refusal.dimer_stop is True and isinstance(refusal, ValueError)
    rendered = refusal._render_traceback_()
    assert len(rendered) == 1 and "\n" not in rendered[0] and "Traceback" not in rendered[0]
    assert refusal.__suppress_context__
    return rendered[0]


def test_the_scripted_refusal_input_is_one_line_naming_the_file(monkeypatch, tmp_path, capsys):
    hostname = tmp_path / "hostname"
    hostname.write_text("runtime-host\n", encoding="utf-8")
    ns = _section4(monkeypatch, tmp_path, byod_path=str(hostname))
    line = _line(ns)
    assert line.startswith("Refused: hostname: ") and "Runtime -> Run after" in line
    out = capsys.readouterr().out
    assert out.count("Refused:") == 1 and "BYOD refused" not in out and "Traceback" not in out
    assert ns["RUN_STATE"]["valid"] is False


def test_a_directory_and_a_cancelled_upload_are_refused_by_name(monkeypatch, tmp_path):
    folder = tmp_path / "etc"
    folder.mkdir()
    (folder / "hostname").write_text("runtime-host\n", encoding="utf-8")
    assert _line(_section4(monkeypatch, tmp_path, byod_path=str(folder))).startswith("Refused: etc: ")
    assert _line(_section4(monkeypatch, tmp_path, byod_path=str(tmp_path / "missing.zip"))).startswith("Refused: missing.zip: BYOD_PATH = ")


def test_the_in_cell_invalidation_is_kept(monkeypatch, tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "got_ocr2_result.json").write_text("{}", encoding="utf-8")
    ns = _section4(monkeypatch, tmp_path, byod_path=str(_zip(tmp_path, "eleven", 11)))
    assert _line(ns).startswith("Refused: eleven.zip: ")
    assert ns["train_records"] is None and ns["test_records"] is None
    marker = json.loads((tmp_path / "outputs" / "got_ocr2_run_invalidated.json").read_text(encoding="utf-8"))
    assert "got_ocr2_result.json" in marker["files_in_outputs"]


def test_a_genuine_failure_keeps_its_type(monkeypatch, tmp_path):
    def broken(path):
        raise RuntimeError("internal bug")

    monkeypatch.setattr(got, "load_byod_dataset", broken)
    with pytest.raises(RuntimeError, match="internal bug"):
        _section4(monkeypatch, tmp_path, byod_path=str(_zip(tmp_path, "twelve", 12)))


def test_the_refusal_cell_id_and_the_stop_channel():
    from test_got_ocr2_colab_review_fixes import NOTEBOOK

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c.get("id") == REFUSAL_CELL)
    source = "".join(cell["source"])
    assert source == _code_after("## 4. Belfort lines, the transcripts and the split") and "BYOD_PATH = ''" in source
    assert "raise InputRefused(f'Refused: {byod_label}: {reason}." in source and "print(f'BYOD refused" not in source
    router = "".join(c for c in ("".join(x["source"]) for x in notebook["cells"] if x["cell_type"] == "code") if "class IsolatedCellStop" in c)
    assert 'if getattr(exc, "dimer_stop", False):' in router and 'elif kind == "stop":' in router
