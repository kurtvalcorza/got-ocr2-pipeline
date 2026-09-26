"""Reproduce Colab's already imported NumPy without model installation."""
import ast
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def test_colab_preloaded_numpy_survives_setup(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "tutorials" / "DIMER_OCR_Document_Extraction_Workshop.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    tree = ast.parse("".join(notebook["cells"][7]["source"]))
    prefix = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(a.name == "numpy" for a in node.names):
            break
        prefix.append(node)
    pin_node = next(n for n in prefix if isinstance(n, ast.Assign)
                    and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "PINS")
    pins = ast.literal_eval(pin_node.value)
    installed = dict(pins) if isinstance(pins, dict) else dict(
        p.split("==", 1) for p in pins if "==" in p
    )
    installed["numpy"] = "2.1.3"
    monkeypatch.setattr(importlib.metadata, "version", lambda name: installed[name])
    for name in ("torch", "torchvision", "torchaudio", "transformers", "PIL", "scipy",
                 "safetensors", "huggingface_hub", "pyarrow"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    loaded_numpy = SimpleNamespace(__version__="2.1.3")
    monkeypatch.setitem(sys.modules, "numpy", loaded_numpy)
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)

    def install(command, **kwargs):
        for item in command:
            if "==" in item:
                name, version = item.split("==", 1)
                installed[name] = version
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", install)
    monkeypatch.setattr(subprocess, "check_call", lambda command, **kw: install(command, **kw).returncode)
    exec(compile(ast.Module(body=prefix, type_ignores=[]), "notebook-setup", "exec"), {})
    assert installed["numpy"] == loaded_numpy.__version__ == "2.1.3"
    assert sys.modules["numpy"] is loaded_numpy
