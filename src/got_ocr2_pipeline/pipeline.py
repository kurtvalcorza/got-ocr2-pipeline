"""Optical character recognition with the pinned ``stepfun-ai/GOT-OCR-2.0-hf`` checkpoint (GOT-OCR 2.0).

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the GOT-OCR2 architecture comes from the pinned ``transformers`` release,
the weights are SafeTensors, and no model-repository code is executed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"
MODEL_REVISION = "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "got-ocr-2.0-hf"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# The two recognition modes the pinned README documents: "plain" (the processor's default prompt,
# plain text) and "format" (processor(..., format=True): formatted text such as Markdown or LaTeX for
# tables and formulas). Region ("box"/"color") and multi-page modes are not exposed.
MODES = ("plain", "format")
DEFAULT_MODE = MODES[0]
# Generation ceilings. 4096 is the max_new_tokens the pinned README's examples pass; the default is a
# practical single-page budget.
MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 1024
DECODING = "greedy"
STOP_STRING = "<|im_end|>"
# Input ceilings. The processor resizes every image to 1024x1024 (preprocessor_config.json, aspect
# ratio not preserved) into 576 image tokens, so image cost is bounded; the side ceiling only guards
# memory during decoding and resizing.
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate of ``hypothesis`` against ``reference`` after lower-casing and whitespace tokenisation.

    Levenshtein edits over words divided by reference words; punctuation is **not** stripped, so a
    stray comma counts. The metric a caller would use to score ``doctags_to_text`` against a known page.
    """
    ref, hyp = _tokens(reference), _tokens(hypothesis)
    if not ref:
        raise ValueError("reference must contain at least one word")
    previous = list(range(len(hyp) + 1))
    for row_index, ref_token in enumerate(ref, 1):
        current = [row_index]
        for column_index, hyp_token in enumerate(hyp, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1] / len(ref)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Character error rate of ``hypothesis`` against ``reference`` after whitespace normalisation.

    Levenshtein edits over characters divided by reference characters; case and punctuation are kept.
    The metric a caller would use to score OCR output against a known transcript.
    """
    ref, hyp = " ".join(reference.split()), " ".join(hypothesis.split())
    if not ref:
        raise ValueError("reference must contain at least one character")
    previous = list(range(len(hyp) + 1))
    for row_index, ref_char in enumerate(ref, 1):
        current = [row_index]
        for column_index, hyp_char in enumerate(hyp, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1] / len(ref)


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one image as PIL.Image.Image (any mode, converted to RGB): a page, a scene with text, a crop",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "modes": list(MODES),
    "max_new_tokens": [1, MAX_NEW_TOKENS],
    "decoding": f"{DECODING} (do_sample=False) until {STOP_STRING} or the budget; deterministic per device",
    "preprocessing": (
        "image converted to RGB; the processor resizes to 1024x1024 (aspect ratio not preserved, CLIP "
        "mean/std) into 576 image tokens and wraps them in the snapshot's chat prompt for the chosen mode"
    ),
    "output": "the recognised text (plain, or formatted Markdown/LaTeX in format mode); no score",
}


def _check_inputs(image: Any, mode: Any, max_new_tokens: Any) -> tuple[Image.Image, str, int]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``recognize`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge.
    """
    rgb = validate_image(image)
    if not isinstance(mode, str):
        raise TypeError("mode must be a str")
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of MODES {MODES}")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an int")
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
    return rgb, mode, max_new_tokens


def validate_inputs(
    image: Image.Image,
    *,
    mode: str = DEFAULT_MODE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Rejection is reported by raising exactly as ``recognize`` would; a caller that wants the finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    _rgb, checked_mode, checked_tokens = _check_inputs(image, mode, max_new_tokens)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (recognize takes one image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "image-0", "mode": image.mode, "size": list(image.size)}],
        "recognition_mode": checked_mode,
        "generation": {"max_new_tokens": checked_tokens, "do_sample": False, "decoding": DECODING},
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any],
    reference_text: str | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``reference_text`` (the transcript the image really carries) the report carries
    ``character_error_rate`` and ``word_error_rate`` of the recognised text against it, verdict
    ``sample-sanity``; without it the verdict is ``not-measurable`` and the report says what labelled
    data would make the task measurable.
    """
    text = str(result["text"])
    base = {
        "task": "optical character recognition: image -> text",
        "score_semantics": (
            "generated text carries no score, no probability and no correctness signal; fluent output is "
            "not evidence that it matches the image. Greedy decoding makes the output reproducible on a "
            "fixed device and dtype, which is a reproducibility property, not a quality one"
        ),
        "recognition_mode": result.get("mode"),
        "sample_kind": sample_kind,
        "n_chars": len(text),
        "truncated": result.get("truncated"),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if not reference_text:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no reference transcript was supplied for the evaluated image",
            "needs": (
                "images with ground-truth transcripts from the deployment domain (scans, scenes, fonts, "
                "languages) scored with character and word error rate; no such labelled set ships with this "
                "repository"
            ),
        }
    metrics = [
        {
            "id": "character_error_rate",
            "value": character_error_rate(reference_text, text),
            "normalisation": "whitespace collapsed; case and punctuation kept",
            "estimation": "one image, no dispersion estimate",
        },
        {
            "id": "word_error_rate",
            "value": word_error_rate(reference_text, text),
            "normalisation": "lower-cased, whitespace-tokenised; punctuation kept",
            "estimation": "one image, no dispersion estimate",
        },
    ]
    return {
        **base,
        "metrics": metrics,
        "verdict": "sample-sanity",
        "reason": (
            "2 sanity measures on one tutorial image whose text you rendered yourself; plumbing evidence, "
            "not an OCR benchmark"
        ),
        "needs": (
            "a labelled image set from the deployment domain (scans, photographs, fonts, scripts, layouts) "
            "for any OCR accuracy claim"
        ),
    }


@dataclass
class GotOcr2Pipeline:
    """``_runner(image, mode, max_new_tokens)`` returns ``{"text": str, "new_tokens": int}``."""

    _runner: Callable[..., dict[str, Any]]
    device: str = "cpu"
    dtype: str = "float32"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> GotOcr2Pipeline:
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        common: dict[str, Any] = {"trust_remote_code": False}
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, common["local_files_only"], source = str(root), True, "local-snapshot"
        elif allow_download:
            location, common["revision"], source = MODEL_ID, MODEL_REVISION, "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if resolved_device.startswith("cuda") else torch.float32
        processor = AutoProcessor.from_pretrained(location, **common)
        model = AutoModelForImageTextToText.from_pretrained(location, dtype=dtype, **common)
        model = model.eval().to(resolved_device)

        def runner(image: Image.Image, mode: str, max_new_tokens: int) -> dict[str, Any]:
            inputs = processor(image, return_tensors="pt", format=(mode == "format")).to(resolved_device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=processor.tokenizer,
                    stop_strings=STOP_STRING,
                    max_new_tokens=max_new_tokens,
                )
            new_ids = generated[0, inputs["input_ids"].shape[1] :]
            decoded = processor.decode(new_ids, skip_special_tokens=True)
            return {"text": decoded, "new_tokens": int(new_ids.shape[0])}

        return cls(runner, resolved_device, str(dtype).removeprefix("torch."), source)

    def recognize(
        self,
        image: Image.Image,
        *,
        mode: str = DEFAULT_MODE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> dict[str, Any]:
        """Recognise the text in one image; ``text`` is the decoded output with the stop string removed."""
        rgb, checked_mode, checked_tokens = _check_inputs(image, mode, max_new_tokens)
        raw = self._runner(rgb, checked_mode, checked_tokens)
        if not isinstance(raw, dict) or "text" not in raw:
            raise RuntimeError("runner must return a dict with 'text'")
        text = str(raw["text"]).replace(STOP_STRING, "").strip()
        new_tokens = int(raw.get("new_tokens", 0))
        return {
            "text": text,
            "mode": checked_mode,
            "image_size": list(rgb.size),
            "new_tokens": new_tokens,
            "truncated": new_tokens >= checked_tokens,
            "generation": {"max_new_tokens": checked_tokens, "do_sample": False, "decoding": DECODING},
            "device": self.device,
            "dtype": self.dtype,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
