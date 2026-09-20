"""Optical character recognition with the pinned ``stepfun-ai/GOT-OCR-2.0-hf`` checkpoint (GOT-OCR 2.0), plus the
adaptation contract: corpus-level evaluation on labelled text lines, bounded fine-tuning of the last decoder layers,
and a verified adapter artifact.

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the GOT-OCR2 architecture comes from the pinned ``transformers`` release,
the weights are SafeTensors, and no model-repository code is executed.
"""
# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width

from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "stepfun-ai/GOT-OCR-2.0-hf"
MODEL_REVISION = "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "got-ocr-2.0-hf"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHT_FILE = "model.safetensors"

# The two recognition modes the pinned README documents: "plain" (the processor's default prompt,
# plain text) and "format" (processor(..., format=True): formatted text such as Markdown or LaTeX for
# tables and formulas). Region ("box"/"color") and multi-page modes are not exposed.
MODES = ("plain", "format")
DEFAULT_MODE = MODES[0]
# Generation ceilings. 4096 is the max_new_tokens the pinned README's examples pass; the default is a
# practical single-page budget; the line budget is what the corpus stages use (a transcribed line of
# up to MAX_TEXT_CHARS characters needs far fewer tokens, and a runaway generation stops there).
MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_LINE_MAX_NEW_TOKENS = 128
DECODING = "greedy"
STOP_STRING = "<|im_end|>"
# Input ceilings. The processor resizes every image to 1024x1024 (preprocessor_config.json, aspect
# ratio not preserved) into 256 image tokens, so model cost is bounded whatever the input size; the
# side and pixel ceilings only guard memory while decoding and resizing (a 9,000 px wide text line is
# fine, a 4096x4096 page is the largest area accepted).
MAX_IMAGE_SIDE = 16_384
MAX_IMAGE_PIXELS = 4096 * 4096
MIN_IMAGE_SIDE = 16
IMAGE_TOKENS = 256  # what the processor emits per image (the config's image_seq_length 576 is not used)
PLAIN_PROMPT_TOKENS = 286  # 256 image tokens + 30 chat-template tokens; identical for every image
# Model facts (measured on the pinned snapshot; tests pin them).
PARAMETER_COUNT = 560_528_640
DECODER_LAYERS = 24
TRAINABLE_LAYERS = 4  # the last decoder layers + the final norm are the adapter
ADAPTER_PARAMETERS = 51_401_728
# Adaptation contract.
ARTIFACT_FORMAT = f"org.valcorza.{MODEL_KEY}.adapter.v1"
ARTIFACT_VERSION = "1.0"
ADAPTER_WEIGHTS = "adapter.safetensors"
ADAPTER_MANIFEST = "manifest.json"
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
MAX_EVAL_RECORDS = 5_000
EVAL_BATCH_SIZE = 8
CACHE_BATCH_SIZE = 4  # images per frozen-prefix forward while caching hidden states
# The vision tower is a SAM-style ViT with eager attention over 4,096 patch tokens: its global-attention layers
# materialise one 4096x4096 float32 matrix per head per image (about 0.8 GB), so images go through it one at a time
# whatever the batch size; the decoder still runs the batch at once.
VISION_BATCH_SIZE = 1
GRAD_CLIP = 1.0
_TRAINABLE_FIRST_LAYER = DECODER_LAYERS - TRAINABLE_LAYERS
_TRAINABLE_PREFIXES = tuple(f"model.language_model.layers.{i}." for i in range(_TRAINABLE_FIRST_LAYER, DECODER_LAYERS)) + ("model.language_model.norm.",)


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


def _weight_digest(root: Path) -> str | None:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    with open(manifest_path, encoding="utf-8") as handle:
        entries = json.load(handle).get("files", [])
    return next((e["sha256"] for e in entries if e["path"] == WEIGHT_FILE), None)


# ---------------------------------------------------------------------------------------------------------
# Text measures
# ---------------------------------------------------------------------------------------------------------


def normalise_text(text: str) -> str:
    """The transcript form every comparison uses: whitespace runs collapsed to one space, ends stripped."""
    return " ".join(str(text).split())


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Levenshtein distance (insertions + deletions + substitutions, unit cost) between two sequences."""
    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_item in enumerate(reference, 1):
        current = [row_index]
        for column_index, hyp_item in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def _tokens(text: str) -> list[str]:
    return text.lower().split()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word error rate of ``hypothesis`` against ``reference`` after lower-casing and whitespace tokenisation.

    Levenshtein edits over words divided by reference words; punctuation is **not** stripped, so a
    stray comma counts. Not capped: a hypothesis longer than the reference can score above 1.0.
    """
    ref, hyp = _tokens(reference), _tokens(hypothesis)
    if not ref:
        raise ValueError("reference must contain at least one word")
    return edit_distance(ref, hyp) / len(ref)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Character error rate of ``hypothesis`` against ``reference`` after whitespace normalisation.

    Levenshtein edits over characters divided by reference characters; case and punctuation are kept.
    Not capped: a hypothesis longer than the reference can score above 1.0.
    """
    ref, hyp = normalise_text(reference), normalise_text(hypothesis)
    if not ref:
        raise ValueError("reference must contain at least one character")
    return edit_distance(ref, hyp) / len(ref)


# ---------------------------------------------------------------------------------------------------------
# Input contract
# ---------------------------------------------------------------------------------------------------------


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"image area {width * height} px > MAX_IMAGE_PIXELS {MAX_IMAGE_PIXELS}")
    return image.convert("RGB")


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one image as PIL.Image.Image (any mode, converted to RGB): a page, a text line, a scene with text, a crop",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "image_pixels_max": MAX_IMAGE_PIXELS,
    "modes": list(MODES),
    "max_new_tokens": [1, MAX_NEW_TOKENS],
    "decoding": f"{DECODING} (do_sample=False) until {STOP_STRING} or the budget; deterministic per device",
    "preprocessing": (
        "image converted to RGB; the processor resizes to 1024x1024 (aspect ratio not preserved, CLIP "
        f"mean/std) into {IMAGE_TOKENS} image tokens and wraps them in the snapshot's chat prompt for the chosen mode"
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
    """Evaluation stage for one image: a machine-readable report even when nothing is measurable.

    With ``reference_text`` (the transcript the image really carries) the report carries
    ``character_error_rate`` and ``word_error_rate`` of the recognised text against it, verdict
    ``sample-sanity``; without it the verdict is ``not-measurable`` and the report says what labelled
    data would make the task measurable. Corpus-level measurement is ``GotOcr2Pipeline.evaluate``.
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
                "languages) scored with character and word error rate over a labelled set; see evaluate"
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


# ---------------------------------------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------------------------------------


def _trainable_names(model: Any) -> list[str]:
    """The last `TRAINABLE_LAYERS` decoder layers and the final norm; the vision tower, the projector, the
    embeddings (tied to the output head) and the earlier decoder layers stay frozen."""
    return [name for name, _ in model.named_parameters() if name.startswith(_TRAINABLE_PREFIXES)]


def _check_artifact_manifest(manifest: Mapping[str, Any], artifact_dir: Path, base_sha256: str) -> None:
    """Refuse an adapter that names another base, another format or a file that does not match its digest."""
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
    base = manifest.get("base", {})
    if base.get("model_id") != MODEL_ID or base.get("revision") != MODEL_REVISION:
        raise ValueError(f"artifact was trained on {base.get('model_id')}@{base.get('revision')}, not {MODEL_ID}@{MODEL_REVISION}")
    if base.get("weight_sha256") != base_sha256:
        raise ValueError("artifact base weight digest does not match the verified snapshot")
    files = manifest.get("files") or []
    if len(files) != 1 or files[0].get("path") != ADAPTER_WEIGHTS:
        raise ValueError(f"artifact manifest must list exactly {ADAPTER_WEIGHTS}")
    weights = artifact_dir / ADAPTER_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"artifact weights missing: {weights}")
    size = weights.stat().st_size
    if size != files[0].get("bytes"):
        raise ValueError(f"{ADAPTER_WEIGHTS}: size {size} != manifest {files[0].get('bytes')}")
    digest = _sha256(weights)
    if digest != files[0].get("sha256"):
        raise ValueError(f"{ADAPTER_WEIGHTS}: sha256 {digest} != manifest {files[0].get('sha256')}")
    names = manifest.get("tensors") or []
    if not names or any(not str(n).startswith(_TRAINABLE_PREFIXES) for n in names):
        raise ValueError(f"artifact tensors must all belong to the last {TRAINABLE_LAYERS} decoder layers or the final norm")


@dataclass
class GotOcr2Pipeline:
    """``_runner(images, mode, max_new_tokens)`` returns one ``{"text": str, "new_tokens": int}`` per image."""

    _runner: Callable[..., list[dict[str, Any]]]
    device: str = "cpu"
    dtype: str = "float32"
    source: str = "injected"
    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)
    weight_sha256: str | None = None
    adapter: dict[str, Any] | None = None

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
        # float32 on every device: the adapter is trained in float32 and overlays without a cast, and
        # CPU, Tesla-class and consumer GPUs then run the same arithmetic (up to kernel rounding).
        dtype = torch.float32
        processor = AutoProcessor.from_pretrained(location, **common)
        model = AutoModelForImageTextToText.from_pretrained(location, dtype=dtype, **common)
        model = model.eval().to(resolved_device)
        for param in model.parameters():
            param.requires_grad_(False)
        pad_id = processor.tokenizer.pad_token_id
        stop_id = processor.tokenizer.convert_tokens_to_ids(STOP_STRING)
        vision_features = model.model.get_image_features

        def chunked_image_features(pixel_values: Any, **kwargs: Any) -> Any:
            """Bound the vision tower's attention memory: `VISION_BATCH_SIZE` images per forward, results concatenated."""
            if pixel_values.shape[0] <= VISION_BATCH_SIZE:
                return vision_features(pixel_values=pixel_values, **kwargs)
            chunks = [vision_features(pixel_values=pixel_values[i : i + VISION_BATCH_SIZE], **kwargs) for i in range(0, pixel_values.shape[0], VISION_BATCH_SIZE)]
            return torch.cat(chunks, dim=0)

        model.model.get_image_features = chunked_image_features

        def runner(images: Sequence[Image.Image], mode: str, max_new_tokens: int) -> list[dict[str, Any]]:
            inputs = processor(list(images), return_tensors="pt", padding=True, format=(mode == "format"))
            if not bool(inputs["attention_mask"].all()):
                raise RuntimeError("prompts in one batch differ in length; batched generation needs identical prompts")
            inputs = inputs.to(resolved_device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=processor.tokenizer,
                    stop_strings=STOP_STRING,
                    max_new_tokens=max_new_tokens,
                )
            prompt_len = int(inputs["input_ids"].shape[1])
            out = []
            for row in generated:
                new_ids = row[prompt_len:]
                ids = new_ids.tolist()
                # generate pads finished sequences with the pad token; the stop token itself counts as generated
                n_new = len(ids)
                for position, token in enumerate(ids):
                    if token in (stop_id, pad_id):
                        n_new = position + (token == stop_id)
                        break
                out.append({"text": processor.decode(new_ids[:n_new], skip_special_tokens=True), "new_tokens": int(n_new)})
            return out

        return cls(runner, resolved_device, str(dtype).removeprefix("torch."), source, model, processor, _weight_digest(root))

    # ------------------------------------------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------------------------------------------

    def _generate(self, images: Sequence[Image.Image], mode: str, max_new_tokens: int) -> list[dict[str, Any]]:
        raw = self._runner(list(images), mode, max_new_tokens)
        if not isinstance(raw, list) or len(raw) != len(images) or any(not isinstance(r, dict) or "text" not in r for r in raw):
            raise RuntimeError("runner must return one dict with 'text' per image")
        out = []
        for item in raw:
            new_tokens = int(item.get("new_tokens", 0))
            out.append({"text": str(item["text"]).replace(STOP_STRING, "").strip(), "new_tokens": new_tokens, "truncated": new_tokens >= max_new_tokens})
        return out

    def recognize(
        self,
        image: Image.Image,
        *,
        mode: str = DEFAULT_MODE,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> dict[str, Any]:
        """Recognise the text in one image; ``text`` is the decoded output with the stop string removed."""
        rgb, checked_mode, checked_tokens = _check_inputs(image, mode, max_new_tokens)
        item = self._generate([rgb], checked_mode, checked_tokens)[0]
        return {
            "text": item["text"],
            "mode": checked_mode,
            "image_size": list(rgb.size),
            "new_tokens": item["new_tokens"],
            "truncated": item["truncated"],
            "generation": {"max_new_tokens": checked_tokens, "do_sample": False, "decoding": DECODING},
            "device": self.device,
            "dtype": self.dtype,
            "source": self.source,
            "adapted": self.adapter is not None,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def transcribe(
        self,
        images: Sequence[Image.Image],
        *,
        mode: str = DEFAULT_MODE,
        max_new_tokens: int = DEFAULT_LINE_MAX_NEW_TOKENS,
        batch_size: int = EVAL_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Recognise many images in batches (every image in a batch shares the mode's prompt, so no padding is
        involved); one ``{text, new_tokens, truncated}`` per image, in order."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be an int in 1..64")
        checked = [_check_inputs(image, mode, max_new_tokens)[0] for image in images]
        out: list[dict[str, Any]] = []
        for start in range(0, len(checked), batch_size):
            out.extend(self._generate(checked[start : start + batch_size], mode, max_new_tokens))
            if progress is not None:
                progress(len(out), len(checked))
        return out

    # ------------------------------------------------------------------------------------------------------
    # Adaptation contract
    # ------------------------------------------------------------------------------------------------------

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            raise RuntimeError("this pipeline has no loaded model (injected runner); use from_pretrained for adapt/save_artifact/load_artifact")
        return self._model, self._processor

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int = DEFAULT_LINE_MAX_NEW_TOKENS,
        batch_size: int = EVAL_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Transcribe every validated record in `plain` mode and score the hypotheses with `metrics.ocr_metrics`
        (micro and macro CER / WER, exact match). Works with an injected runner too."""
        from .metrics import ocr_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        items = self.transcribe([r["image"] for r in checked], mode=DEFAULT_MODE, max_new_tokens=max_new_tokens, batch_size=batch_size, progress=progress)
        hypotheses = [item["text"] for item in items]
        metrics = ocr_metrics(hypotheses, checked)
        metrics.update(
            {
                "hypotheses": hypotheses,
                "truncated": sum(item["truncated"] for item in items),
                "new_tokens": sum(item["new_tokens"] for item in items),
                "max_new_tokens": max_new_tokens,
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None,
        *,
        epochs: int = 6,
        lr: float = 5e-5,
        batch_size: int = 8,
        seed: int = 0,
        max_new_tokens: int = DEFAULT_LINE_MAX_NEW_TOKENS,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning of the last `TRAINABLE_LAYERS` decoder layers and the final norm on transcribed lines
        with the causal language-model loss over the transcript tokens (the prompt and the image tokens are masked
        out; the sequence ends with the stop token). The frozen prefix — vision tower, projector, embeddings and the
        first decoder layers — is run once per line under no gradient and its output hidden states are cached, so
        each step runs only the trainable tail; the loss equals the full model's loss exactly. AdamW (no weight
        decay), gradient clipping at `GRAD_CLIP`, seeded shuffling, no scheduler, no augmentation. Epoch 0 records the
        frozen model's validation metrics; the epoch with the lowest validation CER is kept (the final one without a
        validation split). On any exception the frozen weights are restored."""
        model, processor = self._require_model()  # refuse before importing torch
        import torch

        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not isinstance(lr, int | float) or not 0.0 < float(lr) <= 1e-2:
            raise ValueError("lr must be in (0, 1e-2]")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
            raise ValueError("batch_size must be an int in 1..32")
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None
        names = _trainable_names(model)
        name_set = set(names)
        device = torch.device(self.device)
        language_model = model.model.language_model
        first = _TRAINABLE_FIRST_LAYER
        tokenizer = processor.tokenizer
        stop_id = tokenizer.convert_tokens_to_ids(STOP_STRING)
        pad_id = tokenizer.pad_token_id
        frozen_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
        previous_adapter = self.adapter
        cudnn_flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False  # repeatable on one device
        history: list[dict[str, Any]] = []
        started = time.perf_counter()

        def _val() -> dict[str, Any] | None:
            if val_checked is None:
                return None
            result = self.evaluate(val_checked, max_new_tokens=max_new_tokens)
            return {"cer": result["cer"], "wer": result["wer"], "cer_macro": result["cer_macro"], "exact_match": result["exact_match"], "n": result["n"]}

        def _encode(batch: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], Any]:
            encoded = processor([r["image"] for r in batch], return_tensors="pt", padding=True)
            if not bool(encoded["attention_mask"].all()):
                raise RuntimeError("prompts in one batch differ in length")
            prompt = encoded["input_ids"][0]
            targets = [torch.tensor(tokenizer(r["text"], add_special_tokens=False)["input_ids"] + [stop_id], dtype=torch.long) for r in batch]
            length = int(prompt.shape[0]) + max(int(t.shape[0]) for t in targets)
            input_ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
            labels = torch.full((len(batch), length), -100, dtype=torch.long)
            mask = torch.zeros((len(batch), length), dtype=torch.long)
            for i, target in enumerate(targets):
                n = int(prompt.shape[0]) + int(target.shape[0])
                input_ids[i, : prompt.shape[0]] = prompt
                input_ids[i, prompt.shape[0] : n] = target
                labels[i, prompt.shape[0] : n] = target
                mask[i, :n] = 1
            inputs = {"input_ids": input_ids.to(device), "attention_mask": mask.to(device), "pixel_values": encoded["pixel_values"].to(device)}
            return inputs, labels

        def _tail_loss(hidden: Any, labels: Any) -> Any:
            position_ids = torch.arange(hidden.shape[1], device=device).unsqueeze(0).expand(hidden.shape[0], -1)
            embeddings = language_model.rotary_emb(hidden, position_ids)
            for layer in language_model.layers[first:]:
                hidden = layer(hidden, attention_mask=None, position_ids=position_ids, position_embeddings=embeddings)
            # logits only where a transcript token is predicted (the prompt positions carry no loss): the same
            # cross-entropy as the full model's, without a batch x length x vocabulary logit tensor
            targets = labels[:, 1:]
            keep = targets != -100
            logits = model.lm_head(language_model.norm(hidden[:, :-1][keep]))
            return torch.nn.functional.cross_entropy(logits.float(), targets[keep])

        try:
            # 1. cache the frozen prefix: the hidden states entering the first trainable layer, per line
            cache: list[tuple[Any, Any]] = []
            for start in range(0, len(train_checked), CACHE_BATCH_SIZE):
                batch = train_checked[start : start + CACHE_BATCH_SIZE]
                inputs, labels = _encode(batch)
                with torch.no_grad():
                    hidden = model.model(**inputs, output_hidden_states=True).hidden_states[first]
                for k in range(len(batch)):
                    n = int(inputs["attention_mask"][k].sum())
                    cache.append((hidden[k, :n].detach().to("cpu"), labels[k, :n]))
                del hidden
            cache_seconds = round(time.perf_counter() - started, 3)
            # 2. train the tail on the cached states
            params = []
            for name, param in model.named_parameters():
                if name in name_set:
                    param.requires_grad_(True)
                    params.append(param)
            n_trainable = sum(p.numel() for p in params)
            entry = {"epoch": 0, "train_loss": None, "val": _val(), "note": "frozen model"}
            history.append(entry)
            if progress is not None:
                progress(entry)
            best_epoch, best_score = 0, (history[0]["val"] or {}).get("cer", float("inf"))
            best_state = frozen_state
            optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
            rng = random.Random(seed)
            torch.manual_seed(seed)
            width = int(cache[0][0].shape[1])
            for epoch in range(1, epochs + 1):
                model.train()
                order = list(range(len(cache)))
                rng.shuffle(order)
                losses = []
                for start in range(0, len(order), batch_size):
                    items = [cache[k] for k in order[start : start + batch_size]]
                    length = max(int(h.shape[0]) for h, _ in items)
                    hidden = torch.zeros((len(items), length, width), dtype=items[0][0].dtype)
                    labels = torch.full((len(items), length), -100, dtype=torch.long)
                    for k, (h, lab) in enumerate(items):
                        hidden[k, : h.shape[0]] = h
                        labels[k, : lab.shape[0]] = lab
                    loss = _tail_loss(hidden.to(device), labels.to(device))
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": _val()}
                history.append(entry)
                if progress is not None:
                    progress(entry)
                if val_checked is None or entry["val"]["cer"] < best_score:
                    best_epoch, best_score = epoch, (entry["val"] or {}).get("cer", float("inf"))
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
            model.load_state_dict(best_state, strict=False)
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
        except BaseException:
            model.load_state_dict(frozen_state, strict=False)
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
            self.adapter = previous_adapter
            raise
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_flags
        self.adapter = {
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "first_trainable_layer": first,
            "epochs": epochs,
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "selection": "lowest validation CER" if val_checked is not None else "final epoch (no validation split)",
            "loss": "causal language-model cross-entropy over the transcript tokens and the stop token; prompt and image tokens masked; computed on the cached frozen-prefix hidden states",
            "lr": float(lr),
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "n_train": len(train_checked),
            "n_val": len(val_checked) if val_checked is not None else 0,
            "cache_seconds": cache_seconds,
            "history": history,
            "seconds": round(time.perf_counter() - started, 3),
        }
        return dict(self.adapter)

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the trained tensors as safetensors plus a manifest naming the base, the digests and the training
        configuration. Requires a prior `adapt`."""
        model, _processor = self._require_model()  # refuse before importing torch
        import torch
        from safetensors.torch import save_file

        if self.adapter is None:
            raise RuntimeError("nothing to save: call adapt() first")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = list(self.adapter["trainable_names"])
        state = model.state_dict()
        tensors = {name: state[name].detach().cpu().contiguous() for name in names}
        weights = out / ADAPTER_WEIGHTS
        save_file(tensors, str(weights), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "version": ARTIFACT_VERSION,
            "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_file": WEIGHT_FILE, "weight_sha256": self.weight_sha256},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": names,
            "files": [{"path": ADAPTER_WEIGHTS, "bytes": weights.stat().st_size, "sha256": _sha256(weights)}],
            "torch": torch.__version__,
            "metadata": dict(metadata or {}),
        }
        with open(out / ADAPTER_MANIFEST, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Overlay a saved adapter onto this (freshly loaded) pipeline after checking its manifest, digest and exact
        tensor set. Refuses tensors outside the last decoder layers and the final norm."""
        model, _processor = self._require_model()  # refuse before importing safetensors
        from safetensors.torch import load_file

        artifact = Path(artifact_dir)
        manifest_path = artifact / ADAPTER_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        _check_artifact_manifest(manifest, artifact, self.weight_sha256 or "")
        expected = _trainable_names(model)
        if sorted(manifest["tensors"]) != sorted(expected):
            raise ValueError("artifact tensor set does not match its recorded configuration")
        tensors = load_file(str(artifact / ADAPTER_WEIGHTS))
        if sorted(tensors) != sorted(expected):
            raise ValueError("artifact tensor names differ from the manifest")
        state = model.state_dict()
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != tuple(state[name].shape):
                raise ValueError(f"artifact tensor {name} has shape {tuple(tensor.shape)}, base has {tuple(state[name].shape)}")
        model.load_state_dict({k: v.to(state[k].device, state[k].dtype) for k, v in tensors.items()}, strict=False)
        model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": expected, "history": manifest.get("history", [])}
        return dict(self.adapter)

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> GotOcr2Pipeline:
        """Check the adapter manifest against the base snapshot's recorded weight digest, load the verified base, then
        overlay the adapter (checked again, and the tensor set, before deserialising). A refused manifest never loads
        a model."""
        artifact = Path(artifact_dir)
        manifest_path = artifact / ADAPTER_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        _check_artifact_manifest(manifest, artifact, _weight_digest(root) or "")
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe
