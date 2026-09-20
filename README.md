# GOT-OCR 2.0 optical character recognition pipeline

DIMER pipeline for **GOT-OCR 2.0** (`stepfun-ai/GOT-OCR-2.0-hf`), StepFun's end-to-end "OCR 2.0" vision–language model (a ViT encoder feeding a Qwen2-architecture decoder, 560,528,640 parameters) that transcribes the text in an image — plain text or formatted Markdown/LaTeX — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline accepts one image and one of two modes, decodes greedily under a caller-owned token budget until the `<|im_end|>` stop string, and returns the text with a `truncated` flag; it returns no score and no boxes, and it exposes neither the upstream multi-page nor the region-guided modes. On top of inference it carries the **adaptation contract**: corpus-level character and word error rates over transcribed lines, two non-adapted baselines, a bounded fine-tuning of the last four decoder layers on cached frozen-prefix hidden states, and a verified safetensors adapter that reloads against the pinned base.

## Upstream alignment

- Model: `stepfun-ai/GOT-OCR-2.0-hf`
- Revision: `d3017ef2c2c1395888c8d635c5e0508bcb0ac78d`
- Upstream weight license: Apache-2.0
- Upstream task: image-text-to-text — image → recognised text (plain, or formatted)
- Repository adaptation: **bounded supervised fine-tuning** of the last four decoder layers and the final norm (51,401,728 of 560,528,640 parameters) on `{id, image, text}` line records with the causal language-model loss over the transcript tokens; the vision encoder, the projector, the embeddings and the first twenty decoder layers stay frozen. Trained tensors are exported as a safetensors adapter with a manifest and overlaid on a freshly loaded, re-verified base.

## Quick start

```python
from PIL import Image
from got_ocr2_pipeline import GotOcr2Pipeline, character_error_rate, fetch_sample_dataset

pipe = GotOcr2Pipeline.from_pretrained()          # stages + verifies weights/got-ocr-2.0-hf first
result = pipe.recognize(Image.open("notice.png"))  # plain text with line breaks
print(result["text"])
print(result["new_tokens"], result["truncated"])   # True when the max_new_tokens budget was exhausted

result = pipe.recognize(Image.open("paper.png"), mode="format", max_new_tokens=4096)   # Markdown/LaTeX
print(character_error_rate("the transcript you know", result["text"]))

splits = fetch_sample_dataset()                    # 800 digest-pinned Belfort handwritten lines, 600 / 60 / 140
print(pipe.evaluate(splits["test"])["cer"])        # frozen corpus CER (above 1.0: the model invents text)
pipe.adapt(splits["train"], splits["validation"])  # last four decoder layers, lowest-validation-CER epoch kept
print(pipe.evaluate(splits["test"])["cer"])
pipe.save_artifact("outputs/adapter")
again = GotOcr2Pipeline.from_artifact("outputs/adapter")   # re-verifies the base, checks the manifest, overlays
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the offline test suite (no weights needed; `tests/test_model_backed.py` runs only where the snapshot is staged). On a fresh clone the manifest is committed but the weights and the 18.7 MB `tokenizer.json` are not: `GotOcr2Pipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

## Weights layout

```
weights/got-ocr-2.0-hf/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (8 files)
  config.json                # GotOcr2ForConditionalGeneration: 24-layer Qwen2 text decoder (the processor emits 256 image tokens)
  preprocessor_config.json   # 1024x1024 resize, CLIP mean/std
  generation_config.json  special_tokens_map.json  tokenizer_config.json
  tokenizer.json             # git-ignored (18.7 MB), staged with the weights
  model.safetensors          # git-ignored, 1,121,114,488 bytes
  README.md
weights/belfort/             # git-ignored cache of the eight pinned Belfort-line row groups (fetched at run time)
```

## Input ceilings and request parameters

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 16384`, `MAX_IMAGE_PIXELS = 4096 * 4096` (the processor resizes every image to 1024×1024, so the ceilings bound decode and resize memory, not model cost; a 9,000 px wide text line is accepted); `MODES = ("plain", "format")` (`DEFAULT_MODE = "plain"`); `MAX_NEW_TOKENS = 4096` (the pinned README examples' budget), `DEFAULT_MAX_NEW_TOKENS = 1024`, `DEFAULT_LINE_MAX_NEW_TOKENS = 128` (the corpus stages' budget per line); `DECODING = "greedy"`; `STOP_STRING = "<|im_end|>"`. `recognize` takes one image; `transcribe` and `evaluate` batch images that share a mode (every plain-mode prompt is the same 286 tokens, so a batch needs no padding). Any other mode is refused (`ValueError`). The model is loaded in float32 on every device. See `MODEL_CARD.md` for who owns the budget and the measured timings.

## Adaptation contract

- **Records:** `{id, image, text}` — a PIL image (sides within the ceilings) and its transcript (1..512 characters after whitespace runs are collapsed); `validate_dataset` checks the structure, `split_dataset` de-duplicates by decoded pixels and `check_split_disjoint` asserts no image is shared. The default sample (`samples.py`) is the first eight parquet row groups of the Belfort-line test shard (`Teklia/Belfort-line`, MIT; nineteenth-century French council minutes in cursive) read over HTTPS range requests at an immutable Hub revision, each row group refused on any SHA-256 or byte-total mismatch; `load_byod_dataset` reads a zip or directory of line images plus `transcripts.csv`.
- **Measures (`metrics.py`):** `ocr_metrics` — micro CER and WER (total edits over total reference characters or words), macro rates, exact match, and the hypothesis length; uncapped, so a rate above 1.0 means the model generates text the line does not carry. `empty_baseline` (CER 1.0 by construction) and `constant_baseline` (the medoid training transcript for every line).
- **Fine-tuning:** `adapt(train, val, *, epochs=6, lr=5e-5, batch_size=8, seed=0)` caches the hidden states entering decoder layer 20 for every training line (one frozen forward each), then trains layers 20–23 and the final norm on those states with AdamW (no weight decay), gradient clipping at 1.0 and seeded shuffling; the loss equals the full model's loss exactly. Epoch 0 records the frozen validation rates; the epoch with the lowest validation CER is kept; on any exception the frozen weights are restored.
- **Artifact:** `save_artifact` writes `adapter.safetensors` (about 206 MB) + `manifest.json` (`org.valcorza.got-ocr-2.0-hf.adapter.v1`: base identity and weight digest, tensor names, file size and SHA-256, configuration, history); `from_artifact` re-verifies the base and checks the manifest, digest and exact tensor set before deserialising.
- **Build record (Tesla T4, seed 42 split):** frozen CER @P:FROZEN_CER@ / WER @P:FROZEN_WER@ on the 140 held-out lines (worse than the empty baseline's 1.0 — the model invents printed-looking text), adapted **@P:ADAPTED_CER@** / **@P:ADAPTED_WER@** (epoch @P:BEST_EPOCH@ of 6, @P:ADAPTED_EXACT@ lines exact), reload parity 8/8. One seeded split of one 800-line sample; no dispersion estimate.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/got-ocr2-pipeline/blob/main/tutorials/got_ocr2_colab.ipynb)

`tutorials/got_ocr2_colab.ipynb` is declared `E2E` / `GUIDED` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three pipeline modules, the model identity, the manifest digests and the runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`). Its default `Run all` path stages and verifies the pinned snapshot, fetches the eight pinned Belfort row groups and splits the 800 lines 600 / 60 / 140, recognises a synthetic printed page through the inference contract (`validate_inputs`, `recognize`, `evaluation_report`), measures the frozen model's CER and WER on the held-out lines beside the empty and constant baselines, runs `adapt` with validation-CER epoch selection, scores the held-out lines again, writes six line panels and re-reads the page with the adapted model, and exports the adapter and reloads it with verified transcript parity. BYOD is optional and gated off by default. See `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Candidate.** Static/unit checks — including the standalone generator parity checks (`tools/build_notebook.py --check`, `tests/test_notebook_parity.py`) — do not constitute clean-runtime notebook evidence. The supported-runtime run of the exact release revision is recorded in `docs/release-verification.md` when it exists; until then the notebook is not release-grade.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output contract, adaptation build record, measured runtime.
- `docs/WEIGHTS.md` — weight provenance and hosting notes.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; the Belfort-line sample is MIT and is not redistributed; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
