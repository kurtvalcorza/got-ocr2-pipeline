---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: image-text-to-text
base_model: stepfun-ai/GOT-OCR-2.0-hf
date_published: "2024-11-22"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt` 2024-11-22T23:01:40Z, https://huggingface.co/api/models/stepfun-ai/GOT-OCR-2.0-hf — the Transformers-native conversion); the original `stepfun-ai/GOT-OCR2_0` release and the paper arXiv:2409.01704 are from 2024-09, and the pinned revision is the Hub's `main` as of 2026-09-14"
---

# GOT-OCR 2.0 (DIMER package v0.1.0) — Optical Character Recognition (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-stepfun--ai%2FGOT--OCR--2.0--hf-ffcc4d?style=flat)](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-Ucas--HaoranWei%2FGOT--OCR2.0-181717?style=flat&logo=github&logoColor=white)](https://github.com/Ucas-HaoranWei/GOT-OCR2.0)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2409.01704-b31b1b.svg)](https://arxiv.org/abs/2409.01704)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — stage and verify the pinned upstream revision in a fresh runtime, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/got-ocr2-pipeline/blob/main/tutorials/got_ocr2_colab.ipynb) [`got_ocr2_colab.ipynb`](https://github.com/kurtvalcorza/got-ocr2-pipeline/blob/main/tutorials/got_ocr2_colab.ipynb)  
  *Recognition of a notice page rendered in code with the pinned `stepfun-ai/GOT-OCR-2.0-hf` weights: plain or formatted text under a caller-owned mode and token budget, and `character_error_rate` / `word_error_rate` against the rendered transcript as sanity evidence only — no OCR benchmark.*

---

#### Description

`stepfun-ai/GOT-OCR-2.0-hf` is the Transformers-native release of GOT-OCR 2.0 — "General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model" (Wei et al., arXiv:2409.01704; StepFun and collaborators) — contributed to the Hugging Face ecosystem by Yoni Gozlan and pinned here to revision `d3017ef2c2c1395888c8d635c5e0508bcb0ac78d` (the Hub's `main` on 2026-09-14). The snapshot `config.json` declares `GotOcr2ForConditionalGeneration` (`model_type` got_ocr2): a ViT-style vision encoder whose 1024×1024 input becomes 576 image tokens (`image_seq_length`) projected into a 24-layer Qwen2-architecture text decoder (hidden size 1024, 16 heads, vocabulary 151,860 with a dedicated image token) — about 580M parameters in the 1.12 GB bfloat16 `model.safetensors`. At inference the processor (`GotOcr2Processor`, `preprocessor_config.json`) resizes the image to 1024×1024 without preserving the aspect ratio, normalises with CLIP mean/std and wraps the image tokens in the snapshot's chat prompt for the requested mode; the decoder generates the text greedily until the `<|im_end|>` stop string or the budget. The upstream README documents plain-text, formatted (Markdown/LaTeX), multi-page, cropped-patch and region-guided (box/colour) modes; this pipeline exposes the first two. Nothing is trained or adapted here. What this repository adds is packaging: `verify_snapshot` and `stage_missing_files` (manifest digest checking and fresh-clone staging), `GotOcr2Pipeline.from_pretrained` (verified local loading with `trust_remote_code=False`, float32 on CPU and the checkpoint's bfloat16 on CUDA), `recognize` (input validation, the two modes, greedy decoding with the stop string under a caller-owned `max_new_tokens`, a `truncated` flag), `character_error_rate`, `word_error_rate`, and the `validate_inputs` and `evaluation_report` stage helpers.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is optical character recognition: input one image (`PIL.Image.Image`, any mode, converted to RGB), a mode (`plain` or `format`) and a token budget; output the recognised text, the number of tokens generated and whether the budget was exhausted. Envisioned applications are transcription of rendered or scanned document pages, screenshots and photographed signs or labels for indexing, search and downstream extraction; and, in `format` mode, first-pass conversion of pages with tables, formulas or structured headings into Markdown/LaTeX-style text for a human to check and render. Within DIMER the pipeline is an inference component and a zero-configuration baseline for OCR, not a certified transcription engine for any specific document family, script or camera.

###### Primary Intended Users

Intended users are machine-learning engineers, document-processing developers, and data analysts integrating OCR into research prototypes, internal document tooling, or the DIMER workbench. A user is expected to understand that the output is *generated text* — it carries no per-character confidence, no boxes and no correctness signal, and the model produces text for any image, including one with none — that the token budget is theirs to set (a dense page can exceed 1,024 tokens and be reported `truncated`), that `format` mode adds markup that a plain transcript will count as errors, that the image is squashed to 1024×1024 so extreme aspect ratios and small type suffer, that the upstream training data is not disclosed in detail so the model's behaviour on scans, handwriting, low resolution and non-Latin scripts is a distribution question this repository did not measure, that greedy decoding is reproducible on a fixed device but GPU bfloat16 and CPU float32 outputs need not match, and that accuracy can only be measured on images with ground-truth transcripts they supply. Users who need layout, word boxes, multi-page handling, region-guided recognition or batch throughput are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** no layout or reading-order output, no word or line boxes, no per-character confidence, no multi-page or cropped-patch modes, no region-guided (box/colour) recognition although the upstream model supports them, no rendering of formatted output (LaTeX, tables, charts, sheet music need external tools), no batching, no sampling or beam search, and no abstention (the model cannot say "no text here").
2. **Input boundary:** `recognize` rejects non-PIL images (`TypeError`), sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px, modes not in `MODES`, and budgets outside `[1, MAX_NEW_TOKENS = 4096]` (`ValueError`/`TypeError`). Every image is resized to 1024×1024 with the aspect ratio discarded into 576 image tokens, so glyphs that are only a few pixels tall at that scale are unlikely to be read and a page with more text than the budget allows is returned `truncated`.
3. **Input boundary:** the upstream paper describes training on synthetic and collected document, scene-text, formula, table, chart, molecular and sheet-music renders across English and Chinese; the snapshot README calls the model multilingual without enumerating scripts. Handwriting, degraded scans, photographs with perspective and glare, dense multi-column layouts, right-to-left scripts and scripts absent from training fall outside what the upstream authors report and what this repository measured; results on them are undefined, not merely degraded. An image with no text still produces text (see §Risks and harms).
4. **Decision boundary:** not for autonomous decisions that act on transcribed values — invoice or contract processing feeding payments, clinical-record transcription, identity-document reading, regulatory filings — without a human comparing the text with the image, and a locally measured character error rate on the deployment's own labelled images.

#### Factors

###### Groups

This pipeline is not human-centric by design: it transcribes text in an image and never classifies, identifies or scores people. The upstream training data (per the paper: synthetic and collected document pages, scene text, formulas, tables, charts, molecular formulas and sheet music, in English and Chinese) contains no evaluation groups in the demographic sense, and neither the upstream authors nor this repository audited it for anything of the kind. What does vary is the text population: the model was trained predominantly on English and Chinese renders, so scripts, languages, fonts, handwriting styles, historical typesetting and document conventions outside that mix are the groups whose recognition accuracy is unknown, not known to be equal — and OCR error rates in the literature vary strongly by script and print quality. Where images carry personal data — medical records, HR files, identity documents, correspondence naming individuals — the pipeline's output makes that data machine-readable and searchable; the operator who processes such images is responsible for a fairness and privacy audit on their own image set, stratified by document family and script, before relying on the output.

###### Instrumentation

The upstream training "instrument" is largely a renderer — documents, formulas, tables and charts rendered to images with known text as the target, plus collected scene-text and page images — so crisp glyphs and known fonts dominate. Inference images arrive from whatever produced them — a PDF renderer at some DPI, a scanner, a phone camera with perspective and glare, a screenshot — and resolution, blur, compression, contrast, skew and font all change the visual evidence; the 1024×1024 resize discards resolution and distorts aspect on every image regardless of source. The pipeline validates type and size only; it cannot detect a low-DPI render, a skewed photograph, a page cut mid-column, or an image with no text. The synthetic tutorial page (Pillow's bundled font, black on white, generous margins) is a best case: the model transcribed it exactly, which bounds nothing about any real capture instrument.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, float32 on CPU; CUDA is used automatically when visible (bfloat16, the checkpoint's stored dtype) but was not exercised for this card. The snapshot declares the slow image processor (transformers prints a `use_fast` notice); it is used as declared. Measured on the reference machine with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`) and the Hub offline (`HF_HUB_OFFLINE=1`): `verify_snapshot` on the 8-file, 1.14 GB snapshot 0.59 s; load 6.74 s; one 1000×520 rendered notice page 11.4 s in `plain` mode (154 tokens) and 11.8 s in `format` mode (157 tokens); a blank 1024×1024 image with a 64-token budget 4.7 s (8 tokens) — cost is dominated by autoregressive decoding and scales with the tokens a page needs. Data environment: the model assumes an image of legible text in a script it was trained on; the synthetic page satisfies that assumption and is where the measured behaviour holds. Scans, photographs, handwriting, dense layouts and unfamiliar scripts violate it to degrees this repository did not measure, and the pipeline reports no signal when they do — nor when the image contains no text at all.

#### Metrics

###### Performance Measures

The pipeline reports no accuracy measure. Generated text carries no score, no probability and no correctness signal; `new_tokens` and `truncated` describe the generation, not its quality. The repository ships the two helpers OCR evaluation is built from: `character_error_rate(reference, hypothesis)` (character-level Levenshtein distance over whitespace-normalised text with case and punctuation kept, divided by the reference length) and `word_error_rate(reference, hypothesis)` (word-level, lower-cased). Both need images with ground-truth transcripts that the caller must supply; the benchmarks the upstream paper reports (its own document, scene-text, formula and table sets) are not bundled. The public `evaluation_report(result, reference_text=None)` stage returns that report in machine-readable form: one entry for each error rate with the verdict `sample-sanity` when a reference transcript is supplied, or the verdict `not-measurable` naming the labelled set that would be required when none is. In `format` mode the markup the model adds (for example `\title{}` around a heading) counts as errors against a plain transcript, which the tutorial states rather than corrects. The upstream paper's numbers are upstream-reported and this pipeline does not reproduce or claim them.

###### Decision thresholds

No score threshold exists: the model generates tokens until it emits the stop string or the budget is exhausted, and nothing is filtered. The decision parameters are the **mode** — `plain` (the processor's default prompt) or `format` (the upstream `format=True` prompt; other upstream modes are refused) — and the **token budget** `max_new_tokens`, default `DEFAULT_MAX_NEW_TOKENS = 1024` (a practical single-page budget chosen by this repository; the rendered page needed 154) with ceiling `MAX_NEW_TOKENS = 4096` (the budget every example in the pinned README passes). A budget that is too small is reported, not hidden: `truncated` is true whenever `new_tokens` reaches it, and the caller should raise the budget and rerun. Decoding is greedy (`do_sample=False`) with `stop_strings="<|im_end|>"` passed with the processor's tokenizer, the README's setting; there is no repetition penalty, so a repetition loop runs to the budget. A deployment owns choosing the mode and the budget per document family and deciding whether a `truncated` page is retried or rejected.

###### Approaches to uncertainty and variability

This repository reports no central metric value and therefore no dispersion: the smoke run records timings, token counts and two error rates on one rendered page, not accuracy. Run-to-run variability comes only from floating-point kernel selection across CPU builds and accelerators; there is no sampling and no seed to set, so a fixed input on fixed hardware is repeatable, but because decoding is autoregressive a single differing token changes the rest of the sequence, and the CPU float32 output need not match a CUDA bfloat16 output; the synthetic page's own bytes depend on the Pillow build's bundled font. On the synthetic page the model reproduced the 493-character transcript exactly in `plain` mode (character and word error rate 0.0) and scored 0.020 / 0.025 in `format` mode because of the `\title{}` markup — one observation on one clean page, not an estimate. A caller who needs an accuracy estimate must supply labelled images and compute error rates over many images or bootstrap resamples themselves; a caller who needs a confidence signal per character has none from this model.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The snapshot README does not enumerate the training data; the paper describes a mixture of synthetic renders (documents, formulas, tables, charts, molecular formulas, sheet music) and collected document and scene-text images, in English and Chinese, with sources named only in aggregate. Collected document and scene images can contain personal data — names, addresses, licence plates, faces in the background — so personal data in the training corpus is not ruled out; it is not enumerated by the upstream authors and was not audited here. This repository distributes code, tests, and documentation; it does not distribute the 1,121,114,488-byte `model.safetensors` or the 18.7 MB `tokenizer.json`, which are staged locally under `weights/got-ocr-2.0-hf/` and git-ignored, and it ships no sample images — the tutorial page is rendered in code with fictitious names. The operator must audit the images they submit for personal, proprietary, or otherwise restricted content; the pipeline performs no such check and will transcribe a medical record as readily as a meeting notice.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — transcribing clinical records or prescriptions, invoices or contracts for automated payment or compliance, identity documents for verification, legal filings for screening — would be admissible only with human comparison of the transcript against the image (the model can drop, alter or invent text with no signal), a locally measured character error rate on the deployment's own labelled images, a documented mode and budget policy with `truncated` handling, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks all 8 listed files' byte sizes and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True`, always passes `trust_remote_code=False`, and the smoke run loaded and recognised with `HF_HUB_OFFLINE=1`. No pickle checkpoint exists at the pinned revision. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused; the import-boundary tests assert that a missing or tampered snapshot is refused before `torch` or `transformers` is imported.
- **Input integrity:** the public `validate_inputs(image, *, mode, max_new_tokens)` stage applies exactly the checks `recognize` applies (both route through one shared private checker) and returns an input manifest recording the schema, the ceilings, the observed input, the request and the verdict; `validate_image` rejects non-PIL inputs and sides outside 16–4096 px; modes outside the two exposed forms, non-integer or boolean budgets, and budgets outside `[1, 4096]` are rejected; `recognize` raises on a malformed runner result.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; greedy decoding with no sampling; the stop string is a module constant; every result carries `model_id`, `model_revision`, the mode, the budget, `new_tokens`, `truncated`, the device and the dtype.
- **Refusals:** no batching, no download without the explicit flag, no free-form prompts or upstream modes beyond the two exposed, no sampling, no pickle deserialisation, no attempt to guess whether the image contains text.
- No statistical mitigation (class balancing, subsampling) applies: no training happens in this repository.

###### Risks and harms

- **Invented text on empty or unreadable input:** the model has no abstention — a blank image yielded `(0) = 1 +` in the smoke run — so a photograph without text, a blank scan or an unreadable region produces a plausible fragment with no signal; downstream consumers that trust the string inherit the error silently.
- **Silent substitutions:** a wrong digit in an amount, a date or a reference number is still fluent text; only a character error rate on labelled images reveals the rate, and the tutorial's 0.0 on a rendered page bounds nothing.
- **Truncation and loops:** a dense page can exceed the budget (reported via `truncated`) and a repetition loop runs to the budget; a caller who ignores the flag ships a partial page.
- **Markup confusion:** `format` mode interleaves LaTeX/Markdown constructs with the text; a consumer expecting plain text mis-parses it.
- **Automation bias:** an exact transcript of a clean page invites trust that generated text has not earned on real captures.
- **Privacy exposure:** images containing personal or confidential text are transcribed without any content check and made searchable.
- **Bias amplification:** any script, language, font or document family the English/Chinese render-heavy mixture under-represents is reproduced as uneven accuracy, undetected because no per-family evaluation exists.
- **Resource use:** a 1.12 GB model and ~11 s per page on the reference CPU; a page stream saturates a shared host quickly, and the CUDA path was not measured.

###### Use cases

Prohibited even where the model would work: transcribing documents in order to extract personal data for surveillance, profiling, social scoring, or unlawful discrimination in employment, housing, credit, insurance, education, or healthcare access; processing images the operator has no right to process, or paywalled and licence-restricted material in breach of its terms; deceptive uses that present generated transcripts as verified records or as evidence; and any use that violates the upstream Apache-2.0 licence terms, the DIMER deployment terms, or the consent and data-protection obligations attached to the images processed. Autonomous high-consequence actions triggered by unreviewed transcripts are prohibited by the intended-use contract above.

## Immutable provenance

- Model: `stepfun-ai/GOT-OCR-2.0-hf`
- Revision: `d3017ef2c2c1395888c8d635c5e0508bcb0ac78d`
- Snapshot manifest: `weights/got-ocr-2.0-hf/dimer-base-manifest.json`, 8 files, `totalBytes` 1139869676
- `model.safetensors` SHA-256: `6175ac7868a4e75735f5d59f78c465081ad3427eb4f312d072a0f1d16b333ba4` (1,121,114,488 bytes, bfloat16 tensors)
- `config.json` SHA-256: `cbe8aacd6cd84a2d58eafcd0045c6ac40e02e3a448f24b8cee51cc81d8bdccf2` (608 bytes; `GotOcr2ForConditionalGeneration`, 576 image tokens, Qwen2 text config)
- `tokenizer.json` SHA-256: `36b382a3c48c9a143c30139dac6c8230ddfb0b46a3dc43082af6052abe99d9de` (18,702,549 bytes; git-ignored, staged with the weights)
- Weight format: SafeTensors; loader `AutoModelForImageTextToText.from_pretrained(<dir>, local_files_only=True, trust_remote_code=False, dtype=float32|bfloat16)` with `AutoProcessor` from the same directory; generation with `stop_strings="<|im_end|>"` and the processor's tokenizer. No pickle checkpoint exists at this revision.

## Input/output contract

- `GotOcr2Pipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; `device` defaults to `cuda:0` when visible (bfloat16), else `cpu` (float32).
- `recognize(image, *, mode="plain", max_new_tokens=1024) -> dict` with keys `text` (stop string removed, stripped), `mode`, `image_size`, `new_tokens`, `truncated`, `generation` (`max_new_tokens`, `do_sample` false, `decoding` greedy), `device`, `dtype`, `source`, `model_id`, `model_revision`.
- `character_error_rate(reference, hypothesis) -> float`; `word_error_rate(reference, hypothesis) -> float`.
- Ceilings and constants: `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MODES = ("plain", "format")`, `DEFAULT_MODE = "plain"`, `MAX_NEW_TOKENS = 4096`, `DEFAULT_MAX_NEW_TOKENS = 1024`, `DECODING = "greedy"`, `STOP_STRING = "<|im_end|>"`, `INPUT_SCHEMA`.
- `validate_inputs(image, *, mode, max_new_tokens, names) -> dict`; `evaluation_report(result, reference_text=None, *, sample_kind) -> dict`; `verify_snapshot(path=None) -> dict`; `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`; Python 3.12.
- Precision: float32 on CPU, bfloat16 on CUDA; preprocessing resize to 1024×1024 (aspect ratio not preserved), CLIP mean/std, 576 image tokens (`GotOcr2Processor`, snapshot defaults, slow image processor as declared); greedy decoding to `<|im_end|>`.
- Measured 2026-09-14 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=-1` and `HF_HUB_OFFLINE=1`, device `cpu`: `verify_snapshot` 0.59 s (8 files, 1.14 GB); load 6.74 s; `recognize` on a synthetic 1000×520 notice page (a heading and seven lines of English text with dates, numbers, a postcode and punctuation — 493 characters — rendered with Pillow's bundled font) in `plain` mode with `max_new_tokens=1024` → 154 new tokens, `truncated` false, 11.4 s, `character_error_rate` 0.0 and `word_error_rate` 0.0 against the rendered transcript (verdict `sample-sanity`); the same page in `format` mode → 157 tokens, 11.8 s, the heading wrapped in `\title{...}`, error rates 0.020 / 0.025 against the plain transcript; a 1024×1024 blank image with `max_new_tokens=64` → `(0) = 1 +` (8 tokens, 4.7 s).
- Tutorial execution: `tutorials/got_ocr2_colab.ipynb` ran top-to-bottom in a fresh local kernel (all 8 code cells, 124.3 s including the 1.14 GB staging, same 154 tokens and 0.0 error rates as the smoke run); recorded in `docs/release-verification.md` as pre-flight, not supported-runtime evidence.
- Tests: `pytest -q -o addopts= tests` — offline, no weights required; `ruff check src tests tools` clean.
- Not executed: CUDA/bfloat16 path, the fast image processor, pages that exceed the default budget, any error-rate measurement against labelled images, scans or photographs, non-Latin scripts, formulas, tables or charts in `format` mode (the synthetic page has none), the upstream multi-page and region modes.

## References

- Wei et al. General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model. 2024. https://arxiv.org/abs/2409.01704
- Upstream code: https://github.com/Ucas-HaoranWei/GOT-OCR2.0
- Upstream card (Transformers-native): https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf
- Original release: https://huggingface.co/stepfun-ai/GOT-OCR2_0
- Transformers `GOT-OCR2` documentation: https://huggingface.co/docs/transformers/model_doc/got_ocr2
