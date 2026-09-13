"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "got_ocr2_pipeline",
    "repo_name": "got-ocr2-pipeline",
    "stem": "got_ocr2",
    "notebook_name": "got_ocr2_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "GotOcr2Pipeline",
    "weights_key": "got-ocr-2.0-hf",
    "runtime_imports": ["torch", "transformers"],
    "title": "GOT-OCR 2.0 — DIMER optical character recognition tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/got-ocr2-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/got-ocr2-pipeline/blob/main/tutorials/got_ocr2_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-stepfun--ai%2FGOT--OCR--2.0--hf-ffcc4d?style=flat",
            "https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-Ucas--HaoranWei%2FGOT--OCR2.0-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/Ucas-HaoranWei/GOT-OCR2.0",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2409.01704-b31b1b.svg", "https://arxiv.org/abs/2409.01704"),
    ],
    "capability": "optical character recognition — one image → its text, plain or formatted (Markdown/LaTeX) — using the pinned `stepfun-ai/GOT-OCR-2.0-hf` weights",
    "intro": (
        "At inference the GOT-OCR 2.0 model — a ViT-style vision encoder over a 1024×1024 image producing 576 image tokens "
        "and a 24-layer Qwen2-architecture decoder (about 580M parameters) — reads the image wrapped in the snapshot's chat "
        "prompt for the chosen mode and generates the text token by token until the `<|im_end|>` stop string or the budget. "
        "In `plain` mode the output is the page text with line breaks; in `format` mode it is formatted text (Markdown/LaTeX "
        "constructs such as `\\title{}`, tables, formulas). Decoding is greedy (`do_sample=False`) under a caller-owned "
        "`max_new_tokens` budget. **No adaptation occurs:** no training, fine-tuning, in-context conditioning, or "
        "preprocessing fitting happens in this notebook — the upstream checkpoint supplies the weights, processor and "
        "tokenizer, and the carried module adds snapshot verification, the input contract (image side ceilings, one of two "
        "modes, the token budget), a fixed output contract, and the `character_error_rate`, `word_error_rate`, "
        "`validate_inputs` and `evaluation_report` helpers. The default sample is a notice page rendered in code from known "
        "text, so its transcript is the reference; the resulting error rates are demonstration (plumbing) evidence for one "
        "clean rendered page, not an OCR benchmark."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, render a synthetic printed page with a known transcript (or upload your own "
        "image) and validate it into an input manifest, choose a recognition mode and a token budget, run the supported task, "
        "read the output correctly (generated text, no score, a `truncated` flag), exercise an optional BYOD path, produce an "
        "evaluation report that is `sample-sanity` with `character_error_rate` and `word_error_rate` only when a reference "
        "transcript exists and `not-measurable` otherwise, and export the text and provenance."
    ),
    "exclusions": (
        "PDF or multi-page documents (one image per call; the upstream multi-page mode is not exposed), region-guided OCR "
        "by box or colour (upstream interactive modes, not exposed), rendering the formatted output (LaTeX, Markdown tables "
        "or charts need external tools), layout or reading-order output (only the text stream is returned), batch "
        "throughput, sampling or beam search, evaluation on an OCR benchmark (which needs labelled images; only the "
        "rendered page is scored here), and any training. The model generates text for any image, including one with no "
        "text, and gives no signal when it invents."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (bfloat16, the checkpoint's stored dtype). CPU is adequate but slow: the repository's model card records 6.7 s to load and 11.4 s for the 1000×520 rendered page (154 generated tokens) in the Windows venv (Intel Core Ultra 9 275HX); cost scales with the tokens generated. The pinned `torch==2.14.0` install and the 1.12 GB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what a vision–language model's generated tokens are; what character and word error rate measure; that fluent output is not correct output.",
        "- **Data:** the default sample is a deterministic 1000×520 notice page rendered in code with Pillow's bundled font — a heading and seven lines of English text with dates, numbers and punctuation — so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image decodable by Pillow (PNG/JPEG/WebP and similar) containing text — a document page, a photograph of a sign, a screenshot — any colour mode, sides between 16 and 4096 px. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Render the synthetic printed page or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own reference: a notice page — a heading in a larger face "
                "and seven lines of English text with dates, numbers, a postcode and punctuation — is rendered with Pillow's "
                "bundled font at 1000×520, the same page the repository's smoke run used. The lines drawn on it, joined with "
                "newlines, are the reference transcript for the error-rate sanity check later; they are not a labelled dataset, "
                "so nothing here is an OCR benchmark, and a rendered page is far cleaner than any scan or photograph. The image "
                "digest is printed for the record. BYOD is optional and disabled by default; when enabled, upload one image — no "
                "transcript exists for it, so the evaluation report will be `not-measurable`.\n\n"
                "The recognition mode and the token budget are **caller-owned request parameters**: `mode` is `plain` (the page "
                "text with line breaks) or `format` (formatted Markdown/LaTeX output, the upstream `format=True` path), and "
                "`max_new_tokens` bounds the generation (`DEFAULT_MAX_NEW_TOKENS = 1024`; `MAX_NEW_TOKENS = 4096` is the ceiling "
                "the pinned README's examples use). Nothing is validated in this cell — the next section hands the image and the "
                "request to the pipeline's own validation stage, which is the only checker. Look for a dictionary naming the "
                "sample kind, the page size and digest, the request, and the number of reference characters."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "mode = 'plain'  # @param ['plain', 'format']\n"
                "max_new_tokens = 1024  # @param {{type:\"integer\"}}\n\n"
                "LINES = [\n"
                "    'NOTICE OF ANNUAL GENERAL MEETING',\n"
                "    'The annual general meeting of Northwind Traders Ltd. will be held',\n"
                "    'on Thursday 23 April 2026 at 10:00 in the Harbour Room, Portsmouth.',\n"
                "    \"Agenda: 1. Minutes of the previous meeting. 2. Directors' report.\",\n"
                "    '3. Approval of accounts for the year ended 31 December 2025.',\n"
                "    '4. Re-election of directors. 5. Appointment of auditors.',\n"
                "    'Shareholders unable to attend may appoint a proxy by 21 April 2026.',\n"
                "    'Registered office: 14 Harbour Road, Portsmouth PO1 3AX. Reg. no. 04471120',\n"
                "]\n\n\n"
                "def printed_page(width=1000, height=520):\n"
                "    \"\"\"A notice page rendered with Pillow's bundled font; returns page + reference transcript.\"\"\"\n"
                "    page = Image.new('RGB', (width, height), 'white')\n"
                "    d = ImageDraw.Draw(page)\n"
                "    head, body = ImageFont.load_default(size=30), ImageFont.load_default(size=22)\n"
                "    d.text((60, 40), LINES[0], fill='black', font=head)\n"
                "    y = 110\n"
                "    for line in LINES[1:]:\n"
                "        d.text((60, y), line, fill=(20, 20, 20), font=body)\n"
                "        y += 48\n"
                "    return page, '\\n'.join(LINES)\n\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    reference_text = None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic synthetic page: no randomness, so no seed is needed and the digest is stable per Pillow build.\n"
                "    image, reference_text = printed_page()\n"
                "    image_name = 'synthetic_notice_1000x520.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256, 'recognition_mode': mode, 'max_new_tokens': max_new_tokens, 'reference_chars': None if reference_text is None else len(reference_text)}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `recognize` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px, a mode that is one of `MODES`, and "
                "`max_new_tokens` in `[1, MAX_NEW_TOKENS]` — and returns an **input manifest** naming the schema (including the "
                "preprocessing the processor applies and the decoding rule), the input's observed mode and size, the request, "
                "and the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks "
                "like, the cell also validates a mode the pipeline does not expose and records the pipeline's own error message "
                "as a finding. Inside the pipeline the image is converted to RGB and resized to 1024×1024 by the processor (the "
                "aspect ratio is not preserved); nothing else is dropped or altered. The pipeline cannot tell whether the image "
                "contains text: that contract is the caller's."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'MODES': list(MODES), 'DECODING': DECODING, 'STOP_STRING': STOP_STRING}}}})\n"
                "input_manifest = validate_inputs(image, mode=mode, max_new_tokens=max_new_tokens, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, mode='multi-page')\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'unsupported-mode-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Recognise the text and read the output correctly\n\n"
                "`recognize` returns a dict with `text` — the decoded output with the stop string removed — the `mode`, "
                "`image_size`, `new_tokens`, a `truncated` flag that is true when the budget was exhausted, the generation "
                "settings and the model identity. **No score exists**: generated text carries no probability and no correctness "
                "signal, and fluent output is not evidence that it matches the image. Greedy decoding is deterministic on a fixed "
                "device and dtype; CUDA bfloat16 can change a token and therefore the rest of the sequence, so GPU and CPU outputs "
                "need not match. As recorded in the model card, the repository's CPU smoke on this same page in `plain` mode "
                "generated 154 tokens in 11.4 s and reproduced the transcript **exactly** (character and word error rate 0.0); "
                "`format` mode wrapped the heading in `\\title{{}}` and scored 0.02 / 0.025 against the plain transcript because "
                "of that markup; a blank 1024×1024 image produced `(0) = 1 +`. Those are observations on one rendered page, not "
                "calibration points."
            ),
            "code": (
                "import time\n\n"
                "t0 = time.time()\n"
                "result = pipe.recognize(image, mode=mode, max_new_tokens=max_new_tokens)\n"
                "elapsed = time.time() - t0\n"
                "print({{'seconds': round(elapsed, 1), 'new_tokens': result['new_tokens'], 'truncated': result['truncated'], 'device': pipe.device, 'dtype': pipe.dtype, 'n_chars': len(result['text'])}})\n"
                "print(result['text'][:1500] + ('…' if len(result['text']) > 1500 else ''))\n"
                "if result['truncated']:\n"
                "    print('The token budget was exhausted: the text is incomplete. Raise max_new_tokens (ceiling MAX_NEW_TOKENS) and rerun.')"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No OCR metric is "
                "reported by default: accuracy needs images with ground-truth transcripts from the deployment domain, and this "
                "repository ships none. The repository's metric helpers are `character_error_rate` (character-level Levenshtein "
                "distance over whitespace-normalised text, case and punctuation kept, divided by the reference length) and "
                "`word_error_rate` (word-level, lower-cased); when a reference transcript is supplied the report carries both, "
                "with the verdict `sample-sanity`. On the synthetic path that reference is text **you rendered yourself** on a "
                "clean white page, so a zero error rate proves only that the input contract, forward pass, decoding and stop "
                "handling round-trip; in `format` mode the added markup counts as errors against the plain transcript, which is "
                "expected. On BYOD no reference exists, the verdict is `not-measurable`, and the report states what would make "
                "the task measurable. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, reference_text, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k != 'metrics'}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    print(f\"{{metric['id']:22}} {{metric['value']:.4f}}  ({{metric['normalisation']}})\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No reference transcript exists for this input, so nothing is scored; read the text against the image yourself.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the full result (text, mode, `new_tokens`, `truncated`, the budget), the "
                "evaluation report, the input manifest, the sample identity, digest and reference transcript, the notebook's "
                "source (repository, revision, embedded module digest, generator), the model identifier, the immutable model "
                "revision, the model licence, and the runtime identity (Python, `torch`, `transformers`, device, dtype). The "
                "recognised text is also written verbatim to `outputs/{stem}.txt`, and a side-by-side PNG places the input image "
                "above a panel with the recognised text for visual inspection — a supplement to, not a replacement for, the "
                "machine-readable files. No credentials are recorded."
            ),
            "code": (
                "panel_lines = result['text'].splitlines()[:24] or ['(no text)']\n"
                "panel_height = 24 + 22 * len(panel_lines)\n"
                "annotated = Image.new('RGB', (image.width, image.height + panel_height), 'white')\n"
                "annotated.paste(image.convert('RGB'), (0, 0))\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "draw.line([(0, image.height + 1), (image.width, image.height + 1)], fill=(120, 120, 120), width=2)\n"
                "panel_font = ImageFont.load_default(size=16)\n"
                "for index, line in enumerate(panel_lines):\n"
                "    draw.text((16, image.height + 10 + 22 * index), line[:140], fill=(40, 90, 220), font=panel_font)\n"
                "annotated.save('outputs/{stem}_annotated.png')\n"
                "with open('outputs/{stem}.txt', 'w', encoding='utf-8') as handle:\n"
                "    handle.write(result['text'])\n"
                "payload = {{\n"
                "    'prediction': result,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'reference_text': reference_text}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "        'dtype': pipe.dtype,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The text is what the model generates after reading the image; nothing in the output scores it, and a fluent "
        "transcript can still carry wrong characters, skipped lines or invented content. On the synthetic page the error rates "
        "in the evaluation report compare the output with text you rendered yourself on a clean white page and the verdict is "
        "`sample-sanity`, which proves only that the input contract, forward pass, decoding and stop handling work (the "
        "repository's smoke run reproduced this page exactly in `plain` mode); they say nothing about scans, photographs, "
        "handwriting, low resolution, dense multi-column layouts, non-Latin scripts, formulas or tables, and a BYOD result is a "
        "single-image observation with the verdict `not-measurable`. **The model generates text for any image** and stops only "
        "at the stop string or the budget: a blank page yielded `(0) = 1 +` in the smoke run, so an image with no text produces "
        "an invented fragment rather than an empty result, and `truncated` is the only structural signal you get. Everything "
        "is resized to 1024×1024 without preserving the aspect ratio, so a wide receipt or a tall page is distorted. The "
        "pipeline provides no layout, no boxes, no multi-page or region modes, no rendering of formatted output and no "
        "training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** switch `mode` to `format` and see the heading come back as `\\title{{...}}` (and the error rates "
        "rise for the markup alone); lower `max_new_tokens` to 40 and watch `truncated` turn true; downscale the page to 400 px "
        "wide before recognition and compare the error rates; enable `USE_BYOD` with a photograph of a sign or a scanned page, "
        "then type its transcript and pass it as `reference_text` to `evaluation_report` to see the verdict switch to "
        "`sample-sanity`.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/got-ocr2-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/got-ocr2-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/got-ocr2-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/Ucas-HaoranWei/GOT-OCR2.0\n"
        "- General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model (Wei et al., 2024): https://arxiv.org/abs/2409.01704"
    ),
}
