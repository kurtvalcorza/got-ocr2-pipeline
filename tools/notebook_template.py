"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (three modules,
carried verbatim in dependency order), and the model pin/stage/verify cells are produced by the generator from
repository sources so they cannot drift from the package.

This template configures an E2E handwritten-text-line adaptation workflow: the pinned stepfun-ai/GOT-OCR-2.0-hf
snapshot is digest-verified and loaded, 800 MIT-licensed Belfort handwritten line images with their transcripts are
fetched as eight digest-pinned parquet row groups over HTTPS range requests, the records are validated and split by
line, a synthetic printed page is recognised through the inference contract, the frozen model's character and word
error rates over the held-out lines are measured beside an empty-string and a constant-transcript baseline, a bounded
fine-tuning of the last four decoder layers runs on cached frozen-prefix hidden states with validation-CER epoch
selection, the held-out split is scored again, six held-out lines and the printed page are re-read with the adapted
model, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "got_ocr2_pipeline",
    "repo_name": "got-ocr2-pipeline",
    "stem": "got_ocr2",
    "notebook_name": "got_ocr2_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "pipeline_class": "GotOcr2Pipeline",
    "weights_key": "got-ocr-2.0-hf",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "runtime_imports": ["torch", "transformers"],
    "title": "GOT-OCR 2.0 — DIMER E2E handwritten-line adaptation tutorial (standalone)",
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
    "capability": "optical character recognition (one image → its text, plain or formatted) and bounded supervised fine-tuning of the last decoder layers on transcribed text lines, using the pinned `stepfun-ai/GOT-OCR-2.0-hf` weights",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `stepfun-ai/GOT-OCR-2.0-hf` snapshot (a 1.12 GB `model.safetensors`; no pickle is opened anywhere), fetches "
        "the first eight row groups of the Belfort-line test shard from the Hugging Face Hub at an immutable revision with "
        "HTTPS range requests (about 44 MB; each row group refused on any SHA-256 or byte-total mismatch), validates the 800 "
        "line records and splits them by line into 600 / 60 / 140, recognises a synthetic printed page through the inference "
        "contract with an input manifest and a rejection probe, measures the frozen model's character and word error rates "
        "over the 140 held-out lines beside an empty-string and a constant-transcript baseline, runs a bounded fine-tuning of "
        "the last four decoder layers on cached frozen-prefix hidden states with validation-CER epoch selection, scores the "
        "held-out lines again, re-reads six held-out lines and the printed page with the adapted model, exports the adapter as "
        "safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify transcript parity. The default "
        "path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration "
        "edit (NOTEBOOK_SPEC 2.0 §5). On a Tesla T4 the default path took about 23 minutes of cell time, 27 with the pinned install and its restart (six epochs "
        "924 s, frozen scoring of 140 lines 159 s); a CUDA runtime is used automatically when present, "
        "and **a CPU runtime is not practical for the default path** (autoregressive decoding of some 800 lines plus 600 "
        "cached forwards of a 560M-parameter model)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "of line images plus a `transcripts.csv` (`file`, `text`, optional `id`; one row per image, at least eight images). The "
        "records pass through the same validation, image-disjoint split, baselines, fine-tuning, held-out evaluation, artifact "
        "export and reload-parity cells as the Belfort sample. Uploaded files stay inside this runtime. BYOD is optional and "
        "never part of the default path."
    ),
    "intro": (
        "`stepfun-ai/GOT-OCR-2.0-hf` is the GOT-OCR 2.0 model of Wei et al. (2024) in its Transformers-native release: a "
        "ViT-style vision encoder over a 1024×1024 image whose features are compressed by two stride-2 convolutions into "
        "**256 image tokens**, and a 24-layer Qwen2-architecture decoder (hidden size 1024, tied input and output "
        "embeddings; 560,528,640 parameters in all, published under the **Apache-2.0** licence) that reads those tokens "
        "wrapped in a chat prompt and generates the text greedily until the `<|im_end|>` stop token or the budget. The output "
        "is **generated text with no score**: fluent output is not evidence that it matches the image.\n\n"
        "What this notebook adds to inference is **adaptation on transcribed lines**. GOT-OCR 2.0 was trained on printed "
        "documents, formulas, tables and charts; nineteenth-century French council minutes in cursive handwriting — the "
        "Belfort-line dataset — are far outside that distribution, and on them the frozen model reads almost nothing: a "
        "character error rate of **1.345** on the 140 held-out lines (the build record's Tesla T4 figure), *worse* "
        "than predicting the empty string (CER 1.0 by construction) because it generates text the lines do not carry. So the "
        "honest question is narrow: does a bounded fine-tuning of the last four decoder layers on 600 transcribed lines move "
        "the held-out **CER** and **WER** on a line-disjoint test split past two **non-adapted baselines** and the frozen "
        "model? Nothing here is a claim about your documents or your script: it is one seeded split of one small labelled "
        "set.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (an 8-file manifest with the tokenizer files) — no "
        "pickle is opened anywhere in this notebook. Section 3 stages and digest-verifies those files before the processor or "
        "the model is constructed. The pipeline loads the checkpoint in **float32 on every device** (the weights are stored "
        "as bfloat16): the adapter is trained in float32 and overlays without a cast, and CPU, Tesla-class and consumer GPUs "
        "then run the same arithmetic."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch a digest-pinned labelled line set with its transcripts, validate it and split it by line "
        "without leakage; recognise a synthetic printed page through the public API and read the output contract correctly "
        "(generated text, no score, a `truncated` flag, error rates against a self-rendered page are not a benchmark); measure "
        "the frozen model's corpus CER and WER beside two non-adapted baselines and read what a rate above 1.0 means; run a "
        "bounded fine-tuning with the causal language-model loss, explicit hyperparameters and validation-based epoch "
        "selection; evaluate on a line-disjoint test split; look at the adapted transcripts next to the frozen ones and the "
        "references; and export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "PDF or multi-page documents (one image per call; the upstream multi-page mode is not exposed), region-guided OCR by "
        "box or colour, layout or reading-order output, rendering the formatted output, sampling or beam search, fine-tuning "
        "of the vision encoder, the projector, the embeddings or the first twenty decoder layers, evaluation on an OCR "
        "benchmark proper (only one seeded 800-line sample is scored here), and any claim that French cursive minutes stand "
        "in for your documents. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime with a CUDA GPU (Google Colab or Kaggle GPU, Python 3.12). The default path uses CUDA automatically when present. Generation is batched — every plain-mode prompt is the same 286 tokens (256 image tokens plus the chat template), so a batch needs no padding — and the build record measured 159 s to score 140 lines and 924 s for the six epochs (caching the frozen prefix for 600 lines took 362 s) on a Tesla T4, about 23 minutes of cell time for the whole path (27 with the pinned install); a CPU runtime would take hours. The pinned `torch==2.14.0` install and the 1.12 GB checkpoint are the large downloads of the run; the row groups are about 44 MB.",
        "- **Knowledge:** basic Python and PIL; what a vision–language model's generated tokens are; what character and word error rate measure and why they are not capped at 1; why a self-rendered page is a plumbing check while a held-out split of one labelled set is a measurement of that set only.",
        "- **Data contract:** records are `{id, image, text}` — `image` a PIL image (or a file decodable by Pillow) with sides within 16..16,384 px and at most 4096² pixels, `text` its transcript (1..512 characters after whitespace runs are collapsed; case and punctuation kept). Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..5,000 records; splitting de-duplicates by decoded pixels so no image lands in two splits. BYOD accepts one zip (or directory) of images plus a `transcripts.csv` in the layout named above.",
        "- **Validation is structural, not semantic:** every image is decoded and every transcript checked for length, but nothing checks that a transcript says what its image shows — a mislabelled set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path reads eight row groups of `default/test/0000.parquet` from `https://huggingface.co/datasets/Teklia/Belfort-line/resolve/<revision>/` at the immutable parquet-conversion revision `c4a74bbd…` with HTTPS range requests (the parquet footer plus about 44 MB of row-group bytes out of a 210 MB shard), each row group pinned by SHA-256 and byte total in the carried `samples.py` and refused on any mismatch. Belfort-line is published under the MIT licence (Teklia; Tarride et al. 2023); nothing is redistributed by this repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Belfort lines, the transcripts and the split\n\n"
                "`fetch_corpus` returns the eight pinned row groups from the cache under `weights/belfort/` or the Hub at the "
                "pinned parquet-conversion revision — `pyarrow` reads the shard's footer and exactly those row groups over "
                "HTTPS range requests; every cached file is re-hashed and every fetched row group refused on any SHA-256 or "
                "byte-total mismatch — and `read_corpus` turns each row into a record: the line image (128 px tall, 180 to "
                "8,956 px wide) and its crowdsourced transcript with whitespace runs collapsed. `build_sample_dataset` draws a "
                "seeded line-level split (600 / 60 / 140). `validate_dataset` then checks every record against the contract, "
                "`check_split_disjoint` asserts no image (by decoded-pixel digest) is shared, and the training split's summary "
                "table is written to `outputs/{stem}_train.csv`.\n\n"
                "Look for: 800 lines and 33,117 reference characters, three digests, and four refusal probes — a duplicate id, "
                "an empty transcript, an image below the side floor, and a dataset too small to use — each rejected before the "
                "model does anything."
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_zip = Path('work') / 'byod.zip'\n"
                "    byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_zip.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_zip)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    t0 = time.perf_counter()\n"
                "    corpus_groups = fetch_corpus(cache_dir='weights/belfort')\n"
                "    corpus = read_corpus(corpus_groups)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} @ {{CORPUS_REVISION[:12]}} ({{CORPUS_LICENSE}})'\n"
                "    raw_rows = {{'row_groups': len(corpus_groups), 'lines': sum(len(v) for v in corpus_groups.values()), 'bytes': sum(len(r['image']) + len(r['text'].encode('utf-8')) for v in corpus_groups.values() for r in v), 'seconds': round(time.perf_counter() - t0, 1)}}\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'chars': manifest['text_chars'], 'words': manifest['text_words']['total'], 'width': manifest['image_width'], 'height': manifest['image_height'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "example['image'].save('outputs/{stem}_example_line.png')\n"
                "print({{'example': {{'id': example['id'], 'image': list(example['image'].size), 'text': example['text']}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'empty transcript': [{{**train_records[0], 'text': '   '}}, *train_records[1:8]],\n"
                "    'image below the side floor': [{{**train_records[0], 'image': Image.new('RGB', (200, 8))}}, *train_records[1:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Recognise a synthetic printed page through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: a deterministic 1000×520 "
                "notice page — a heading and seven lines of English text with dates, numbers, a postcode and punctuation — "
                "rendered in code with Pillow's bundled font, with the lines drawn on it as the reference transcript; a "
                "different image family from the handwritten lines, and a page the adapted model will read again in "
                "Section 9. `validate_inputs` applies exactly the checks `recognize` applies (image sides "
                "`MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` and at most `MAX_IMAGE_PIXELS`, a mode in `MODES`, a budget in "
                "`[1, MAX_NEW_TOKENS]`) and returns an input manifest; a mode the pipeline does not expose is validated too and "
                "its rejection recorded as a finding. `recognize` returns the decoded text with the stop string removed, the "
                "token count and a `truncated` flag — **no score exists**, and the mode and the token budget are caller-owned "
                "request parameters. The per-image `evaluation_report` against the self-rendered transcript is `sample-sanity` "
                "(character and word error rate on one page) — plumbing evidence, not a measurement; whether the model is *good "
                "at handwriting* is what Section 6 measures on 140 lines. The inference-only card recorded a character error "
                "rate of 0.0 on this page in `plain` mode."
            ),
            "code": (
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
                "    canvas = Image.new('RGB', (width, height), 'white')\n"
                "    d = ImageDraw.Draw(canvas)\n"
                "    head, body = ImageFont.load_default(size=30), ImageFont.load_default(size=22)\n"
                "    d.text((60, 40), LINES[0], fill='black', font=head)\n"
                "    y = 110\n"
                "    for line in LINES[1:]:\n"
                "        d.text((60, y), line, fill=(20, 20, 20), font=body)\n"
                "        y += 48\n"
                "    return canvas, '\\n'.join(LINES)\n\n\n"
                "page, page_reference = printed_page()\n"
                "page_name = 'synthetic_notice_1000x520.png'\n"
                "page_sha256 = hashlib.sha256(np.asarray(page).tobytes()).hexdigest()\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_IMAGE_PIXELS': MAX_IMAGE_PIXELS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DEFAULT_LINE_MAX_NEW_TOKENS': DEFAULT_LINE_MAX_NEW_TOKENS, 'MODES': list(MODES), 'DECODING': DECODING, 'STOP_STRING': STOP_STRING, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'device': pipe.device, 'dtype': pipe.dtype}}}})\n"
                "input_manifest = validate_inputs(page, mode='plain', max_new_tokens=DEFAULT_MAX_NEW_TOKENS, names=[page_name])\n"
                "try:\n"
                "    validate_inputs(page, mode='multi-page')\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'unsupported-mode-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'page': page_name, 'sha256': page_sha256[:16] + '...', 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n\n\n"
                "def recognise_page(pipeline, label):\n"
                "    started = time.perf_counter()\n"
                "    result = pipeline.recognize(page, mode='plain', max_new_tokens=DEFAULT_MAX_NEW_TOKENS)\n"
                "    elapsed = time.perf_counter() - started\n"
                "    checks = {{\n"
                "        'text_is_str': isinstance(result['text'], str),\n"
                "        'stop_string_removed': STOP_STRING not in result['text'],\n"
                "        'not_truncated': result['truncated'] is False,\n"
                "        'identity_reported': result['model_id'] == MODEL_ID and result['model_revision'] == MODEL_REVISION,\n"
                "    }}\n"
                "    if not all(checks.values()):\n"
                "        raise RuntimeError(f'recognize output failed a sanity check: {{checks}}')\n"
                "    report = evaluation_report(result, page_reference, sample_kind='synthetic (rendered in this notebook)')\n"
                "    with open(f'outputs/{stem}_page_{{label}}.txt', 'w', encoding='utf-8') as handle:\n"
                "        handle.write(result['text'] + '\\n')\n"
                "    print({{label: {{'seconds': round(elapsed, 2), 'new_tokens': result['new_tokens'], 'checks': checks, 'metrics': {{m['id']: round(m['value'], 4) for m in report['metrics']}}, 'verdict': report['verdict']}}}})\n"
                "    print(result['text'][:160])\n"
                "    return result, report\n\n\n"
                "frozen_page_result, frozen_page = recognise_page(pipe, 'frozen')"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model on the test lines\n\n"
                "Two non-adapted baselines frame the adaptation, each scored by `ocr_metrics` (carried in `metrics.py`): the "
                "**character error rate** and **word error rate** as micro averages — total Levenshtein edits over total "
                "reference characters or words, the corpus CER/WER of the handwriting-recognition literature — beside the "
                "macro (per-line mean) rates and the exact-match rate. Neither rate is capped: a hypothesis longer than its "
                "reference pushes the rate **above 1.0**, the signal that the model is generating text the line does not carry. "
                "The **empty-string** baseline predicts nothing and scores CER 1.0 exactly (every reference character is a "
                "deletion) — the floor any recogniser must beat to do better than silence. The **constant-transcript** baseline "
                "predicts one training transcript — the medoid, the line closest on average to the others — for every test "
                "line: what corpus statistics buy without reading the image. The **frozen model** is scored by `pipe.evaluate`, "
                "which transcribes the lines in batches of `EVAL_BATCH_SIZE` under a `LINE_MAX_NEW_TOKENS` budget and returns "
                "the hypotheses with the rates. Expect the frozen model **below the empty baseline**: the build record measured "
                "1.345 (its hypotheses were 1.6 times the reference length, mostly invented "
                "printed-looking words); read four of them under the references to see what a rate above 1.0 looks like."
            ),
            "code": (
                "METRICS = ('cer', 'wer', 'cer_macro', 'exact_match')\n"
                "LINE_MAX_NEW_TOKENS = 128  # @param {{type:\"integer\"}}\n\n"
                "baseline_empty = empty_baseline(test_records)\n"
                "baseline_constant = constant_baseline(train_records, test_records)\n"
                "print({{'empty_baseline': {{k: round(baseline_empty[k], 3) for k in METRICS}}, 'n': baseline_empty['n'], 'note': baseline_empty['baseline']}})\n"
                "print({{'constant_baseline': {{k: round(baseline_constant[k], 3) for k in METRICS}}, 'note': baseline_constant['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, max_new_tokens=LINE_MAX_NEW_TOKENS, batch_size=EVAL_BATCH_SIZE)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'ref_chars': frozen_test['ref_chars'], 'hyp_chars': frozen_test['hyp_chars'], 'truncated': frozen_test['truncated'], 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "for record, hypothesis in zip(test_records[:4], frozen_test['hypotheses'][:4], strict=True):\n"
                "    print({{'id': record['id'], 'reference': record['text'], 'frozen': hypothesis[:120]}})"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the last decoder layers\n\n"
                "`pipe.adapt` trains only the last four of the 24 decoder layers and the final norm — 51,401,728 of 560,528,640 "
                "parameters — while the vision encoder, the projector, the embeddings (tied to the output head) and the first "
                "twenty decoder layers stay frozen. Each training line is the prompt the processor builds (the 256 image tokens "
                "and the chat template) followed by the transcript's tokens and the stop token; the loss is the **causal "
                "language-model cross-entropy** over the transcript tokens with the prompt masked out — the checkpoint's own "
                "training objective. Because everything before layer 20 is frozen, its output for every training line is "
                "computed once under no gradient and cached (the **frozen-prefix cache**), and each step runs only the four "
                "trainable layers, the norm and the output head on those cached states — the loss equals the full model's loss "
                "exactly, at a fraction of the cost. AdamW without weight decay at a fixed learning rate, gradient clipping at "
                "1.0, seeded shuffling, no scheduler, no augmentation. Epoch 0 records the frozen model's validation rates; "
                "every epoch is scored on the 60 validation lines, and the epoch with the **lowest validation CER** is kept.\n\n"
                "Watch the validation CER fall from 1.554 to 0.634 (epoch 4 in the build "
                "record) while the loss drops from about 3.99 to 0.05: four layers are enough to move the "
                "decoder from inventing printed words to reading cursive French — half the characters right, most words still "
                "wrong. The build record's counter-examples are about data, not the rate: the same recipe on the first 400 lines "
                "only (280 training lines) reached 1.302 on its own 80-line test split at lr 5e-5 and 1.040 at lr 1e-4 — worse "
                "than silence — while 600 training lines reached 0.759 on 140; the default is the corpus size that "
                "moved the held-out number."
            ),
            "code": (
                "EPOCHS = 6  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, max_new_tokens=LINE_MAX_NEW_TOKENS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'first_trainable_layer': adapt_result['first_trainable_layer'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'loss': adapt_result['loss'], 'cache_seconds': adapt_result['cache_seconds'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test lines were never used for training or epoch selection, and no image appears in two splits. The "
                "adapted model is scored exactly as the frozen model was in Section 6 and the four systems are put side by "
                "side. Read it in this order: **CER** first (the measure the epoch was selected on — the build record measured "
                "1.345 → **0.759**, past both baselines), then **WER** (1.780 → 1.017: "
                "whole words, not just characters), then the hypothesis length (from 1.6 times the reference "
                "length to 1.1: the adapted model stops inventing text), then the exact-match rate "
                "(3 of the 140 lines read perfectly). The cell asserts the adapted CER is below the frozen one and "
                "below the empty baseline's 1.0. Eighty lines from one seeded split give **no dispersion estimate**; the deltas "
                "are sample-sanity evidence that the adaptation contract works, not a benchmark, and a gain on one French "
                "council's minutes says nothing about other hands, other languages or other scripts until you measure them."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, max_new_tokens=LINE_MAX_NEW_TOKENS, batch_size=EVAL_BATCH_SIZE)\n"
                "adapted_val = pipe.evaluate(val_records, max_new_tokens=LINE_MAX_NEW_TOKENS, batch_size=EVAL_BATCH_SIZE)\n"
                "comparison = {{metric: {{'empty': round(baseline_empty[metric], 3), 'constant': round(baseline_constant[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['hypothesis_length'] = {{'ref_chars': adapted_test['ref_chars'], 'frozen_hyp_chars': frozen_test['hyp_chars'], 'adapted_hyp_chars': adapted_test['hyp_chars'], 'frozen_truncated': frozen_test['truncated'], 'adapted_truncated': adapted_test['truncated']}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'baselines': {{'empty': {{k: v for k, v in baseline_empty.items() if k != 'rows'}}, 'constant': {{k: v for k, v in baseline_constant.items() if k != 'rows'}}}},\n"
                "    'frozen_test': {{k: v for k, v in frozen_test.items() if k != 'rows'}},\n"
                "    'validation_metrics': {{k: v for k, v in adapted_val.items() if k != 'rows'}},\n"
                "    'test_metrics': {{k: v for k, v in adapted_test.items() if k != 'rows'}},\n"
                "    'per_line': [{{**frozen_row, 'frozen_hypothesis': frozen_hyp, 'adapted_cer': adapted_row['cer'], 'adapted_hypothesis': adapted_hyp}} for frozen_row, frozen_hyp, adapted_row, adapted_hyp in zip(frozen_test['rows'], frozen_test['hypotheses'], adapted_test['rows'], adapted_test['hypotheses'], strict=True)],\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['cer'] < frozen_test['cer']\n"
                "assert adapted_test['cer'] < baseline_empty['cer']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'adapted_beats_both_baselines': adapted_test['cer'] < min(baseline_empty['cer'], baseline_constant['cer'])}})"
            ),
        },
        {
            "md": (
                "## 9. Look at the lines, export the adapter and reload it\n\n"
                "Six held-out lines are written as panels (`outputs/{stem}_examples/`: the line image with the reference, the "
                "frozen transcript and the adapted transcript beneath it) so the numbers can be checked by eye: the adapted "
                "rows should read the cursive the frozen rows replaced with invented print. The printed page from Section 5 "
                "is then read again by the adapted model — an image family the adaptation never saw, so this is a small look "
                "at what the adaptation did *outside* its corpus: the build record measured a character error rate of "
                "0.000 on the page after adaptation, the same as before — four decoder layers tuned on cursive French still "
                "read clean print exactly on this one page, an observation to record, not a guarantee against forgetting.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the four layers and the norm, about 206 MB in float32 — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and revision, "
                "the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the training "
                "configuration and the epoch history (OUT8). `GotOcr2Pipeline.from_artifact` re-verifies the base snapshot, "
                "checks the artifact manifest, its digest and its exact tensor set **before** deserialising, refuses any tensor "
                "outside the last four decoder layers and the norm, and overlays the tensors onto a freshly loaded base — a new "
                "object from files, not the in-memory model (VER2). The cell asserts identical transcripts on eight test lines "
                "(VER4)."
            ),
            "code": (
                "import shutil\n\n"
                "examples_dir = Path('outputs/{stem}_examples')\n"
                "shutil.rmtree(examples_dir, ignore_errors=True)\n"
                "examples_dir.mkdir(parents=True)\n"
                "caption_font = ImageFont.load_default(size=18)\n"
                "for record, frozen_hyp, adapted_hyp in zip(test_records[:6], frozen_test['hypotheses'][:6], adapted_test['hypotheses'][:6], strict=True):\n"
                "    line = record['image']\n"
                "    width = min(1400, line.width)\n"
                "    line = line.resize((width, max(1, round(line.height * width / record['image'].width))))\n"
                "    sheet = Image.new('RGB', (max(width, 1400), line.height + 96), (255, 255, 255))\n"
                "    sheet.paste(line, (0, 0))\n"
                "    marker = ImageDraw.Draw(sheet)\n"
                "    for i, (tag, text) in enumerate((('REF', record['text']), ('FROZEN', frozen_hyp), ('ADAPTED', adapted_hyp))):\n"
                "        marker.text((8, line.height + 6 + i * 28), f'{{tag}}: {{text[:140]}}', fill=(20, 20, 20) if tag != 'FROZEN' else (150, 40, 40), font=caption_font)\n"
                "    sheet.save(examples_dir / f\"{{record['id']}}.png\")\n"
                "print({{'examples': sorted(p.name for p in examples_dir.iterdir()), 'rows': ['reference', 'frozen transcript', 'adapted transcript']}})\n\n"
                "adapted_page_result, adapted_page = recognise_page(pipe, 'adapted')\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...', 'best_epoch': artifact_manifest['adapter']['best_epoch']}})\n\n"
                "reloaded = GotOcr2Pipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [item['text'] for item in pipe.transcribe([r['image'] for r in test_records[:8]], max_new_tokens=LINE_MAX_NEW_TOKENS)]\n"
                "after = [item['text'] for item in reloaded.transcribe([r['image'] for r in test_records[:8]], max_new_tokens=LINE_MAX_NEW_TOKENS)]\n"
                "parity = {{'identical_lines': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_lines'] == parity['of']\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': pipe.weight_sha256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'file': CORPUS_FILE, 'license': CORPUS_LICENSE, 'language': CORPUS_LANGUAGE, 'row_groups': sorted(ROW_GROUP_PINS), 'shard_bytes': CORPUS_BYTES}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'page': {{'name': page_name, 'sha256': page_sha256, 'reference_chars': len(page_reference)}}, 'frozen_report': frozen_page, 'adapted_report': adapted_page, 'output_files': ['outputs/{stem}_page_frozen.txt', 'outputs/{stem}_page_adapted.txt']}},\n"
                "    'comparison': comparison,\n"
                "    'examples': 'outputs/{stem}_examples',\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': pipe.dtype}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A document OCR model reads the print it was trained on, and nineteenth-century cursive French is not among it: the "
        "frozen model scores a character error rate of 1.345 on the Belfort lines — worse than silence, because it "
        "invents printed-looking text for strokes it cannot read. A bounded fine-tuning of the last four decoder layers on 600 "
        "transcribed lines teaches it to read the hand (0.759 CER and 1.017 WER in the build record, "
        "3 of the held-out lines exact), with a 206 MB adapter that reloads line-for-line. That is the claim: "
        "the adaptation contract works end to end on a real labelled set, and the numbers it produces are read as micro and "
        "macro rates, against two non-adapted baselines and the frozen model, with the hypothesis length beside them rather "
        "than in isolation.\n\n"
        "The test split is 140 lines from one seeded draw of one 800-line sample, the validation split that picks the epoch is "
        "60, and both rates are corpus edit distances over one crowdsourced transcription — not a benchmark, not a measure of "
        "reading order or layout. So a gain here says the contract works on one council's minutes, not that the adapted model "
        "handles other hands, other languages, other scripts or your scans; and fine-tuning on a narrow domain can move the "
        "model elsewhere — the printed page re-read in Section 9 was still read exactly, one image of evidence that four "
        "decoder layers did not forget clean print, not a measurement of what else they may have moved. The decoder was adapted, not the vision encoder: "
        "what the encoder cannot resolve in a 128-px line squashed into a 1024×1024 square stays unread.\n\n"
        "Three things to carry to real data. **Baselines first:** the empty and constant-transcript rates on *your* transcripts, "
        "and the frozen model's hypothesis length, are the numbers to read before any adapted one. **Rates above 1.0:** an "
        "uncapped CER tells you the model is generating, not reading; a capped one would hide it. **Leakage:** keep every image "
        "in one split (the contract de-duplicates by decoded pixels) and split by page, writer or volume when your lines come "
        "from few sources — lines cut from the same page share a hand.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled line set, validate the "
        "demonstrated dataset contract without leakage, execute the inference contract and a bounded fine-tuning with the "
        "model's own objective, evaluate against two non-adapted baselines and the frozen model on a line-disjoint split, and "
        "emit the shown machine-readable artifacts — without the repository being reachable. It does **not** establish "
        "benchmark superiority, recognition quality on any other hand, language or document family, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `LEARNING_RATE` to `1e-4` and read a faster, noisier "
        "validation curve (the build record's 280-line probe reached 0.674 validation CER at that rate and still failed its "
        "test split); raise `EPOCHS` and watch the validation CER pick the epoch; lower `LINE_MAX_NEW_TOKENS` "
        "to `64` and read how the truncation count and the frozen hypothesis length change; re-run Section 5 in `format` mode "
        "(`mode='format'`) to see what markup the adapted decoder still emits; or bring your own transcribed lines through "
        "BYOD and read the two baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/got-ocr2-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/got-ocr2-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/got-ocr2-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/Ucas-HaoranWei/GOT-OCR2.0\n"
        "- General OCR Theory: Towards OCR-2.0 via a Unified End-to-end Model (Wei et al., 2024): https://arxiv.org/abs/2409.01704\n"
        "- Belfort-line dataset (Teklia, MIT): https://huggingface.co/datasets/Teklia/Belfort-line — Tarride et al., Handwritten Text Recognition from Crowdsourced Annotations (HIP 2023): https://doi.org/10.1145/3604951.3605517\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
