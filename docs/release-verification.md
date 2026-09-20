# Release verification

`tutorials/got_ocr2_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the exact
notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record for the
notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the Belfort-line
  parquet-conversion revision `c4a74bbd39f2df314752e7e6026649a39d365cbb` is the one other 40-hex revision the
  documents may cite);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `GotOcr2Pipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path, `read_corpus`,
  `build_sample_dataset(corpus, seed=SPLIT_SEED)` / `load_byod_dataset` + `split_dataset`, `validate_dataset` per
  split, `check_split_disjoint`, `write_dataset_csv`, the four dataset refusal probes, the ceiling print,
  `validate_inputs` with the unsupported-mode refusal probe, `recognize` with the sanity checks and the per-image
  `evaluation_report` on the synthetic page, `empty_baseline`, `constant_baseline`, `pipe.evaluate` on the frozen
  model, `pipe.adapt` with its explicit hyperparameters, `pipe.evaluate` on the validation and test splits after
  adaptation with the two CER assertions, `recognize` + `evaluation_report` on the page after adaptation, the example
  panels, `pipe.save_artifact`, `GotOcr2Pipeline.from_artifact` and the transcript-parity assertion, and the result
  fields `weight_file` / `weight_format` / `weight_sha256` and the `corpus` block), the seven expected `outputs/`
  paths, the learner-facing statements (Apache-2.0 weights, 256 image tokens, generated text with no score,
  adaptation on transcribed lines, the two non-adapted baselines, CER and WER, rates above 1.0, the empty-string and
  constant-transcript baselines, the causal language-model loss, the frozen-prefix cache, lowest validation CER, no
  dispersion estimate, float32 on every device, the leakage guidance, the excluded tasks, the snapshot note) and the
  gated-off BYOD default; forbidden patterns (credential-in-URL, any `git clone` / `github.com/kurtvalcorza` /
  repository import on the primary path, a mutable `revision='main'`, direct `from transformers import` /
  `AutoModelForImageTextToText` / `AutoProcessor` / `stop_strings=` / `model.generate(` / `torch.inference_mode(` /
  `from huggingface_hub import` / `urllib.request` / `pyarrow` / `safetensors` imports / `torch.optim` /
  `.backward(` / `requires_grad` / `pipe._model` / `pipe._processor` / `extractall(` use **outside the carried module
  cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `huggingface-hub`, `numpy`,
`pillow` and `pyarrow`, the package with `--no-deps`, runs `ruff check src tests tools`, `tools/build_notebook.py
--check`, and the unit suite (`tests/`, including `test_adaptation.py`, `test_import_boundary.py`,
`test_role_helpers.py`, `test_notebook_parity.py`; injected runner and parquet opener, no weights —
`tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (CUDA; a CPU runtime is not practical for the default path) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Kaggle script kernel (pre-flight only) | Fresh GPU container that clones the candidate branch, installs the pins and runs `tests/test_model_backed.py` plus the package-API recipe probe | Builder pre-flight to catch defects and fix the recipe before spending a notebook run; **not** promotion evidence for the notebook blob |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CUDA runtime (Colab GPU, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/got-ocr-2.0-hf/` or the row-group cache `weights/belfort/` (the standalone path writes the
   manifest itself, stages the missing files from the Hub and reads the pinned row groups over range requests, so
   neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `LINE_MAX_NEW_TOKENS = 128`, `EPOCHS = 6`, `LEARNING_RATE = 5e-5`,
   `BATCH_SIZE = 8`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`,
   `pillow==11.3.0`, `huggingface-hub==0.36.2`, `pyarrow==25.0.1` (an interpreter restart after the install is
   expected where the runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `GotOcr2Pipeline`, `verify_snapshot`, `stage_missing_files`,
     `validate_inputs`, `evaluation_report`, `character_error_rate`, `word_error_rate`, `edit_distance`,
     `normalise_text`, `ocr_metrics`, `empty_baseline`, `constant_baseline`, `medoid_transcript`, `fetch_corpus`,
     `read_corpus`, `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `split_dataset`,
     `load_byod_dataset`, `write_dataset_csv` and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 8 manifest entries fetched from `stepfun-ai/GOT-OCR-2.0-hf` at the immutable
     revision on a clean runtime, `verify_snapshot` returning its dict (8 files, the 1.12 GB `model.safetensors`
     re-hashed), and `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory on `cuda:0` in
     float32;
   - Section 4: `fetch_corpus` reading the eight pinned row groups over HTTPS range requests with every SHA-256 and
     byte total matching (800 lines, about 44 MB); the seeded split into 600 / 60 / 140 with `check_split_disjoint`
     reporting no shared image and the three dataset digests printed; `outputs/…_train.csv` and
     `outputs/…_example_line.png` written; the four dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`MIN_IMAGE_SIDE` 16, `MAX_IMAGE_SIDE` 16384, `MAX_IMAGE_PIXELS` 16777216,
     `MAX_NEW_TOKENS` 4096, `DEFAULT_MAX_NEW_TOKENS` 1024, `DEFAULT_LINE_MAX_NEW_TOKENS` 128, the two `MODES`,
     `DECODING` greedy, `STOP_STRING`, `MIN_RECORDS` 8, `MAX_RECORDS` 5000, `MAX_TEXT_CHARS` 512) surfaced; the
     synthetic 1000×520 page rendered; `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`,
     one recorded rejection finding from the unsupported-mode probe); `recognize` on the page with every sanity check
     `True`, `outputs/…_page_frozen.txt` written and the per-image `evaluation_report` verdict `sample-sanity` (the
     inference-only card recorded a character error rate of 0.0 on this page — an observation, not an assertion);
   - Section 6: the empty baseline (CER 1.0 exactly), the constant-transcript baseline (≈ 1.0) and the frozen model's
     test rates (≈ 1.345 CER / 1.780 WER in the Tesla T4 build record, hypotheses about
     1.6 times the reference length) with four hypotheses printed under their references;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 51,401,728 trainable of 560,528,640 parameters,
     `first_trainable_layer` 20, and a six-epoch history with the validation CER falling (build record:
     1.554 → 0.875 / 0.886 / 0.707 / 0.634 / 0.657 / 0.788, `best_epoch` 4);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison, the hypothesis
     lengths and `outputs/…_evaluation_report.json` written (the cell asserts the adapted test CER is below the frozen
     one and below 1.0 — 0.759 against 1.345 in the build record, WER 1.780 →
     1.017; the adapted model also clears the constant baseline, reported, not asserted);
   - Section 9: six example panels under `outputs/…_examples/`; the page re-read by the adapted model with the
     `sample-sanity` report and `outputs/…_page_adapted.txt` (build record: character error rate
     0.000 after adaptation, as before — an observation, not an assertion);
     `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,manifest.json}` (49 tensors, about 206 MB)
     and `GotOcr2Pipeline.from_artifact` reloading it with 8/8 identical transcripts on eight test lines (the cell
     asserts it); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity and licence, the
     snapshot block (`weight_file`, `weight_format`, `weight_sha256`), the `corpus` block, the inference-contract
     reports, the comparison, the artifact digest, the reload parity, the runtime versions, device and dtype;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the row-group cache were
   clean, outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `got_ocr2_colab.ipynb` (`E2E`) | pending | — | — | pending: the supported-runtime run of the exact candidate blob is not yet recorded |
| `got_ocr2_colab.ipynb` (`TASK-INFERENCE`, superseded) | `70b4431` / `84828c509441` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-got-ocr2` v1) | PASS — 8/8 ok code cells, 300.9 s, 18 files, 1140 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/got_ocr2_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/got_ocr2_colab.ipynb`). Wall times, when recorded, are the sum of per-cell times
reported by the executor and include installs and the model download; they are measurements for the stated runtime,
not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | package API at `91d1c00` (pre-flight, not the notebook blob) | Kaggle Tesla T4 script kernel (`kurtvalcorza/dimer-probe-got-ocr2-e2e` v3; `torch 2.14.0+cu130`, `transformers 4.57.6`, Python 3.12, `cuda:0`, float32), branch cloned, pins installed, snapshot staged from the Hub | `tests/test_model_backed.py` (7 passed in 129 s) and the recipe probe: the eight pinned row groups read over range requests (800 lines, digest `b7e1dd68…`, 11 s), empty and constant baselines, the frozen model on both test splits, three arms with validation-CER selection — 280 training lines at lr 5e-5 for eight epochs, 280 at lr 1e-4 for six, 600 at lr 5e-5 for six — adapted evaluation, artifact round trip (reload parity 8/8) | 2950 s | **PASS** — frozen 1.345 / 1.780 CER / WER on the 140-line split (29 of 140 truncated, hypotheses 1.56× the reference length); arm 600 lines × lr 5e-5 × 6: validation CER 1.554 → 0.875 / 0.886 / 0.707 / 0.634 / 0.657 / 0.788 (epoch 4 kept), cache 347 s, 883 s in all, peak 6.0 GB, test **0.759 / 1.017**, exact 3 of 140, 3 truncated; the two 280-line arms landed at 1.302 and 1.040 on their 80-line split — the chosen recipe |
| 2026-09-14 | `70b4431` / `84828c509441` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-got-ocr2` v1) | Default sample path | 300.9 s | PASS — 8/8 ok code cells, 18 files, 1140 MB staged; not evidence for the `E2E` blob |
| 2026-09-14 | notebook blob `4980a8f9c842` (`TASK-INFERENCE`, superseded) | Local Windows-venv harness (`run_nb_local.py`, `CUDA_VISIBLE_DEVICES=-1`) | Default synthetic path, all 8 code cells | 124.3 s | PASS — pre-flight only for the earlier notebook |

## Current status

**Candidate.** The `E2E` notebook has not yet executed in a supported runtime; the registry stays **Candidate** until a
clean run of the exact candidate blob is recorded above and an integrator promotes it. What exists: static validation
(`tools/validate_release_assets.py`), the generator parity checks (`--check` OK), the offline unit suite, and the Tesla
T4 pre-flight of the package API at `af9bcfe` (table above), which exercised the model-backed suite and the recipe the
notebook carries.

Facts a reviewer should weigh: the sample is nineteenth-century French cursive, far outside the checkpoint's printed
training distribution, which is why the frozen model scores above the empty baseline's CER 1.0 (it generates
printed-looking text for strokes it cannot read) and why the gain is large — it is a repair of a domain gap, not evidence
about other hands or scripts; the rates are uncapped micro CER/WER over one crowdsourced transcription and the notebook
says so; the 60-line validation split selects the epoch; the vision encoder is frozen, so what it cannot resolve in a
128-px line squashed into 1024×1024 stays unread; and the adapted decoder still read the printed page exactly (one page —
not evidence against forgetting elsewhere). Greedy decoding is deterministic on a fixed device and dtype,
but the training of four decoder layers is not bit-reproducible across GPUs, so a Kaggle number a few hundredths off
the build record is the expected spread, not a finding.
