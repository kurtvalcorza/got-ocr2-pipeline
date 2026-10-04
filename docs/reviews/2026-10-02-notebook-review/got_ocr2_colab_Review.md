# GOT-OCR 2.0 Handwritten-Line Adaptation E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/got-ocr2-pipeline`  
**Notebook:** `tutorials/got_ocr2_colab.ipynb`  
**Reviewed commit:** `a21f5ad800cd72fa517a395f8b3c6608dc87f4b2` (`main`, confirmed with `gh api repos/kurtvalcorza/got-ocr2-pipeline/commits/main`)  
**Notebook Git blob:** `951ad49e098a0634d3765c04e5d9b470279fc62b`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-20 (commit `b72789c`); the notebook last changed in `d421842`. Later commits on `main` touch only the supplemental workshop notebook and its records.  
**Finding prefix:** `GOT`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`. The notebook declares 2.0.  
**Scope:** the generated `E2E` tutorial only. The supplemental `DIMER_OCR_Document_Extraction_Workshop.ipynb` in the same folder is out of scope.

## Executive assessment

The engineering is careful. The notebook carries its three modules verbatim (the generator's `--check` and the release-asset validator both pass at this commit), digest-verifies the 1.14 GB snapshot, reads eight digest-pinned Belfort row groups over HTTPS range requests, demonstrates four dataset refusals and one input refusal, scores an empty-string and a constant-transcript baseline beside the frozen and adapted models with uncapped micro/macro CER and WER, fine-tunes the last four decoder layers with stated hyperparameters and validation-CER epoch selection, and reloads a safetensors adapter with an 8/8 transcript-parity assertion. A CPU run of the data stage in this review reproduced the recorded split exactly (600 / 60 / 140, the same three digests, `SAMPLE_DIGEST` match).

| Measure | This review (CPU, data stages only) | Kaggle T4 record (blob `951ad49e`) |
|---|---|---|
| Code cells completed | cells 3 (install skipped), 5, 7, 9, 13; model cells not run | 11/11 on pass 2; pass 1 stopped at the install guard |
| Belfort fetch | 800 lines, 42,403,899 bytes, digests match, 29.6 s | identical counts, 8.1 s |
| Split digests | `64186dce…` / `5768e17f…` / `0e815da4…` | identical |
| Test lines with a neighbouring corpus line in train | **131 / 140** | not reported |
| BYOD smallest accepted dataset | **50 images** (8, 20, 38, 49 refused) | not run |
| Test CER empty / constant / frozen / adapted | not run | 1.000 / 0.944 / 1.345 / **0.759** |
| Test WER empty / constant / frozen / adapted | not run | 1.000 / 0.999 / 1.780 / **1.017** |

Five problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (GOT-M1).** The recorded run stopped at the install cell's stale-module guard and passed only after a restart; the opening cell itself budgets "27 with the pinned install and its restart", and the repository marks the blob `Release-grade` on that run.
2. **Guided layer largely absent (GOT-M2).** Declared `GUIDED`, but there is no audience, how-to-use, roadmap, glossary, prediction prompt, checkpoint, troubleshooting section or conclusion template, and 1,338 lines of carried modules sit in three unlabelled, uncollapsed cells.
3. **The held-out split is not leakage-free in the sense the notebook teaches (GOT-M3).** 131 of 140 test lines have the line just before or after them in the corpus in the training split, while the notebook calls the split "without leakage" and advises splitting by page.
4. **The notebook's own question is answered only for CER (GOT-M4).** It asks whether adaptation moves "the held-out CER and WER … past two non-adapted baselines". On WER it does not (adapted 1.017 against empty 1.000 and constant 0.999), and no cell says so; the printed flag `adapted_beats_both_baselines: True` tests CER only.
5. **The prescribed reruns compare against the wrong model (GOT-M5).** BYOD is to be run by re-running "from that cell" (Section 4) after the full run, and the experiments by re-running Section 7; the model is loaded only in Section 3, so the Section 5 "frozen" page, the Section 6 "frozen model" and `adapt`'s epoch 0 "frozen model" are then the Belfort-adapted model.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites (cell 1): basic Python and PIL; what a vision–language model's generated tokens are; what CER and WER measure and why they are not capped; self-rendered page vs held-out split |
| Supported runtime | "a fresh supported runtime with a CUDA GPU (Google Colab or Kaggle GPU, Python 3.12)"; "a CPU runtime would take hours" |
| Promised outcomes | Pinned install; carried package; digest-verified snapshot; digest-pinned Belfort fetch, validation and split "by line without leakage"; the inference contract on a synthetic printed page (input manifest, rejection probe, sanity checks, `sample-sanity` rates); frozen CER/WER beside two baselines; bounded fine-tuning with explicit hyperparameters and validation epoch selection; line-disjoint test evaluation answering whether CER **and WER** move past both baselines; six panels and the page re-read after adaptation; safetensors adapter with reload parity; BYOD zip through the same cells |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `941ec15b` (its three modules are byte-identical to `main`) |
| Release status | **`Release-grade`** (`STATUS.md`, `README.md`, `tutorials/README.md`, `docs/release-verification.md` Current status) |

### Evidence actually obtained

- **Source inspection.** All 25 cells (11 code; cells 5, 7, 9 are the carried `pipeline.py` 852 lines, `metrics.py` 99 lines, `samples.py` 387 lines). Also read: the generator, template and `tools/validate_release_assets.py` (relevant parts); `pipeline.py` (`adapt`, `evaluate`, `save_artifact`, `from_artifact`), `samples.py` (`build_sample_dataset`, `split_dataset`, `load_byod_dataset`, `validate_dataset`); `README.md`, `STATUS.md`, `tutorials/README.md`, `docs/release-verification.md`. The repository has no `AGENTS.md`; `docs/execution-evidence/` holds only workshop-notebook evidence.
- **Documented execution evidence.** `docs/release-verification.md` row 2026-09-20 and the workspace run archive (`.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-got-ocr2/v3/evidence/`: `run_summary.json`, `executed-pass1.ipynb`, `executed.ipynb`, `outputs/`): Kaggle Tesla T4, **the reviewed blob** (SHA-1 verified before execution), clean HF cache; pass 1 `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.`, `restarted_after_install_cell: true`, pass 2 11/11, 1543.4 s. No Colab run of this blob; no BYOD run; no experiment run.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), Python 3.12.10, torch 2.14.0+cpu, transformers 4.57.6, numpy 2.5.3, pillow 11.3.0, pyarrow 25.0.1, safetensors 0.8.0, huggingface-hub 0.36.2 — the notebook's pins, taken read-only from another workspace `.venv`. Nothing was installed. Cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, the notebook's executor hook.
  - **Probes (about 60 s):** P0 static checks; P1 the three carried cells executed in one namespace and compared with the package (no colliding names, no differing constants); P2 cell 13 at defaults from an empty working directory (real Hub fetch of the eight row groups); P3 cell 13's BYOD branch through a shimmed `google.colab.files.upload` with eleven archives; P4 metric sanity; P5 `tools/build_notebook.py --check` (exit 0) and `tools/validate_release_assets.py` (exit 0, "static source validation only"); P6 `adapt` rerun semantics from source; P7 the numbers in the archived Kaggle run.
- **Not verified:** Sections 3 and 5–9 (model load, inference, baselines on the model, fine-tuning, evaluation, export, reload) beyond the Kaggle record; any Colab run; the real upload dialog; BYOD beyond the validation/split stage; every optional experiment.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** strong supply-chain and data-integrity handling (row groups refused on any digest mismatch; P2 digests identical to the Kaggle record; generator parity holds). Defects: the install pattern forces a restart (GOT-M1); `adapt` takes the pipeline's current weights as its starting point and labels them "frozen model", so every documented rerun compounds training on the already adapted model (GOT-M5); the held-out assertions turn a negative result into a crash (GOT-m2).
- **Scientific validity:** two baselines, uncapped micro/macro rates, hypothesis length, validation-only epoch selection and an untouched test split are sound. The weak points are the split unit (GOT-M3), the CER-only reading of a question posed for CER and WER (GOT-M4), and expected-output prose that hides the validation curve's overfitting and quotes a wrong test-split size (GOT-m3).
- **Promise fulfilment:** default-path promises are met on the documented run, except one-pass `Run all` (GOT-M1), "without leakage" (GOT-M3) and the WER half of the stated question (GOT-M4). The BYOD path's stated minimum is wrong (GOT-m1), its documented rerun measures against the adapted model (GOT-M5), and a legitimate negative result stops before export (GOT-m2).
- **Learner experience:** precise prose, "Look for"/"Watch"/"Expect" notes, four frozen hypotheses printed to show what a rate above 1.0 looks like, panels by eye, and a careful closing interpretation; but no guided layer and no troubleshooting (GOT-M2), and the experiments name no rerun scope (GOT-m4).
- **Spec conformance:** unresolved applicable MUSTs — RUN1, RUN10, ENV6, REL2, REL11 (GOT-M1); SPL3, SPL5 (GOT-M3); EVAL3, EVAL15 (GOT-M4); DAT13, DAT14 (GOT-M5); DAT12, DAT19, VAL7 (GOT-m1); DAT13 (GOT-m2); ENV8 (GOT-m3). SHOULD deviations: GDL1–GDL14, UX8 (GOT-M2); GDL14 (GOT-M4); GDL10 (GOT-M5); EXE2, UX10 (GOT-m1); RUN9, UX10 (GOT-m2); GDL8 (GOT-m3); GDL10, UX5, EXE1 (GOT-m4); EXE5 (GOT-S2).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all` | cell 3 in-kernel `pip install` + stale-module guard | Kaggle pass 1 `RuntimeError`, restart, pass 2 11/11 | cell 0 budgets "its restart" in a timing sentence | **Not met** (GOT-M1) |
| Digest-verified pinned snapshot | cell 11 | 8/8 fetched and verified, `cuda:0`, float32 (Kaggle) | clear | Met (documented) |
| Digest-pinned Belfort fetch, validation, refusals | cell 13 | P2: 800 lines, digests match, four refusals with clear messages; identical to Kaggle | "Look for" note matches | Met |
| "Split … by line without leakage" | `build_sample_dataset` (line-level shuffle) + `check_split_disjoint` (pixel digest) | P2: 0 shared images, but 131/140 test lines adjacent to a training line; 5 test and 2 validation transcripts occur verbatim in train | cell 24 advises page/writer splits for *your* data | **Partly met** (GOT-M3) |
| Inference contract on the printed page | cell 15 | Kaggle: checks all `True`, CER 0.0, unsupported mode recorded as a finding | well explained, `sample-sanity` | Met (documented) |
| Frozen CER/WER beside two baselines | cell 17 | Kaggle: CER 1.0 / 0.944 / 1.345; WER 1.0 / 0.999 / 1.780; four frozen hypotheses printed | well explained | Met (documented) |
| Bounded fine-tune, explicit hyperparameters, validation selection | cell 19 → `adapt` | Kaggle: 51,401,728 trainable, best epoch 4 (val CER 0.634); val CER 0.788 at epoch 6 | "loss drops … to 0.05" paired with the falling CER (GOT-m3); seed (0) not printed | Met (documented) |
| "Does adaptation move the held-out CER **and WER** past two baselines?" | cell 21 | CER 0.759 < 0.944 < 1.0 yes; WER 1.017 > 1.0 and > 0.999 **no**; flag `adapted_beats_both_baselines: True` computed on CER | WER presented as a gain; the answer for WER is never given | **Partly met** (GOT-M4) |
| Panels, page re-read, adapter export and fresh reload with parity | cell 23 | Kaggle: 6 panels, page CER 0.0, 49 tensors, 205,613,096 bytes, parity 8/8 | explained | Met (documented) |
| BYOD zip through the same cells, "at least eight images" | cell 13 BYOD branch; rerun "from that cell" | P3: 8–49 images refused; 50 accepted. Source: rerun reuses the adapted model as "frozen" | contract states 8 | **Not met as stated** (GOT-m1, GOT-M5) |

| Learning objective (opening cell) | Learner activity | Evidence exercised |
|---|---|---|
| Install, read the carried package, stage and verify the snapshot | run cells | printed identity and verified-file count |
| Fetch, validate and split without leakage | run cell 13 | digests and refusal probes printed; no prompt to check the split unit |
| Read the output contract on the printed page | read output | prose explains no score, `truncated`, `sample-sanity`; no prediction or checkpoint |
| Measure frozen CER/WER beside baselines; read a rate above 1.0 | read four printed hypotheses | the best activity in the notebook, but it is a stated expectation, not a learner prediction |
| Fine-tune, evaluate, look at the adapted transcripts | run, read panels | no question asks why validation CER rose after epoch 4, or why WER stayed above 1.0 |
| Export and reload with parity | run | parity printed and asserted |

The objectives are operations the code performs rather than learner actions with a check (GDL5); there is no learner-controlled Predict → Change one thing → Run → Observe → Explain activity with rerun scope (GDL10). The "Optional experiments" paragraph is the only transfer prompt.

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 25 cells | Each section opens with an accurate explanation and a "Look for"/"Watch"/"Expect" note, and the closing interpretation is careful about scope. Missing: audience statement, how-to-use, roadmap, Input → Model → Output contract, glossary (image tokens, greedy decoding, stop string, Levenshtein, micro/macro, medoid, frozen-prefix cache, causal LM loss, safetensors), predictions, checkpoints with sample answers, troubleshooting, conclusion template; the three carried cells (1,338 lines) are unlabelled and uncollapsed (GOT-M2). Section 8 calls the 140-line test split "Eighty lines" (GOT-m3). A learner reading Section 8 and the interpretation is told the adaptation "teaches it to read the hand" with WER 1.017 and is never told that this WER is worse than predicting nothing (GOT-M4). |
| **Clean default** | Documented (Kaggle T4, reviewed blob) + direct (CPU, data stages, install skipped) | Kaggle: pass 1 failed at the install guard, pass 2 11/11 after a restart (GOT-M1); outputs and numbers match the prose apart from GOT-m3. Direct: cells 3/5/7/9/13 from an empty working directory, Belfort fetch over range requests 29.6 s, split and digests identical to Kaggle, `SAMPLE_DIGEST` match, refusals identical; generator `--check` and validator pass. Sections 3, 5–9 not executed. No Colab run. |
| **Active learning** | Source inspection | No documented exercise states a prediction or rerun scope (GOT-m4). Rerunning cell 19 after changing `EPOCHS`/`LEARNING_RATE` continues from the adapted weights and labels epoch 0 "frozen model" (GOT-M5, from `adapt` source, P6). The `format`-mode experiment needs an edit inside the `recognise_page` helper (GOT-m4). An experiment whose adapted CER does not beat the frozen one stops at a bare `AssertionError` in cell 21 (GOT-m2). Not executed. |
| **Reuse and recovery** | Direct (P3, cell 13 with shimmed upload) + source; real upload dialog and BYOD model stages not verified | 50 synthetic lines → split 32/8/10, all splits validated. Refused: 8, 20, 38, 49 lines with "2/4/6/7 records; 8..5000 are required" (the split is not named); non-image member (clear); missing `transcripts.csv` (clear); 513-character transcript (clear); not a zip (clear); same basename in two folders → "duplicate id 'x'" with no mention that the archive was flattened; cancelled upload → bare `StopIteration`. 70 records with 10 pixel duplicates → 60 split, nothing reported (GOT-m1). The documented rerun "from that cell" then scores the Belfort-adapted model as "frozen" and fine-tunes on top of it (GOT-M5). |

## 5. Findings

### Major

#### GOT-M1 — `Run all` needs a manual restart after the install cell, and the blob is marked `Release-grade` on that run

- **Cell/section:** cell 3, Section 1 prose (cell 2: "the cell stops with a restart instruction"), opening cell 0 ("27 with the pinned install and its restart"). Generator: `tools/build_notebook.py` lines 48–70 (install-cell body, guard at line 69); `tools/notebook_template.py` line 66. Records: `docs/release-verification.md` procedure step 4 ("an interpreter restart after the install is expected"), Manual evidence, Recorded executions and Current status; `STATUS.md`; `README.md` line 71; `tutorials/README.md` registry row.
- **Observed issue:** the cell `pip install`s nine pins into the running kernel, then raises `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.` when a loaded distribution changed. The opening cell promises that **Run all** in a fresh runtime installs, stages, trains, evaluates, exports and reloads.
- **Consequence:** on a stock Kaggle (and, by the same mechanism, Colab) image the learner's **Run all** stops in the first code cell and must be restarted and re-run; RUN1, RUN10 and ENV6 forbid this, and a restart-dependent run is not REL2 evidence. The repository discloses the restart but records `PASSED` and promotes the blob to `Release-grade`; the procedure declares the restart expected.
- **Evidence:** documented — Kaggle T4 run of blob `951ad49e`: pass 1 `RuntimeError` naming `cuda-bindings` 12.9.4 → 13.4.2 and `numpy` 2.0.2 → 2.5.3, `restarted_after_install_cell: true`, pass 2 11/11 (P7). Source — P0: `pip_install_in_kernel: true`, `restart_instruction_in_install_cell: true`, `uses_uv: false`, `cell0_restart_in_timing: true`.
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in `tools/build_notebook.py` (and the validator's install-cell expectations in `tools/validate_release_assets.py`), regenerate, re-qualify with a one-pass hosted Run all, and return the registry to `Candidate` until then; correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** a fresh Kaggle or Colab GPU runtime completes every code cell in one pass with no restart and no error, recorded in `docs/release-verification.md` with the notebook blob id and `restarted: false`; `grep -n "Restart the runtime" tutorials/got_ocr2_colab.ipynb` and `grep -n "its restart" tutorials/got_ocr2_colab.ipynb` return nothing; no document marks a blob `Release-grade` on a run with `restarted_after_install_cell: true`.
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11; §31 release-readiness rule.

#### GOT-M2 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** opening cells 0–1, every section boundary, cells 3, 5, 7, 9 and 11, end of notebook. Generator: `tools/notebook_template.py` section texts and `tools/build_notebook.py` section assembly.
- **Observed issue:** no intended-learner statement, no **How to use this notebook**, no roadmap, no Input → Model → Output task contract, no glossary, no prediction before the frozen scoring, the fine-tune or the held-out comparison, no interpretation checkpoint with a sample answer, no troubleshooting section, no conclusion template. Cells 5 (852 lines), 9 (387), 7 (99), 3 and 11 carry no **Infrastructure** label and no `cellView: form`. The "Expect"/"Look for"/"Watch" notes state the build record's numbers rather than asking the learner to predict.
- **Consequence:** a self-paced learner gets precise explanations but no help predicting, checking their reading of the four-way comparison, recovering from a hosted-runtime failure (out-of-memory, a Hub fetch error, a digest mismatch), or writing a bounded conclusion; the carried modules dominate the scroll before any model runs.
- **Evidence:** source inspection; P0 `guided_markers` (how_to_use, roadmap, glossary, check_your_reasoning, what_to_notice, conclusion_template, infrastructure_label, intended_learner, predict_prompt, troubleshooting all false; look_for true), `cellView_form_cells: []`.
- **Recommended correction:** add the GDL layer in the template per NOTEBOOK_SPEC 2.2 (reference notebook §25.13): audience and how-to-use, roadmap, task contract, glossary, a learner prediction before Sections 6, 7 and 8, "What to notice" after each principal stage, collapsible "Check your reasoning" answers (for example: why the frozen model scores worse than silence; why validation CER rises after epoch 4 while the loss keeps falling; why WER stays above 1.0 while CER falls), one Predict → Change one thing → Run → Observe → Explain activity with its rerun scope (see GOT-M5, GOT-m4), a troubleshooting section, and a conclusion scaffold; title cells 3, 5, 7, 9 and 11 `# @title Infrastructure: …` with `cellView: form`.
- **Acceptance check:** each of GDL1–GDL14 maps to a named cell in a checklist added to `tutorials/README.md`, and cells 3, 5, 7, 9 and 11 carry `cellView: form` with an Infrastructure title.
- **Spec:** GDL1–GDL14, UX8.

#### GOT-M3 — The "without leakage" split shuffles individual lines, so test lines sit beside their own page's training lines

- **Cell/section:** cell 13 (`build_sample_dataset(corpus, seed=SPLIT_SEED)`), cell 12 prose ("seeded line-level split"), cell 0 objective ("split it by line without leakage"), cell 20 ("no image appears in two splits"), cell 24 Leakage advice and "validate the demonstrated dataset contract without leakage". Generator: `src/got_ocr2_pipeline/samples.py` `build_sample_dataset` (line 193) / `split_dataset` (line 313); `tools/notebook_template.py` lines 103, 134, 471 and 474.
- **Observed issue:** the 800 lines are the first eight row groups of the test shard in corpus order, so neighbouring lines come from the same page and hand. `build_sample_dataset` shuffles single lines; `check_split_disjoint` only rejects pixel-identical images. Cell 24 then tells the learner that "lines cut from the same page share a hand" and to split by page, writer or volume — advice the notebook's own split does not follow — and does not say that the reported held-out CER is a same-page estimate.
- **Consequence:** the learner is taught that a pixel-disjoint line split is "without leakage" and is given a held-out gain whose independence assumption is not stated (SPL3) and whose grouping boundary is not preserved (SPL5). Unlike a model that barely reads, this adapted model moves CER by 0.586, so same-hand training lines can plausibly flatter the test number; the size of that effect is not measured here.
- **Evidence:** direct (P2): 0 shared images; **131 of 140** test lines have the corpus line immediately before or after them in the training split; test lines come from all eight row groups (11–23 each); 5 test transcripts and 2 validation transcripts occur verbatim in training (for example "Le Conseil Municipal", "Après en avoir délibéré, à l'unanimité :"). Source: cell 0 and cell 24 text (P0 `cell0_without_leakage_claim`, `cell24_split_page_advice`).
- **Recommended correction:** split by contiguous blocks (for example whole row groups, or fixed runs of consecutive lines as a page proxy) so that no test line has a training neighbour, assert that property in the notebook, and state the independence assumption; or, if the line-level split is kept for size reasons, drop "without leakage" and say plainly that test lines share pages and hands with training lines, so the test CER is an optimistic same-document estimate. Re-record the build-record numbers after a change of split.
- **Acceptance check:** either the notebook asserts that no test line id is within ±1 (or the chosen block size) of a training line id and the prose describes the grouping, or the words "without leakage" no longer describe the line split and cells 20 and 24 state the same-page limitation explicitly.
- **Spec:** SPL3, SPL5 (MUST), SPL10.

#### GOT-M4 — The notebook asks whether adaptation moves CER **and WER** past both baselines; on WER it does not, and the notebook never says so

- **Cell/section:** cell 0 ("does a bounded fine-tuning … move the held-out **CER** and **WER** on a line-disjoint test split past two **non-adapted baselines** and the frozen model?"), cell 20 ("then **WER** (1.780 → 1.017: whole words, not just characters)"), cell 21 (`'adapted_beats_both_baselines': adapted_test['cer'] < min(baseline_empty['cer'], baseline_constant['cer'])`), cell 24 ("teaches it to read the hand (0.759 CER and 1.017 WER …)"). Generator: `tools/notebook_template.py` lines 342–343, 377 and 455; the opening question in the cell-0 text.
- **Observed issue:** on the recorded run the adapted model's test WER is 1.017, above the empty baseline (1.000) and the constant-transcript baseline (0.999). The notebook reports the frozen-to-adapted WER drop as a gain, prints a flag named `adapted_beats_both_baselines` that is computed from CER alone, and closes with WER 1.017 as evidence that the model learned "to read the hand". Only cell 18's "most words still wrong" hints at it.
- **Consequence:** the central question the notebook poses has a split answer — yes for CER, no for WER — and the learner is shown only the yes. A learner who carries the "baselines first" lesson to their own data is taught, by this example, to read the frozen model rather than the baselines as the reference for WER, and a generically named flag says "both baselines" were beaten when on WER neither was.
- **Evidence:** documented (P7, Kaggle evaluation report of the reviewed blob): `comparison.wer = {empty: 1.0, constant: 0.999, frozen: 1.78, adapted: 1.017}`; `adapted_wer_beats_empty: false`, `adapted_wer_beats_constant: false`; printed `adapted_beats_both_baselines: True`. Source: P0 `cell0_question_cer_and_wer_past_baselines: true`, `cell21_beats_both_flag_uses_only_cer: true`.
- **Recommended correction:** answer the stated question per metric in Section 8 and the interpretation (for example: "CER: past both baselines. WER: still worse than predicting nothing — most words still contain an error, and an inserted or split word costs a whole word"); rename the flag `adapted_cer_beats_both_baselines` and add the WER counterpart to the printout and the exported comparison; add a checkpoint asking why CER and WER disagree.
- **Acceptance check:** Section 8 or the interpretation states, per metric, whether the adapted model beats each baseline, consistent with the run's numbers; no printed or exported key named `…beats_both_baselines` is computed from a single metric unless its name says which; on the recorded numbers the notebook's text says WER did not beat the baselines.
- **Spec:** EVAL3, EVAL15 (MUST); GDL14, EVAL10.

#### GOT-M5 — The prescribed reruns (BYOD "re-run from that cell", the optional experiments) measure against the already adapted model

- **Cell/section:** cell 0 BYOD instruction ("set `USE_BYOD = True` in Section 4 and re-run from that cell"); cell 15 (`recognise_page(pipe, 'frozen')`), cell 17 (`frozen_test = pipe.evaluate(…)`), cell 19 (`pipe.adapt(…)`), cell 24 Optional experiments (`LEARNING_RATE`, `EPOCHS`, `LINE_MAX_NEW_TOKENS`); model constructed only in cell 11. Generator: `tools/notebook_template.py` lines 72, 262, 290, 478–481; `src/got_ocr2_pipeline/pipeline.py` `adapt` (`frozen_state` cloned from the current `state_dict`, epoch 0 `"note": "frozen model"`).
- **Observed issue:** after a full run `pipe` holds the Belfort-adapted decoder tail. Re-running from Section 4 (BYOD) or Section 6/7 (experiments) does not reload the base model: Section 5 writes the adapted model's page transcript to `got_ocr2_page_frozen.txt`, Section 6 scores the adapted model as the "frozen model", and `adapt` snapshots the adapted weights as its restore point, labels them epoch 0 "frozen model", and trains on top of them. `adapt` does not refuse an already adapted pipeline.
- **Consequence:** the BYOD branch — promised to pass "through the same … baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells" — reports a frozen-vs-adapted comparison whose "frozen" side is the Belfort-tuned model, and exports an adapter trained on Belfort and then on the user's data while its manifest records only the user's training. The learning-rate and epoch experiments compound two fine-tunes and misread the starting point. Each of these is the documented route, not misuse.
- **Evidence:** source inspection (P6: `restore_point_is_current_weights: true`, `epoch0_labelled_frozen_model: true`, `refuses_when_already_adapted: false`, `model_loaded_only_in_cell11: [11]`, cells 15/17 use the shared `pipe`). Not executed (no model run in this review).
- **Recommended correction:** make every documented rerun start from the pinned base: instruct "re-run from Section 3" (which re-verifies and reloads), or reload the base in the BYOD branch and at the top of Section 6, or have `adapt` refuse with an actionable message when `self.adapter` is set (and offer a `reset()` that restores the base tail). Name the rerun scope in each experiment.
- **Acceptance check:** following the BYOD and experiment instructions exactly after a completed default run yields an epoch-0 validation CER equal to the frozen model's on that split (for the default split, the recorded 1.554 within run-to-run tolerance), a `got_ocr2_page_frozen.txt` produced by the base model, and an adapter whose manifest history starts from the base; or `adapt` refuses on an adapted pipeline with a message naming the cell to re-run.
- **Spec:** DAT13, DAT14 (MUST); GDL10.

### Minor

#### GOT-m1 — BYOD: the stated minimum is 8 records but the branch needs 50, and several refusals are not actionable

- **Cell/section:** cell 0 ("at least eight images"), cell 1 ("a dataset needs 8..5,000 records"), cell 13 BYOD branch (`split_dataset` then `validate_dataset` per split with the default `min_records=8`). Generator: `tools/notebook_template.py` lines 73, 120 and 147–156; `samples.py` `split_dataset`, `load_byod_dataset` (line 336).
- **Observed issue:** `split_dataset`'s default fractions give splits below 8 records for any dataset under 50 distinct images, and each split is then validated with the 8-record minimum; the error names a count but not the split. Pixel-duplicate images are dropped without a message. Two archive members with the same basename in different folders surface as "duplicate id 'x'" without mentioning the flattening. A cancelled upload raises a bare `StopIteration`. There is no location field, so an executor, a Kaggle user, or a user with a mounted file must use the Colab upload dialog.
- **Consequence:** a user who follows the stated contract with 8–49 lines is refused with "2 records; 8..5000 are required" for a dataset of 8 and cannot tell why; dropped duplicates change the split silently; on Kaggle the branch cannot be used at all.
- **Evidence:** direct (P3, shimmed upload through cell 13): 8 → "2 records; 8..5000 are required", 20 → 4, 38 → 6, 49 → 7, 50 → accepted (32/8/10); 70 records with 10 pixel duplicates → 60 split (39/9/12), nothing reported; same basename in `a/` and `b/` → "duplicate id 'x'"; empty upload → `StopIteration`. Clear refusals: non-image member, missing `transcripts.csv`, 513-character transcript, non-zip. Source: P0 `byod_location_field: false`, `byod_uses_files_upload: true`.
- **Recommended correction:** state the real minimum (or size the validation/test minimums to the split, with a warning below a stated size), name the split in the error, report dropped duplicates (count and ids), reject duplicate basenames with a message about flattening, check for an empty upload, and add a `BYOD_PATH = ''  # @param` location field that reads a zip or directory without importing `google.colab`.
- **Acceptance check:** a BYOD zip of the stated minimum size passes cell 13; a smaller one is refused before splitting with a message naming the minimum; a zip with duplicates prints the number dropped; a cancelled upload raises a `ValueError` naming the next action; setting `BYOD_PATH` reads the file without importing `google.colab`.
- **Spec:** DAT12, DAT19, VAL7 (MUST); EXE2, UX10.

#### GOT-m2 — The held-out assertions turn a legitimate negative result into a crash before export

- **Cell/section:** cell 21 (`assert adapted_test['cer'] < frozen_test['cer']`, `assert adapted_test['cer'] < baseline_empty['cer']`). Generator: `tools/notebook_template.py` lines 375–376; `tools/validate_release_assets.py` lines 69–70 require them.
- **Observed issue:** the assertions encode the expected result on the Belfort sample, but the same cell runs for BYOD and for the experiments. The notebook's own build record shows the recipe landing above the empty baseline on smaller data (280 training lines: 1.302 at lr 5e-5, 1.040 at lr 1e-4), and cell 24 suggests the lr 1e-4 experiment. On such a run the cell raises a bare `AssertionError` after writing the evaluation report but before the panels, the page re-read, the export and the reload.
- **Consequence:** the BYOD branch promised to reach "artifact export and reload-parity cells" stops on an honest negative result with no explanation, and an experiment the notebook recommends can end the same way; a negative result is treated as a failure instead of being reported.
- **Evidence:** source inspection (P0 `cell21_hard_asserts`); build-record figures from cell 18 and `docs/release-verification.md`. Not executed.
- **Recommended correction:** keep the assertions for the default sample only (or turn them into a reported verdict such as `adapted_beats_frozen: false` with an explanatory message), and adjust the validator's expectation.
- **Acceptance check:** with `USE_BYOD = True` and a dataset on which the adapted CER is not lower than the frozen CER, cells 21–23 complete, the report and result JSON record the negative comparison, and the adapter is still exported and reloaded.
- **Spec:** DAT13 (MUST); RUN9, UX10.

#### GOT-m3 — Expected-output prose: a wrong split size, a loss/CER pairing that hides overfitting, and no statement of run-to-run variability

- **Cell/section:** cell 20 ("Eighty lines from one seeded split give **no dispersion estimate**"), cell 18 ("Watch the validation CER fall from 1.554 to 0.634 (epoch 4 in the build record) while the loss drops from about 3.99 to 0.05"), cells 16–24 (build-record numbers to three decimals). Generator: `tools/notebook_template.py` lines 312–313 and 346.
- **Observed issue:** the test split is 140 lines, not eighty (eighty was the 280-line probe's split). The loss reaches 0.05 at epoch 6, where validation CER has risen back to 0.788; the sentence pairs it with the epoch-4 minimum, so the learner does not see the overfitting that validation selection exists to catch. No cell says that the GPU fine-tune and its scores vary between runs and devices while the prose quotes them to three decimals.
- **Consequence:** a learner checking the text against the output finds a wrong count, misses the main lesson of the validation curve, and cannot tell whether a result a few hundredths away from the quoted numbers is expected.
- **Evidence:** documented (P7 history: epoch 4 loss 0.500 / val CER 0.634; epoch 6 loss 0.052 / val CER 0.788; test n 140); source (P0 `cell20_eighty_lines: true`, `cell18_loss_claim`, `run_to_run_variability: false`).
- **Recommended correction:** say 140 lines; describe the curve as falling to its minimum at epoch 4 and rising as the loss keeps falling, and point out that the kept epoch is 4; add one sentence that the fine-tune's numbers vary by GPU and run and give the expected spread if one is known.
- **Acceptance check:** `grep -n "Eighty lines" tutorials/got_ocr2_colab.ipynb` returns nothing; Section 7 names the rise after the best epoch; Sections 7 or 8 state run-to-run variability.
- **Spec:** ENV8 (MUST); GDL8.

#### GOT-m4 — The optional experiments name no rerun scope or prediction, and the `format`-mode experiment needs a code edit

- **Cell/section:** cell 24 "Optional experiments"; cell 15 (`recognise_page` hard-codes `mode='plain'`). Generator: `tools/notebook_template.py` lines 478–481 and the cell-15 body.
- **Observed issue:** none of the five experiments says which cells to re-run or asks the learner to predict the outcome; "re-run Section 5 in `format` mode" requires editing `mode='plain'` inside the `recognise_page` helper because there is no form field for the mode.
- **Consequence:** the experiments are the notebook's only active-learning route, but they are unscoped (see GOT-M5 for what the obvious rerun does) and one requires changing infrastructure code.
- **Evidence:** source inspection (P0 `experiments_name_rerun_scope: false`, `cell24_format_mode_needs_code_edit: true`).
- **Recommended correction:** frame one experiment as Predict → Change one thing → Run → Observe → Explain with its rerun scope; add `PAGE_MODE = 'plain'  # @param ["plain", "format"]` to Section 5.
- **Acceptance check:** each experiment names the cell range to re-run and a question to answer; the `format` experiment is a form-field change only.
- **Spec:** GDL10, UX5, EXE1.

### Suggestions

- **GOT-S1 — Declare the current spec.** The notebook, `tutorials/README.md` and the validator declare NOTEBOOK_SPEC 2.0; regenerate against 2.2 when the template is revised.
- **GOT-S2 — Document `DIMER_NOTEBOOK_CI_PREINSTALLED`.** Cell 3 reads it to skip the install, but no markdown mentions it (EXE5).
- **GOT-S3 — Record a Colab run and correct the runtime statement.** The Colab badge is the entry point, but the only hosted record of this blob is Kaggle; cell 1 says "Python 3.12" for Colab while this repository's own Colab record for the workshop notebook (2026-09-26) shows Python 3.13.15.
- **GOT-S4 — Show the curve and the distribution.** Plot or tabulate the per-epoch validation CER and loss, print the fine-tune seed beside `EPOCHS`/`LEARNING_RATE`/`BATCH_SIZE`, and show the per-line CER distribution beside the corpus rates.

## 6. Readiness

**Needs revision.** Open Majors GOT-M1 to GOT-M5; unresolved MUSTs RUN1, RUN10, ENV6, REL2, REL11 (GOT-M1), SPL3, SPL5 (GOT-M3), EVAL3, EVAL15 (GOT-M4), DAT13, DAT14 (GOT-M5), DAT12, DAT19, VAL7 (GOT-m1), DAT13 (GOT-m2), ENV8 (GOT-m3). The repository's `Release-grade` status rests on a restart-dependent run and should return to `Candidate`. Remaining gates after the fixes: a one-pass hosted Run all of the regenerated blob (Colab and Kaggle), a re-recorded held-out result on a grouped (or explicitly qualified) split, and hosted BYOD positive and negative runs that start from the base model and reach export.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, install skipped, data stages only):** the Belfort fetch, validation, refusal probes and split (identical to Kaggle), the split-adjacency and duplicate-transcript counts (P2), the absence of carried-module name collisions (P1), the BYOD branch of cell 13 with eleven archives (P3), metric sanity (P4), and generator/validator parity (P5).
- **Verified from documented execution:** the install-cell restart and every model-stage number quoted here, including the WER comparison and the validation curve (Kaggle T4, reviewed blob, P7).
- **Inferred from source:** the rerun behaviour of BYOD and the experiments (GOT-M5), the negative-result crash (GOT-m2), Colab behaviour and the real upload dialog.
- **Most likely to be wrong:** GOT-M4's severity. The prose does say "most words still wrong", and a reader could take the WER sentence as a frozen-vs-adapted comparison only; the finding rests on the notebook's own question naming WER and both baselines, and on a flag named for both baselines that tests CER. It may fairly be judged Minor.

Probe ZIP: `got_ocr2_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
