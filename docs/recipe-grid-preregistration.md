# Pre-registered recipe grid and fresh test (review GOT2-M1)

Status: **pre-registered 2026-10-17, before any grid run.** This file was committed before the grid was run on
any hardware. A later commit may add results below the "Results" heading. It must not change the grid, the rules or
the fresh-test pin above it. Any change to them is a new pre-registration with its own date.

Source: the maintainer's decision of 2026-10-17 on review finding GOT2-M1 (row 30 of the tutorial triage). The
review found that the notebook and the model card justified the 600-line training set, and the recipe kept with it,
by test-split CER. This grid replaces that justification with a validation-only choice and a single test look.

## What is fixed in advance (not tuned)

| Item | Value |
|---|---|
| Data | The 800-line Belfort sample (row groups 0–7 of `Teklia/Belfort-line` `default/test/0000.parquet` at `c4a74bbd39f2df314752e7e6026649a39d365cbb`). Split seed 42, line-image split 600 / 60 / 140 (`SAMPLE_DIGEST` `b7e1dd68…`). |
| Training lines | All 600 |
| Epochs | 6 |
| Batch size | 8 |
| Adaptation seed | 0 |
| Trainable part | The last 4 decoder layers and the final norm (`TRAINABLE_LAYERS = 4`) |
| Gradient clipping | 1.0; AdamW without weight decay; no scheduler; no augmentation |
| Token budget | `LINE_MAX_NEW_TOKENS = 128`, greedy decoding |
| Epoch rule | Within one config, the epoch with the lowest validation CER is kept |
| Validation | The 60 validation lines, scored at every epoch; epoch 0 is the frozen model |

## The grid

| Config | Learning rate | Role |
|---|---|---|
| G1 | 5e-5 | The current default |
| G2 | 1e-4 | The only alternative an earlier development run tried, and the Section 10 activity's rate |
| G3 | 2.5e-5 | A smaller step |

## Selection rules (applied to validation CER only)

1. Each config keeps its best epoch by validation CER.
2. **G2 or G3 replaces G1 only if its best validation CER is lower than G1's by more than 0.05.** If both qualify,
   the one with the lower best validation CER replaces G1. Otherwise G1 is kept.
3. **If no config's best validation CER is lower than the frozen model's (epoch 0) by more than 0.05, the frozen
   model is the result.** In the notebook, `adapt(min_gain=0.05)` applies this rule inside each config: it keeps
   epoch 0 when the best epoch does not clear the margin.
4. The test data play no part in rules 1–3.

The 0.05 margin is an estimate of what 60 validation lines can resolve: the epoch-to-epoch jitter near the
minimum in the 2026-10-04 run was 0.023–0.07. The per-line spread was never recorded, so the margin may be too
small. The review names this as its most likely error.

## Test looks

- **Fresh test.** Row groups 8–9 of the same shard (200 lines, ids `belfort-test-800` to `belfort-test-999`).
  No run of this repository fetched, scored or looked at them before 2026-10-17. Pins in `samples.py`:
  row group 8 `f4036cc009855fb2e710fe55b5222b5a016b039a3b8aaf31d32126768ffc3b71` / 5,000,587 bytes, row group 9
  `d26c4fa972ddcef0cb2152432ee6821f69cd9498cb140edff2baac4094013869` / 4,683,815 bytes, `FRESH_TEST_DIGEST`
  `59e018c45e80fbb38f425e120fd31de0d84ca01c1df87e3041eec59f35c3d245`. `build_fresh_test` refuses the set unless
  its ids, digest and row groups match. It also refuses any shared id, source row group or decoded image with the
  train, validation or development-test split. Measured 2026-10-17: 0 shared images, 0 shared ids, 7,345 reference
  characters and 1,225 words. 8 fresh transcripts also occur verbatim in training: these are formulaic lines, and
  they are counted, not removed. The parquet has no page or writer field, so the fresh test is not page- or
  writer-disjoint. It is still a same-collection estimate, but it removes every earlier look.
- **The chosen configuration is scored on the fresh test once**, beside the two baselines and the frozen model.
- **Exposed development test.** The 140-line test split of the sample was looked at during the 2026-09
  development runs, and those runs chose the data design. It is reported once for the chosen configuration, under
  that label, and is not the result.

## Superseded development runs

Three runs scored on a test split chose the data design (MODEL_CARD build record, 2026-09-20, package API at
`af9bcfe`/`91d1c00`). Two used 280 training lines and 80 test lines, and one used 600 / 140. They are kept in the
model card's history as **superseded development runs scored on a test split**. They are not a selection, and
the notebook no longer quotes them.

## How the grid is run

- `RUN_RECIPE_GRID` is a Section 7 form field, **off by default**. The learner's default path trains G1 only,
  which takes about 30 minutes of model-stage time on a T4.
- The grid runs **once**, on a hosted T4, from the companion branch
  `fix/got-ocr2-tutorial-review-2026-10-17-gridrun`. That branch is the PR head plus one generated commit, which
  sets only `RUN_RECIPE_GRID = True`. The notebook prints the validation table and writes it to
  `outputs/got_ocr2_recipe_grid.json`.
- The table, the run's commit and blob, and the chosen config are recorded below and in
  `docs/release-verification.md`. If the rules pick a config other than G1, the default `LEARNING_RATE` changes to
  that config's rate in a later commit, and the notebook names this run as its source.

## Amendment 1 — 2026-10-17, before any hosted run

Status: committed before any grid run and before any hosted run of the notebook. It changes how the fresh test is
used and who reports it. It changes no config, no fixed setting, no margin and no pin above. The original
pre-registration is commit `7342639` on the branch `fix/got-ocr2-tutorial-review-2026-10-17`.

Source: the maintainer's fleet-wide score-once rule of 2026-10-17 (`SCORE_ONCE_RULE.md`, phase-3 fresh-test rows) and
the peer review of PR #12 (M1–M3, m1, m3, m6, S3).

1. **The grid run is selection-only.** With `RUN_RECIPE_GRID = True` the notebook records the validation table and
   the chosen config. It never scores the fresh test. It prints that the fresh test is scored on the default path with
   the chosen config. Its Sections 8–9 compare on the exposed development test only, and they write
   `*_exploratory_<n>.json` files. The grid also records the fixed settings it ran under. A grid whose fixed settings
   differ from the table above is labelled exploratory (`preregistered_fixed_settings: false`).
2. **Run order.** The grid run comes first. If its rules choose G2 or G3, the default `LEARNING_RATE` (and
   `RECIPE_DEFAULT`) changes to the winner in a commit made before the evaluation run.
3. **The evaluation.** The fresh test is scored only when every pre-registered setting holds (`PREREGISTERED_SETTINGS`
   in `samples.py`):
   - the sample, split seed 42 and 600 / 60 / 200 lines with the pinned fresh-test digest;
   - 6 epochs, batch 8 and a 128-token line budget;
   - the selected config's learning rate;
   - adaptation seed 0, 4 trainable layers, the 0.05 margin and gradient clipping at 1.0.

   The notebook computes one `preregistered` boolean over all of them. A run with any other setting is exploratory:
   a changed form field, BYOD, an optional experiment, the Section 10 activity, or `BATCH_SIZE = 4` against an
   out-of-memory error. An exploratory run compares on validation and on the exposed development test, never on the
   fresh test, and names the settings that differ.
4. **The result and reproductions.** The first recorded hosted run of the default path under the pre-registered
   settings is **the** result. A later hosted run with identical settings is a **reproduction of the pre-registered
   evaluation**, provided nothing was changed in response to the test number. It is logged and reported as one. The
   notebook labels every pass `pre-registered evaluation`, `reproduction` or `not scored (exploratory)`. It keeps a run
   history in `outputs/got_ocr2_run_history.json`. It never overwrites the pre-registered result file with a
   reproduction (`*_reproduction_<n>.json`) or an exploratory pass (`*_exploratory_<n>.json`).
5. **Where the frozen model is scored.** Section 6 now shows the baselines and the frozen model on the exposed
   development test. Under the pre-registered settings, Section 7 scores the frozen model and the two baselines on the
   fresh test, before adapting. The primary metric is unchanged: corpus CER and WER on all 200 fresh-test lines. The
   lines whose transcript repeats verbatim in training stay in it. They are reported as a separate row beside the
   192 others, with a case- and punctuation-folded count beside the exact one.
6. **Tie-break (stated explicitly).**
   - If both G2 and G3 clear the margin over G1, the one with the lower best validation CER wins. This sentence was
     already in the original commit `7342639`.
   - An exact tie between G2 and G3 goes to G2, the first in grid order.
7. **Text normalisation.** `normalise_text` adds Unicode NFC before collapsing whitespace. All 1,000 Belfort
   transcripts (row groups 0–9) were already NFC on 2026-10-17, so `SAMPLE_DIGEST` and `FRESH_TEST_DIGEST` do not
   change.
8. **Informative statistics (they decide nothing).** These figures sit beside the rules above and do not change them:
   - The grid records per-line validation counts and a paired line-bootstrap interval for G2−G1 and G3−G1 at each
     config's best epoch.
   - The evaluation prints 95 % line-bootstrap intervals for the adapted CER and WER, and for the difference between
     the adapted and frozen models.

## Results

Recorded 2026-10-17 (details and evidence files in `docs/release-verification.md`). The grid and the rules above were applied unchanged.

**Grid run (selection only)** — gridrun commit `d165723`, blob `9ec5eb71`, Colab CLI 0.7.4 on a fresh Tesla T4; fixed settings equal to the pre-registration (`fixed_settings_differ: []`); the fresh test was not scored. Validation CER (60 lines):

| Config | LR | ep 0 | ep 1 | ep 2 | ep 3 | ep 4 | ep 5 | ep 6 | best epoch | best val CER |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | 5e-5 | 1.554 | 0.875 | 0.886 | 0.707 | **0.634** | 0.657 | 0.788 | 4 | 0.634 |
| G2 | 1e-4 | 1.554 | 1.011 | 0.828 | 0.750 | 0.687 | 0.691 | 0.643 | 6 | 0.643 |
| G3 | 2.5e-5 | 1.554 | 1.200 | 1.097 | 0.671 | 0.653 | 0.764 | 0.727 | 4 | 0.653 |

G1 is kept: neither G2 (0.643) nor G3 (0.653) beats G1 (0.634) by more than 0.05. Paired bootstrap (informative): G2 − G1 +0.009 (95 % −0.039 to 0.058), G3 − G1 +0.020 (−0.036 to 0.072). The frozen-model rule was not triggered. The default stays G1 (`LEARNING_RATE = 5e-5`).

**Pre-registered evaluation** — PR head `f564d9e`, blob `59bb22a6`, default path, Colab CLI 0.7.4 on a fresh Tesla T4, `test_look: pre-registered evaluation`, no setting differing. Fresh test (200 lines): CER frozen 1.157 → adapted 0.902 (95 % 0.733–1.121; difference −0.255, −0.509 to −0.002), WER 1.604 → 1.308 (1.043–1.646; difference −0.296, −0.761 to 0.135). The 192 lines whose transcript does not occur in training: adapted CER 0.910 / WER 1.317; the 8 that do: 0.444 / 0.762. Exposed development test (history): adapted 0.759 / 1.017. Reproductions: none yet.

