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

## Results

Pending: no grid run has been recorded yet.
