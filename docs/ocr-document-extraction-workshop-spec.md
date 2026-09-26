# DIMER OCR & Structured Document Extraction Workshop

## GOT-OCR 2.0 vs SmolDocling

**Proposed filename:** `DIMER_OCR_Document_Extraction_Workshop.ipynb`  
**DIMER Notebook Specification:** `2.1`  
**Profile:** `TASK-INFERENCE`  
**Pedagogical mode:** `WORKSHOP`  
**Comparison scope:** `MULTI-MODEL / MULTI-CAPABILITY`  
**Standalone:** `true`  
**Recommended runtime:** Kaggle / Colab Tesla T4  
**Task family:** Optical character recognition and structured document extraction  
**Models:** GOT-OCR 2.0 + SmolDocling-256M-preview  
**Adaptation:** None  
**Default evaluation corpus:** Belfort-line held-out split  
**Supplemental evaluation:** Deterministic in-code rendered document-page suite

---

# 1. Purpose

This workshop compares two different approaches to extracting information from document images:

## GOT-OCR 2.0

```text
document image
↓
OCR vision-language model
↓
plain text
or
formatted Markdown / LaTeX-like text
```

## SmolDocling

```text
document image
↓
document conversion VLM
↓
DocTags
├── text
├── headings
├── locations
├── reading order
├── tables / OTSL
├── formulas
├── captions
└── other document elements
```

The central lesson is:

> **OCR and document extraction overlap, but they are not the same task.**

A model that produces an excellent transcript may provide little explicit layout information.

A model that produces structured document markup may preserve tables and layout but make different transcription errors.

---

# 2. Learning objectives

By the end of the notebook, learners should be able to:

1. distinguish OCR from structured document conversion;
2. describe the output contract of GOT-OCR and SmolDocling;
3. recover plain text from both models under one normalized comparison protocol;
4. calculate character error rate and word error rate;
5. explain why CER/WER can exceed 1.0;
6. identify hallucinated, repeated, omitted, and truncated text;
7. understand GOT-OCR's `plain` and `format` modes;
8. understand SmolDocling DocTags;
9. inspect element types and location tokens;
10. inspect OTSL table output;
11. distinguish transcription quality from structural fidelity;
12. analyze token-budget truncation;
13. compare page preprocessing strategies;
14. understand when a dedicated OCR model or document-conversion model is more appropriate; and
15. evaluate their own labelled document images using the same protocol.

---

# 3. Models

## 3.1 GOT-OCR 2.0

**DIMER profile:** GOT-OCR 2.0 Optical Character Recognition

| Field | Value |
|---|---|
| Model | `stepfun-ai/GOT-OCR-2.0-hf` |
| Revision | `d3017ef2c2c1395888c8d635c5e0508bcb0ac78d` |
| License | Apache-2.0 |
| Parameters | 560,528,640 |
| Weight format | SafeTensors |
| Weight bytes | 1,121,114,488 |
| Weight SHA-256 | `6175ac7868a4e75735f5d59f78c465081ad3427eb4f312d072a0f1d16b333ba4` |
| Vision encoder | SAM-style ViT |
| Text decoder | 24-layer Qwen2 architecture |
| Image tokens | 256 |
| Decoding | Greedy |
| Natively exposed modes | `plain`, `format` |

Snapshot total:

```text
1,139,869,676 bytes
8 files
```

---

# 4. GOT-OCR image processing

Every image is resized to:

```text
1024 × 1024
```

without preserving its original aspect ratio.

The visual tower begins with 4,096 patch tokens and projects them to:

```text
256 image tokens
```

for the language decoder.

This behavior is particularly important for:

- extremely wide handwriting lines;
- narrow receipts;
- landscape documents;
- high-resolution pages.

The notebook must explicitly explain that geometric distortion can affect recognition.

---

# 5. GOT-OCR output modes

## Plain

```text
mode = "plain"
```

Expected output:

```text
recognized textual content
```

Used for the **common quantitative text-recovery benchmark**.

---

## Format

```text
mode = "format"
```

Intended for:

- headings;
- tables;
- mathematical content;
- Markdown-like formatting;
- LaTeX-style formatting.

This is included as a **native-capability demonstration**, not scored as though its structure were equivalent to SmolDocling DocTags.

---

# 6. GOT-OCR generation limits

```text
MAX_NEW_TOKENS = 4096
DEFAULT_PAGE_MAX_NEW_TOKENS = 1024
DEFAULT_LINE_MAX_NEW_TOKENS = 128
```

Stop string:

```text
<|im_end|>
```

Every result must retain:

```text
text
new_tokens
truncated
```

A generated result is marked truncated when the model exhausts the caller-owned token budget.

---

# 7. SmolDocling

**DIMER profile:** SmolDocling-256M-preview Document → DocTags

| Field | Value |
|---|---|
| Model | `docling-project/SmolDocling-256M-preview` |
| Revision | `ce51f56c4ebe36e0b1c3a55f67b261ba22a50bf8` |
| License | CDLA-Permissive-2.0 |
| Parameters | 256,484,928 |
| Weight format | SafeTensors |
| Weight bytes | 513,028,808 |
| Weight SHA-256 | `cdcdf5d823c5684029c7d8e52177cf10f9034b3aba6577549cfb1a9ce36ad0a2` |
| Vision encoder | SigLIP-style |
| Language decoder | 30-layer SmolLM2 |
| Architecture | Idefics3-style |
| Decoding | Greedy |

Snapshot total:

```text
517,896,538 bytes
13 files
```

---

# 8. SmolDocling image processing

The page is:

1. resized so its longest edge is at most approximately `2048 px`;
2. divided into `512 × 512` tiles;
3. represented by visual tokens from those tiles;
4. supplemented by a global page view.

This differs substantially from GOT-OCR's fixed 1024×1024 squash.

That preprocessing difference is part of the deployed checkpoint system.

---

# 9. SmolDocling DocTags

SmolDocling's native output is **DocTags**, a structured markup representation.

Examples of supported elements include:

```text
<section_header_level_1>
<section_header_level_2>
<section_header_level_3>

<text>
<paragraph>

<list_item>
<ordered_list>
<unordered_list>

<otsl>
<picture>
<caption>
<formula>
<code>

<page_header>
<page_footer>
<footnote>

<chart>
<key_value_region>
```

Each page element can carry four location tokens:

```text
<loc_N><loc_N><loc_N><loc_N>
```

on a normalized coordinate grid:

```text
0..500
```

---

# 10. SmolDocling table representation

Tables can be represented with **OTSL — Optimized Table Structure Language**.

Relevant tokens include:

```text
<fcel>
<ecel>
<ched>
<rhed>
<srow>
<lcel>
<ucel>
<xcel>
<nl>
```

The notebook should demonstrate that table extraction is more than recovering the visible strings.

It also involves recovering:

```text
table structure
+
cell relationships
+
reading order
```

---

# 11. SmolDocling supported instructions

The pinned live carrier recognizes seven upstream-supported instructions:

```text
Convert this page to docling.

Convert chart to table.

Convert formula to LaTeX.

Convert code to text.

Convert table to OTSL.

Find all 'text' elements on the page, retrieve all section headers.

Detect footer elements on the page.
```

The canonical workflow uses:

```text
Convert this page to docling.
```

for common extraction.

The specialized instructions are demonstrated separately.

---

# 12. SmolDocling generation limits

```text
MAX_NEW_TOKENS = 8192
DEFAULT_PAGE_MAX_NEW_TOKENS = 2048
```

Every conversion returns:

```text
doctags
text
new_tokens
truncated
```

where:

```text
text = normalized plain text recovered from DocTags
```

The common quantitative evaluator operates on this recovered `text`.

---

# 13. Why these models can be compared

They share a meaningful external task:

```text
document image
→ textual content
```

Therefore the notebook can compare:

- CER;
- WER;
- exact match;
- hypothesis length;
- omissions;
- insertions;
- repetition;
- truncation;
- runtime.

---

# 14. Why their full outputs should NOT be forced into one schema

GOT-OCR `format` output and SmolDocling DocTags are not equivalent representations.

GOT-OCR may output:

```text
Markdown
LaTeX
formatted text
```

SmolDocling emits:

```text
typed document elements
bounding locations
reading order
OTSL tables
```

The notebook therefore has:

```text
COMMON EVALUATION
    text recovery

MODEL-NATIVE EVALUATION
    GOT format behavior
    SmolDocling DocTags / layout / OTSL
```

This separation is load-bearing.

---

# 15. Notebook profile

Because the workshop does not adapt either model:

```text
Profile = TASK-INFERENCE
```

Pedagogical mode:

```text
WORKSHOP
```

The existing individual E2E notebooks remain the canonical adaptation tutorials.

---

# 16. Models intentionally excluded

Do not add:

- Florence-2;
- SmolVLM;
- Pix2Struct;
- LayoutLM.

Reason:

This workshop is intended to contrast:

```text
dedicated OCR
vs
structured document conversion
```

Florence-2 and SmolVLM are more general-purpose multimodal systems and already have separate live DIMER capabilities.

Document QA remains a separate workshop.

---

# 17. Evaluation design

Use two complementary samples.

## A. Real handwriting benchmark

```text
Belfort-line
```

Measures text transcription.

## B. Deterministic rendered page suite

Generated entirely in notebook code.

Measures:

```text
printed-page text recovery
+
structured-document behavior
+
token-budget behavior
```

---

# 18. Belfort-line corpus

Use the exact shared corpus already carried by both live pipelines:

**Dataset:** `Teklia/Belfort-line`

Content:

```text
19th–20th century French municipal council minutes
handwritten lines
crowdsourced reference transcripts
```

License:

```text
MIT
```

Pinned parquet conversion revision:

```text
c4a74bbd39f2df314752e7e6026649a39d365cbb
```

Shard:

```text
default/test/0000.parquet
```

Declared shard bytes:

```text
210,579,166
```

Declared rows:

```text
3,819
```

---

# 19. Pinned row groups

Use the first eight parquet row groups.

Each contains 100 rows.

| RG | SHA-256 | Decoded content bytes |
|---:|---|---:|
| 0 | `1dc3141e4809ea628b17c3ca7b81d64e6ca92bce18dd5765ecd618bfc7867954` | 5,481,145 |
| 1 | `c9d3b52013933c803f4886edbce68da0ae483347a4a6ff3e0f5ced1db4a7e653` | 5,465,901 |
| 2 | `6e0578a90a07a9e25e65b765881d3fa33d6a797624425e01026980d7287f0bf6` | 5,379,166 |
| 3 | `00cdfb7aabe924f31b9f1bb1ba4849040051e5619567b68bf99fdcbcab15131a` | 5,821,303 |
| 4 | `fc063442fb20e7a60c2533ab44dcc69a22ad59f5ce24921fe6af5f53ceab7e1a` | 5,163,559 |
| 5 | `2d7e29331bd4e93e0c8a1caa9a83b8f1ede9b17af6dae9377b83f56c56f05689` | 4,713,140 |
| 6 | `49423d91780cb184c4b0069630e85acccb314a113256ced9692a5138c4782ef1` | 5,069,794 |
| 7 | `3a010831456f185399579b4ecf9d46222f95368c3cdbbc2ff103a258b9c16e2f` | 5,309,891 |

The notebook should access only the required row groups through HTTPS range requests rather than downloading the 210 MB shard in full.

---

# 20. Belfort sample

Total:

```text
800 text lines
```

Split:

```text
train        600
validation    60
test          140
```

Seed:

```text
42
```

Pinned sample digest:

```text
b7e1dd684691a0eedb63a609311f4964e7732e5c1a8d254fe4e1293a8cd0964d
```

The workshop performs **no training**, but the existing split is reused to preserve comparability with the live carrier records.

---

# 21. Why keep train/validation if nothing is trained?

The training portion is needed only for the model-free constant baseline:

```text
constant transcript = medoid training transcript
```

Validation is retained for provenance and split parity.

Only the 140 held-out lines are scored against the two models.

---

# 22. Belfort common benchmark

For every held-out image:

## GOT-OCR

```text
mode = plain
max_new_tokens = 128
```

## SmolDocling

```text
instruction = "Convert this page to docling."
max_new_tokens = 160
```

For SmolDocling:

```text
doctags
↓
doctags_to_text
↓
normalized text
```

Both hypotheses then pass through exactly the same text normalization and evaluator.

---

# 23. Text normalization

Before CER/WER:

1. Unicode-preserve text;
2. strip outer whitespace;
3. replace whitespace runs with a single space.

Do **not**:

- lowercase;
- remove accents;
- remove punctuation;
- normalize spelling;
- translate.

The reference and hypothesis receive the same normalization.

---

# 24. Character error rate

Use corpus-level micro CER:

```text
total character edit distance
--------------------------------
total reference characters
```

Also report macro CER:

```text
mean per-line CER
```

CER must remain **uncapped**.

A value above `1.0` is valid when hallucinated text causes more edit operations than the reference length.

---

# 25. Word error rate

Use:

```text
substitutions + deletions + insertions
---------------------------------------
number of reference words
```

Report:

```text
micro WER
macro WER
```

Also uncapped.

---

# 26. Additional common transcription metrics

For each model report:

```text
exact_match
reference_characters
hypothesis_characters
hypothesis/reference length ratio
truncated_lines
empty_hypotheses
```

Also calculate:

```text
median per-line CER
P90 per-line CER
```

for error-distribution context.

---

# 27. Common baselines

## Empty baseline

Predict:

```text
""
```

for every line.

By construction:

```text
CER = 1.0
WER = 1.0
```

for non-empty references.

---

## Constant-transcript baseline

Find the transcript medoid from the 600 training lines.

Predict that same transcript for all 140 test images.

Report its CER/WER under the same evaluator.

---

# 28. Primary handwriting comparison table

Produce:

| System | CER | CER macro | WER | WER macro | Exact | Length ratio | Truncated |
|---|---:|---:|---:|---:|---:|---:|---:|
| Empty baseline | measured | measured | measured | measured | measured | — | — |
| Constant baseline | measured | measured | measured | measured | measured | measured | — |
| GOT-OCR 2.0 | measured | measured | measured | measured | measured | measured | measured |
| SmolDocling | measured | measured | measured | measured | measured | measured | measured |

Do not hard-code the historical carrier results.

The workshop must record the current execution.

---

# 29. Error-category analysis

Automatically identify line examples from:

### Exact

Hypothesis equals normalized reference.

### Mostly correct

```text
CER < 0.20
```

### Partial

```text
0.20 <= CER < 0.80
```

### Severe

```text
CER >= 0.80
```

### Runaway / hallucinated

Example diagnostic:

```text
hypothesis length > 1.5 × reference length
```

### Empty

No recognized text.

### Truncated

Token budget exhausted.

These are analysis categories, not benchmark conventions.

---

# 30. Cross-model line disagreement

For every test line export:

```text
image_id
reference

got_text
got_cer
got_wer
got_truncated

smoldocling_text
smoldocling_cer
smoldocling_wer
smoldocling_truncated

lower_cer_model
```

Useful aggregate categories:

```text
both_good
got_only_good
smoldocling_only_good
both_poor
```

with:

```text
good = CER < 0.20
```

Again, this is diagnostic only.

---

# 31. Deterministic rendered-page suite

Create document pages entirely in notebook code using Pillow-bundled fonts.

No external page-image assets.

Recommended canonical suite:

```text
Page A — Notice
Page B — Report with tables
Page C — Invoice-style document
Page D — Formula / technical note
```

Every page stores:

```text
image
reference_text
expected structural elements
page_kind
```

---

# 32. Page A — Notice

Include:

- one large title;
- one subtitle/date;
- three short paragraphs;
- one three-item bulleted list;
- footer.

Purpose:

```text
simple printed OCR
+
reading-order sanity
+
page header/footer behavior
```

---

# 33. Page B — Report with tables

Include:

- one level-1 heading;
- two paragraphs;
- one `8 × 5` table;
- one `5 × 3` table;
- short closing text.

This mirrors the kind of synthetic report page already exercised by the SmolDocling carrier.

Purpose:

```text
running text
+
table content
+
table structure
+
reading order
```

---

# 34. Page C — Invoice-style document

Include:

```text
INVOICE

Invoice Number
Date
Customer
Address

line-item table

Subtotal
Tax
Total

payment note
```

Purpose:

```text
key-value style layout
+
small tables
+
numeric OCR
+
spatial organization
```

The notebook should state that this is an authored synthetic page, not an invoice benchmark.

---

# 35. Page D — Formula / technical note

Include:

- title;
- one explanatory paragraph;
- two rendered mathematical expressions;
- units / numeric values;
- one concluding note.

Purpose:

```text
plain OCR
+
formula-oriented native outputs
```

The formula rendering should use only characters supported by the bundled font or deterministic drawing primitives.

No hidden font downloads.

---

# 36. Common full-page text evaluation

For every synthetic page:

## GOT-OCR

Run:

```text
plain
```

## SmolDocling

Run:

```text
Convert this page to docling.
```

then extract plain text from DocTags.

Measure:

```text
CER
WER
exact match
hypothesis/reference length ratio
truncated
new tokens
```

---

# 37. Token precision and recall

Because full-page reading order can change WER substantially, also calculate bag-of-word-token overlap:

```text
token_precision
token_recall
token_F1
```

using token multiplicity.

This is supplemental.

It does not replace WER.

---

# 38. Printed-page comparison

Produce:

| Page | Model | CER | WER | Token F1 | Length ratio | Tokens generated | Truncated |
|---|---|---:|---:|---:|---:|---:|---|
| Notice | GOT | ... | ... | ... | ... | ... | ... |
| Notice | SmolDocling | ... | ... | ... | ... | ... | ... |
| Report | GOT | ... | ... | ... | ... | ... | ... |
| Report | SmolDocling | ... | ... | ... | ... | ... | ... |
| Invoice | ... | ... | ... | ... | ... | ... | ... |
| Technical note | ... | ... | ... | ... | ... | ... | ... |

---

# 39. Native GOT-OCR `format` section

Run GOT-OCR `format` mode on:

```text
Report
Invoice
Formula / technical note
```

Export the raw generated strings.

The notebook SHOULD display them in Markdown code blocks.

Do not score their markup against SmolDocling DocTags.

Record only objective generation diagnostics:

```text
new_tokens
truncated
output_characters
line_count
```

Optionally count common formatting characters:

```text
|
#
$
\
_
*
```

as descriptive output statistics.

These counts are not structure-quality metrics.

---

# 40. Why no common structure metric for GOT `format`

Markdown/LaTeX-style output and DocTags encode structure differently.

Forcing both into one parser would introduce a large notebook-authored normalization layer that could dominate the comparison.

Therefore:

```text
text fidelity = shared quantitative metric

structure = native-model analysis
```

---

# 41. Native SmolDocling structure evaluation

For every rendered page, calculate:

```text
element counts by type
number of document elements
number of <loc_N> tokens
number of OTSL table-cell tokens
wrapped_in_doctag
DocTags character count
```

Compare observed element counts with the authored expected counts.

---

# 42. Expected element counts

The renderer should explicitly track expected logical elements, for example:

```python
{
    "section_header_level_1": 1,
    "text": 3,
    "otsl": 2,
    "page_footer": 1,
}
```

The precise expected schema depends on each rendered page.

Do not derive expected counts by inspecting model output.

---

# 43. Element-count diagnostics

For each page/tag report:

```text
expected_count
observed_count
absolute_count_error
```

Aggregate:

```text
exact_count_tags
mean_absolute_count_error
```

This is a **sample sanity metric**, not a document-layout benchmark.

---

# 44. Location-token validation

For SmolDocling output check:

```text
all loc indices in 0..500

n_loc_tokens divisible by 4
```

for element types that carry boxes.

Also record:

```text
n_location_groups
```

No geometric IoU is scored unless a future rendered-page contract records canonical element boxes in the same coordinate system.

---

# 45. OTSL table diagnostics

For pages containing tables report:

```text
expected tables
observed <otsl> elements

expected logical cells
observed OTSL cell tokens
```

Separately report counts of:

```text
fcel
ecel
ched
rhed
srow
lcel
ucel
xcel
nl
```

The notebook should visually expose the raw OTSL output for inspection.

---

# 46. Specialized SmolDocling instruction demonstrations

Use deterministic pages only.

### Table page

```text
Convert table to OTSL.
```

### Formula page

```text
Convert formula to LaTeX.
```

### Report page

```text
Find all 'text' elements on the page, retrieve all section headers.
```

### Notice page

```text
Detect footer elements on the page.
```

These are demonstrations.

The common text benchmark remains based on:

```text
Convert this page to docling.
```

---

# 47. Capability matrix

The notebook should render:

| Capability | GOT-OCR 2.0 | SmolDocling |
|---|---|---|
| Plain text transcription | Yes | Yes, via DocTags text |
| Formatted text | Yes | Via structured DocTags |
| Explicit document elements | No | Yes |
| Explicit element locations | No | Yes |
| Reading-order markup | Limited/implicit | Yes |
| OTSL tables | No | Yes |
| Formula-specific instruction | Format mode | Yes |
| Table-specific instruction | Format mode | Yes |
| Per-character confidence | No | No |
| Word boxes | No | Element-level locations only |
| PDF rendering | No | No |
| Multi-page workflow in DIMER carrier | No | No |

---

# 48. Token-budget experiment

Generation length is a practical deployment variable.

On the rendered report page run:

## GOT-OCR

```text
256
512
1024
2048
```

## SmolDocling

```text
256
512
1024
2048
4096
```

For each run record:

```text
generated tokens
truncated
CER
WER
text length
```

For SmolDocling also record:

```text
element count
OTSL cell count
```

---

# 49. Interpretation of token budgets

The experiment should demonstrate:

```text
too-small budget
→ incomplete extraction
```

but also:

```text
larger budget
≠ automatically better extraction
```

because generative models can repeat or hallucinate content.

No universal page-token budget should be recommended.

---

# 50. Image-shape robustness experiment

The models preprocess pages very differently.

Use the rendered notice page to create:

```text
portrait original
wide aspect ratio
tall aspect ratio
low-resolution page
```

The actual reference content remains unchanged.

Measure common text CER/WER.

This illustrates:

```text
GOT fixed-square distortion
vs
SmolDocling tiled aspect-preserving page processing
```

without claiming preprocessing alone causes every observed difference.

---

# 51. Low-resolution variant

Create a deterministic low-resolution copy by:

1. downsampling to approximately 50% linear resolution;
2. upsampling back to the original dimensions.

Run through normal model preprocessing.

Measure changes in:

```text
CER
WER
length ratio
```

---

# 52. Workshop exercise

Before revealing outputs from the table page, ask:

1. Which model do you expect to recover the visible words more accurately?
2. Which model can explicitly represent table structure?
3. Could the model with lower WER still have worse structural extraction?
4. Could a structurally valid result contain incorrect cell text?
5. Why is a single "document extraction accuracy" number insufficient?

Then reveal both outputs.

---

# 53. Runtime

Use the shared current carrier environment:

```text
Python 3.12

torch==2.14.0
torchvision==0.29.0
torchaudio==2.11.0

transformers==4.57.6
safetensors==0.8.0

numpy==2.5.3
pillow==11.3.0
huggingface-hub==0.36.2
pyarrow==25.0.1
```

GPU recommended.

---

# 54. Precision

For reproducibility with the current carriers:

```text
float32
```

for both models.

The notebook should not introduce a mixed-precision optimization unless separately qualified.

---

# 55. Sequential model execution

Do not hold both models in GPU memory simultaneously.

Canonical sequence:

```text
stage both snapshots

load GOT
→ Belfort evaluation
→ rendered-page plain evaluation
→ format demonstrations
→ token-budget / robustness experiments
→ retain outputs
→ unload GOT
→ empty CUDA cache

load SmolDocling
→ Belfort evaluation
→ rendered-page conversion
→ structure analysis
→ specialized instructions
→ token-budget / robustness experiments
→ retain outputs
→ unload
```

---

# 56. GOT-OCR supply-chain contract

Manifest:

```text
stepfun-ai/GOT-OCR-2.0-hf
@ d3017ef2c2c1395888c8d635c5e0508bcb0ac78d
```

Required principal asset:

```text
model.safetensors
1,121,114,488 bytes
SHA-256
6175ac7868a4e75735f5d59f78c465081ad3427eb4f312d072a0f1d16b333ba4
```

The full 8-file manifest must be carried.

---

# 57. SmolDocling supply-chain contract

Manifest:

```text
docling-project/SmolDocling-256M-preview
@ ce51f56c4ebe36e0b1c3a55f67b261ba22a50bf8
```

Required principal asset:

```text
model.safetensors
513,028,808 bytes
SHA-256
cdcdf5d823c5684029c7d8e52177cf10f9034b3aba6577549cfb1a9ce36ad0a2
```

The full 13-file manifest must be carried.

---

# 58. Snapshot loading rules

For both models:

1. fetch only manifest-listed files;
2. fetch from immutable revision only;
3. verify bytes;
4. verify SHA-256;
5. reject any mismatch;
6. load from verified local directory;
7. use `trust_remote_code=False`;
8. no network fallback after verification.

---

# 59. Standalone requirement

The notebook MUST NOT:

```text
git clone

pip install -e .

import got_ocr2_pipeline

import smoldocling_document_extraction_pipeline

download DIMER Python modules

call a DIMER worker

call a DIMER API
```

Carry notebook-local implementations of the required inference/evaluation functionality.

---

# 60. Default parameters

```python
USE_BYOD = False
BYOD_PATH = ""
BYOD_TRANSCRIPTS = ""

RUN_BELFORT = True
RUN_PAGE_SUITE = True
RUN_TOKEN_BUDGET_EXPERIMENT = True
RUN_SHAPE_ROBUSTNESS = True

GOT_LINE_MAX_NEW_TOKENS = 128
GOT_PAGE_MAX_NEW_TOKENS = 1024

SMOLDOC_LINE_MAX_NEW_TOKENS = 160
SMOLDOC_PAGE_MAX_NEW_TOKENS = 2048

OUTPUT_DIR = "outputs/document_extraction"
```

No interaction required for default `Run all`.

---

# 61. Input limits

Common validation ceiling:

```text
minimum side = 16 px
maximum side = 16,384 px
maximum pixels = 4096 × 4096
```

One image corresponds to one page/region.

Images are converted to RGB.

---

# 62. Common result schema

For the common text path:

```text
model
sample_id
sample_kind

reference_text
hypothesis_text

cer
wer
exact_match

reference_chars
hypothesis_chars
length_ratio

new_tokens
truncated

inference_seconds
```

---

# 63. Model-native result fields

## GOT

```text
mode
raw_text
```

## SmolDocling

```text
instruction
doctags
plain_text

element_counts
n_elements
n_loc_tokens
n_table_cells
wrapped_in_doctag
```

---

# 64. BYOD

Support two modes.

## A. Labelled OCR evaluation

Directory or ZIP:

```text
images/
transcripts.csv
```

CSV:

```text
file,text
```

optional:

```text
id
```

Recommended:

```text
8..500 images
```

---

# 65. BYOD labelled workflow

For every image run:

```text
GOT plain
SmolDocling default conversion
```

Evaluate both against the reference transcript.

Produce the same:

```text
CER
WER
exact match
length
truncation
```

comparison.

---

# 66. BYOD unlabelled workflow

If no transcript file is supplied:

```text
evaluation verdict = not-measurable
```

Still export:

```text
GOT plain output
GOT format output

SmolDocling DocTags
SmolDocling plain text
SmolDocling structure summary
```

---

# 67. BYOD structured expectations

Optionally support:

```text
structure.json
```

with page-level expectations such as:

```json
{
  "file": "report.png",
  "expected_counts": {
    "section_header_level_1": 1,
    "text": 3,
    "otsl": 2
  }
}
```

Only SmolDocling structure diagnostics use this file.

GOT must not be penalized for not producing DocTags.

---

# 68. BYOD privacy

Document images may contain:

- names;
- addresses;
- signatures;
- account numbers;
- legal records;
- financial data;
- research data;
- proprietary content.

Warn prominently:

> Do not upload confidential, restricted, personal, regulated, or otherwise unauthorized documents to a hosted notebook runtime.

The standalone models run locally inside the selected notebook runtime.

---

# 69. Required exports

Output directory:

```text
outputs/document_extraction/
```

---

# 70. `belfort_results.csv`

```text
sample_id
reference

model
hypothesis

cer
wer
exact_match
reference_chars
hypothesis_chars
length_ratio
new_tokens
truncated
inference_seconds
```

---

# 71. `belfort_metrics.csv`

```text
system
cer
cer_macro
wer
wer_macro
exact_match
median_cer
p90_cer
reference_chars
hypothesis_chars
length_ratio
empty_hypotheses
truncated
```

---

# 72. `page_text_metrics.csv`

```text
page_id
page_kind
model

cer
wer
token_precision
token_recall
token_f1

reference_chars
hypothesis_chars
length_ratio

new_tokens
truncated
inference_seconds
```

---

# 73. `got_format_outputs.json`

For every formatted-page demonstration:

```text
page_id
raw_output
new_tokens
truncated
characters
lines
```

---

# 74. `smoldocling_structure.csv`

```text
page_id
element_type
expected_count
observed_count
absolute_count_error
```

---

# 75. `smoldocling_summary.json`

For every page:

```text
element_counts
n_elements
n_loc_tokens
n_location_groups
n_table_cells
wrapped_in_doctag
doctags_chars
```

---

# 76. `specialized_instruction_outputs.json`

Store SmolDocling outputs for:

```text
table → OTSL
formula → LaTeX
text + section headers
footer detection
```

---

# 77. `token_budget.csv`

```text
model
page_id
token_budget

generated_tokens
truncated

cer
wer
text_chars

n_elements
n_table_cells
```

SmolDocling-only structure fields may be empty for GOT.

---

# 78. `shape_robustness.csv`

```text
model
variant

cer
wer
length_ratio
new_tokens
truncated
```

---

# 79. `metrics.json`

Nested structure:

```text
baselines

Belfort:
  GOT
  SmolDocling

rendered_pages:
  GOT
  SmolDocling

SmolDocling_structure

token_budget

shape_robustness

timings
```

---

# 80. `provenance.json`

Record:

```text
notebook_spec
profile
pedagogical_mode

GOT:
  model ID
  revision
  license
  manifest
  weight SHA-256
  parameter count
  preprocessing
  decoding

SmolDocling:
  model ID
  revision
  license
  manifest
  weight SHA-256
  parameter count
  preprocessing
  decoding

Belfort:
  repo
  revision
  shard
  row-group pins
  sample digest
  split seed
  split sizes

rendered-page generator version
page references
expected structures

runtime
device
dtype
timings
```

---

# 81. Example panels

Export a predetermined subset of Belfort lines:

```text
reference
GOT output
SmolDocling output
```

Recommended:

```text
6 examples
```

selected by stable test index, not by model performance.

---

# 82. Page visual outputs

Export:

```text
pages/
  notice.png
  report.png
  invoice.png
  technical_note.png
```

and optional sidecar text files:

```text
reference.txt

got_plain.txt
got_format.txt

smoldocling.txt
smoldocling.doctags
```

---

# 83. Runtime/resource comparison

Record:

| Model | Params | Weight bytes | Input strategy | Mean line latency | Mean page latency | Peak VRAM |
|---|---:|---:|---|---:|---:|---:|
| GOT-OCR 2.0 | 560.5M | 1.12 GB | 1024 square | measured | measured | measured |
| SmolDocling | 256.5M | 513 MB | 2048 + tiles | measured | measured | measured |

Timing must exclude:

- model download;
- dataset download;
- model load.

Model-load time is reported separately.

---

# 84. Important resource interpretation

A smaller checkpoint does not necessarily imply a faster page conversion.

Generation length, image token count, tile count, decoder architecture, and page complexity also affect runtime.

Do not reduce the comparison to parameter count.

---

# 85. SmolDocling preview status

The notebook should state that the pinned checkpoint is explicitly named:

```text
SmolDocling-256M-preview
```

The workshop evaluates that exact live DIMER profile.

Results must not automatically be attributed to later Docling/Granite successors.

---

# 86. Hallucination warning

Both systems are generative.

They can:

```text
omit visible text
invent text
repeat text
reorder text
generate output for an image with little/no text
```

Neither output contains a calibrated correctness score.

Human verification remains necessary where transcription accuracy matters.

---

# 87. Blank-page probe

Include a deterministic blank white page.

Run both models.

Record:

```text
text generated
characters
tokens
truncated
```

Expected result is **not asserted**.

Any non-empty output is evidence that generative OCR output should not be treated as proof that text was visibly present.

---

# 88. Noise-page probe

Create deterministic visual noise with a fixed seed.

Run both models.

Do not score against text.

Record generated output only.

Purpose:

```text
illustrate generative behavior under out-of-distribution inputs
```

---

# 89. Explicit non-goals

This notebook does not perform:

- model fine-tuning;
- Belfort adaptation;
- adapter export;
- PDF rasterization;
- multi-page document aggregation;
- document classification;
- Document QA;
- key-value QA;
- table QA;
- handwriting model training;
- calibrated OCR confidence;
- word-box OCR;
- post-hoc LLM correction;
- spelling correction;
- translation;
- production DIMER API calls.

---

# 90. Why no fine-tuning?

Both individual DIMER carriers already provide validated E2E adaptation workflows on the same Belfort sample.

Repeating those adaptation stages here would turn the notebook into:

```text
model-specific fine-tuning comparison
```

rather than:

```text
OCR vs structured extraction
```

The workshop intentionally evaluates the **frozen live capabilities**.

---

# 91. Relationship to other Document Intelligence notebooks

The track becomes:

```text
Document Intelligence
│
├── Document-Type Classification
│     └── DiT / RVL-CDIP
│
├── Document QA
│     ├── LayoutLM
│     └── Pix2Struct DocVQA
│
├── OCR / Document Extraction
│     ├── GOT-OCR 2.0        ← this notebook
│     └── SmolDocling
│
└── Table Intelligence
      ├── Table Transformer Detection
      ├── Table Structure Recognition
      ├── TAPAS
      └── DePlot
```

---

# 92. Notebook cell plan

| # | Type | Section |
|---:|---|---|
| 0 | Markdown | Title, profile, objectives |
| 1 | Markdown | OCR vs document extraction |
| 2 | Markdown | GOT-OCR architecture and contract |
| 3 | Markdown | SmolDocling architecture and DocTags |
| 4 | Code | Form parameters |
| 5 | Markdown | Runtime |
| 6 | Code | Install/check pinned dependencies |
| 7 | Markdown | Immutable model provenance |
| 8 | Code | Two manifests + staging helpers |
| 9 | Markdown | Belfort provenance |
| 10 | Code | Pinned range-reader and row-group validation |
| 11 | Code | Reproduce 600/60/140 split |
| 12 | Markdown | Common text metrics |
| 13 | Code | CER/WER/baseline helpers |
| 14 | Markdown | Rendered-page suite |
| 15 | Code | Generate Notice/Report/Invoice/Technical Note |
| 16 | Markdown | GOT-OCR inference |
| 17 | Code | Load verified GOT |
| 18 | Code | Belfort plain evaluation |
| 19 | Code | Rendered-page plain evaluation |
| 20 | Code | GOT format demonstrations |
| 21 | Code | GOT budget/shape/blank/noise probes |
| 22 | Code | Unload GOT |
| 23 | Markdown | SmolDocling inference |
| 24 | Code | Load verified SmolDocling |
| 25 | Code | Belfort DocTags→text evaluation |
| 26 | Code | Rendered-page common text evaluation |
| 27 | Markdown | SmolDocling structure analysis |
| 28 | Code | Element counts / loc / OTSL metrics |
| 29 | Markdown | Specialized instructions |
| 30 | Code | OTSL / formula / headers / footer |
| 31 | Code | SmolDocling budget/shape/blank/noise probes |
| 32 | Markdown | Common model comparison |
| 33 | Code | Belfort + printed-page tables |
| 34 | Markdown | Workshop prediction exercise |
| 35 | Code | Cross-model disagreement examples |
| 36 | Markdown | Resource comparison |
| 37 | Code | Timing/VRAM/parameter table |
| 38 | Markdown | Machine-readable exports |
| 39 | Code | CSV/JSON/text outputs |
| 40 | Markdown | BYOD |
| 41 | Code | Optional labelled/unlabelled BYOD |
| 42 | Markdown | Interpretation and limitations |
| 43 | Code | Terminal summary + output assertions |

---

# 93. Required assertions

## Snapshots

```text
GOT model ID matches
GOT revision matches
all 8 GOT files verified

SmolDocling model ID matches
SmolDocling revision matches
all 13 SmolDocling files verified
```

---

## Dataset

```text
8 pinned row groups
800 lines
sample digest matches

train = 600
validation = 60
test = 140

no decoded image appears across splits
```

---

## Text results

```text
one result per test image

all references non-empty

CER/WER finite
generated-token counts non-negative

truncated field boolean
```

CER/WER are **not** required to be ≤1.

---

## GOT

```text
mode in {plain, format}
generation budget within 1..4096
```

---

## SmolDocling

```text
instruction in supported instruction list
generation budget within 1..8192

returned DocTags string
plain text recoverable
all parsed loc indices within 0..500
```

---

# 94. Assertions that MUST NOT exist

Do not require:

```text
GOT must have lower CER

SmolDocling must have better WER

SmolDocling must recover both tables

GOT must preserve formulas

blank page must produce empty output

larger token budget must improve CER

one model must be faster
```

Those are empirical outcomes.

---

# 95. Release verification

Preferred qualification:

```text
Kaggle Tesla T4
Python 3.12
```

Cold-run conditions:

```text
empty HF cache
empty Belfort cache
no repository checkout
no credentials
no DIMER service
```

Record:

```text
notebook commit
notebook blob SHA

runtime
GPU

GOT:
  revision
  snapshot digest
  load time
  peak GPU memory
  line timing
  page timing

SmolDocling:
  revision
  snapshot digest
  load time
  peak GPU memory
  line timing
  page timing

Belfort:
  row-group pins
  sample digest
  split sizes

baseline metrics
GOT text metrics
SmolDocling text metrics

page-suite metrics
structure diagnostics
token-budget experiment
shape robustness
blank/noise outputs

cell success count
wall time
output inventory
```

---

# 96. Terminal summary

Final cell:

```text
DIMER OCR & Document Extraction Workshop
-----------------------------------------

Belfort held-out lines: 140

System                    CER      WER     Exact
------------------------------------------------
Empty baseline            ...      ...      ...
Constant baseline         ...      ...      ...
GOT-OCR 2.0               ...      ...      ...
SmolDocling               ...      ...      ...

Rendered pages: 4

GOT-OCR 2.0
  mean page CER: ...
  mean page WER: ...
  truncated pages: ...

SmolDocling
  mean page CER: ...
  mean page WER: ...
  exact expected element-count tags: ...
  total OTSL cells: ...
  truncated pages: ...

Outputs:
  outputs/document_extraction/
```

No automatic:

```text
Winner
Best model
```

---

# 97. Interpretation requirements

## OCR quality and extraction quality are different

A transcript can be textually accurate but structurally poor.

A structured document can have correct layout elements while containing transcription errors.

Both dimensions matter.

---

## Reading order affects WER

A model may recover nearly every visible word yet receive poor WER if it emits them in a different order.

Token-F1 supplements WER for this reason.

---

## Structure is model-specific

SmolDocling exposes explicit structural semantics that GOT-OCR's output contract does not.

Absence of DocTags from GOT is not a failure.

---

## GOT format mode is not DocTags

Markdown/LaTeX-like formatting and a document-layout representation solve related but different problems.

---

## Generation budgets matter

Dense pages can legitimately require many output tokens.

A `truncated=true` result must never be presented as a complete extraction.

---

## Both models can hallucinate

Generated text should be verified against source imagery for consequential use.

---

## Domain transfer matters

Belfort handwriting differs considerably from modern printed business documents.

The rendered-page suite is synthetic.

Neither is sufficient evidence for a specific production deployment.

---

# 98. Workshop learning arc

**Page image**  
↓  
*What information is visually present?*

**OCR**  
↓  
*Recover the textual sequence.*

**GOT-OCR plain mode**  
↓  
*Focus on transcription.*

**Formatted OCR**  
↓  
*Attempt to preserve some textual structure.*

**Document conversion**  
↓  
*Represent text together with document elements and layout.*

**SmolDocling DocTags**  
↓  
*Headings, text regions, locations, tables and reading order.*

**Common text benchmark**  
↓  
*CER/WER compare transcription fidelity.*

**Native structure analysis**  
↓  
*Structure is evaluated according to each model's actual output contract.*

**Deployment lesson**  
↓  
*The right document model depends on whether the downstream system needs words alone, formatted content, or an explicit structured representation of the page.*