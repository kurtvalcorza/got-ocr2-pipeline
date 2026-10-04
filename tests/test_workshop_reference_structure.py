"""Re-review v2 checks for the supplemental OCR notebook: authored references, structural
counts, learner-visible outputs, single-page BYOD input and transcript headers.

Since 2026-10-03 the notebook runs this code as stages of the carried file tools/ocr_workshop.py in its uv
isolated environment. The stages run here in-process, handing results over as state files exactly as the
separate stage processes do, with deterministic generation stand-ins; the Belfort panel cell still runs from
the notebook. These are offline orchestration checks, not real-model or hosted-runtime evidence.
"""
import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from test_workshop_optional_paths import STAGE_FILE, run_byod, stage_ns
from test_workshop_optional_paths import pages as byod_pages
from test_workshop_optional_paths import setup as byod_setup

NOTEBOOK = Path(__file__).resolve().parents[1] / 'tutorials/DIMER_OCR_Document_Extraction_Workshop.ipynb'
CELLS = {c['id']: ''.join(c['source']) for c in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']}

# Faithful transcripts written independently of the renderer, in visual reading order.
INVOICE = ('INVOICE Invoice Number: INV-2026-0926 Date: 26 September 2026 '
           'Customer: Sample Research Office Address: C.P. Garcia Avenue, Diliman, Quezon City '
           'Description Qty Unit Price Amount Document scan 3 120.00 360.00 OCR review 2 250.00 500.00 '
           'Structure check 1 400.00 400.00 Subtotal: 1,260.00 Tax: 151.20 Total: 1,411.20 '
           'Payment note: This synthetic invoice is for OCR and document extraction testing only.')
NOTICE = ('PUBLIC NOTICE Document Processing Workshop - 26 September 2026 The records office will '
          'conduct a scheduled document digitization activity this Friday. Participants should bring '
          'one sample page and verify all extracted text against the original document. Generated OCR '
          'output may contain omissions, substitutions, repetitions, or invented text and must be '
          'reviewed. Prepare the source document. Check the recognized text. Review the exported '
          'structure. DIMER Document Intelligence Workshop')
TECHNICAL = ('TECHNICAL NOTE The following expressions are rendered as plain text for deterministic '
             'extraction testing. Energy: E = m * c^2 Mean: x_bar = (x1 + x2 + x3) / 3 Measured value: '
             '12.5 kJ. Tolerance: plus or minus 0.2 kJ. The formulas above are instructional content, '
             'not production measurements.')


def base_ns(tmp_path):
    ns = stage_ns(tmp_path)
    ns['pages'] = ns['render_pages']()
    return ns


def test_invoice_reference_follows_drawing_order(tmp_path):
    ns = base_ns(tmp_path)
    pages = ns['pages']
    for pid, transcript in (('invoice', INVOICE), ('notice', NOTICE), ('technical_note', TECHNICAL)):
        metrics = ns['one_metrics'](pages[pid]['reference_text'], transcript)
        assert (metrics['cer'], metrics['wer']) == (0, 0), pid
    # Totals read before the table is a different order, and is charged as one.
    reordered = INVOICE.replace(' Subtotal: 1,260.00 Tax: 151.20 Total: 1,411.20', '').replace(
        'Quezon City', 'Quezon City Subtotal: 1,260.00 Tax: 151.20 Total: 1,411.20')
    assert ns['one_metrics'](pages['invoice']['reference_text'], reordered)['cer'] > 0
    assert pages['notice']['expected_counts']['list_item'] == 3


def test_renderer_refuses_characters_the_font_cannot_draw(tmp_path):
    ns = base_ns(tmp_path)
    page = ns['PageText'](Image.new('RGB', (200, 60), 'white'))
    for text in ('Workshop — 2026', 'régulariser', '• item'):
        with pytest.raises(ValueError, match='default font cannot draw'):
            page.text((0, 0), text, 20)
    page.text((0, 0), 'item', 20, bullet=True)
    assert page.parts == ['item']
    for pid, page in ns['pages'].items():
        assert not ns['missing_glyphs'](page['reference_text'], 20), pid


def test_panels_print_full_text_with_font_fallback(tmp_path, capsys, monkeypatch):
    import importlib.util
    ns = base_ns(tmp_path)
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: None)
    record = {'text': "régulariser l'achat", 'image': Image.new('RGB', (300, 40), 'white')}
    results = {'got_belfort.json': {'items': [{'text': 'GOT é'}] * 116},
               'smol_belfort.json': {'items': [{'text': ''}] * 116}}
    # Kernel helpers of the runtime cell: the staged test lines and the Belfort stages' state files.
    ns.update(OUT_ROOT=tmp_path, RUN_BELFORT=True, display=lambda *a: None,
              belfort_split=lambda name: [record] * 116 if name == 'test' else [], state=results.__getitem__)
    exec(CELLS['9165c76c'], ns)
    out = capsys.readouterr().out
    assert 'Panel font: Pillow default; accented letters may show as boxes' in out
    assert out.count("REF:  régulariser l'achat") == 6 and 'GOT:  GOT é' in out
    assert len(list((tmp_path/'belfort_examples').glob('example_*.png'))) == 6


def test_word_diff_names_the_changed_words(tmp_path):
    ns = base_ns(tmp_path)
    diff = ns['word_diff']('Total: 1,411.20 paid', 'Total: 1,417.20')
    assert diff == [{'op': 'replace', 'reference': '1,411.20 paid', 'model': '1,417.20'}]


@pytest.mark.parametrize(('doctags', 'cells', 'rows', 'grid', 'tokens'), [
    ('<otsl><fcel>a<fcel>b<nl><fcel>c<fcel>d<nl></otsl>', 4, 2, 4, 6),
    ('<otsl><ched>h<lcel><nl><fcel>c<fcel>d<nl></otsl>', 3, 2, 4, 6),
    ('<text>no table</text>', 0, 0, 0, 0),
])
def test_table_counts_exclude_row_delimiters(tmp_path, doctags, cells, rows, grid, tokens):
    summary = base_ns(tmp_path)['doctags_summary'](doctags)
    assert (summary['n_table_cells'], summary['n_table_rows']) == (cells, rows)
    assert (summary['n_table_grid_positions'], summary['n_otsl_tokens']) == (grid, tokens)


def test_unscored_types_are_not_expected_absent(tmp_path):
    ns = base_ns(tmp_path)
    doctags = ('<doctag><section_header_level_1>PUBLIC NOTICE</section_header_level_1><unordered_list>'
               + '<list_item>x</list_item>' * 3 + '</unordered_list><picture></picture></doctag>')
    expected = ns['pages']['notice']['expected_counts']
    rows = ns['structure_rows']('notice', expected, ns['doctags_summary'](doctags))
    by_tag = {r['element_type']: r for r in rows}
    assert by_tag['list_item']['absolute_count_error'] == 0
    assert by_tag['unordered_list'] == {
        'page_id': 'notice', 'element_type': 'unordered_list', 'scored': False,
        'expected_count': None, 'observed_count': 1, 'absolute_count_error': None}
    assert by_tag['page_footer']['scored'] and by_tag['page_footer']['absolute_count_error'] == 1
    assert 'caption' not in by_tag  # neither expected nor observed


def model_ns(tmp_path, budget=True, shape=True):
    ns = base_ns(tmp_path)
    calls = Counter()

    def got_generate(model, processor, pad, stop, images, mode, budget):
        calls['got'] += 1
        return [dict(text=f'GOTMARK-{mode}-{calls["got"]}', new_tokens=5, truncated=False,
                     inference_seconds=0.1) for _ in images]

    def smol_generate(model, processor, end, pad, images, instruction, budget):
        calls['smol'] += 1
        tags = f'<doctag><text>SMOLMARK-{calls["smol"]}</text><otsl><fcel>a<fcel>b<nl></otsl></doctag>'
        return [dict(doctags=tags, text=ns['doctags_to_text'](tags), instruction=instruction, new_tokens=7,
                     truncated=False, inference_seconds=0.1) for _ in images]

    ns.update(got_generate=got_generate, smol_generate=smol_generate,
              load_got=lambda: (None, None, 0, 1, 0.5), load_smoldoc=lambda: (None, None, 1, 0, 0.25),
              GOT_PAGE_MAX_NEW_TOKENS=1024, SMOLDOC_PAGE_MAX_NEW_TOKENS=2048, RUN_PAGE_SUITE=True,
              RUN_BELFORT=False, RUN_TOKEN_BUDGET_EXPERIMENT=budget, RUN_SHAPE_ROBUSTNESS=shape)
    ns['stage_pages']()  # each stage reloads the rendered pages from the files this stage writes
    return ns, calls


def test_page_text_structure_and_specialized_outputs_are_shown(tmp_path, capsys):
    ns, _ = model_ns(tmp_path)
    for stage in ('stage_got_pages', 'stage_smol_pages', 'stage_smol_specialized', 'stage_compare'):
        ns[stage]()
    for page in ('report', 'invoice'):
        ns['stage_inspect'](page)
    out = capsys.readouterr().out
    assert 'GOTMARK-plain' in out and 'GOTMARK-format' in out  # GOT plain previews and native format
    assert '<otsl><fcel>a<fcel>b<nl></otsl>' in out  # native DocTags for one structure
    assert 'Reference · invoice' in out and 'Invoice Number: INV-2026-0926' in out
    assert "reference='INVOICE Invoice Number: INV-2026-0926" in out  # a named word-level difference
    assert 'Convert table to OTSL.' in out and out.count('SMOLMARK') >= 8
    assert 'Unscored observed types' in out
    assert all(r['n_table_cells'] == 2 for r in ns['load_state']('smol_pages.json')['summaries'].values())


@pytest.mark.parametrize('budget', [False, True])
@pytest.mark.parametrize('shape', [False, True])
def test_probe_flags_and_paired_budget_table(tmp_path, capsys, budget, shape):
    ns, calls = model_ns(tmp_path, budget, shape)
    ns['stage_got_probes']()
    ns['stage_smol_probes']()
    out = capsys.readouterr().out
    assert calls['got'] == 2 + 4*budget + 4*shape and calls['smol'] == 2 + 5*budget + 4*shape
    assert ('Paired token-budget comparison' in out) is budget
    if budget:
        smol_budget_rows = ns['load_state']('smol_probes.json')['budget_rows']
        assert all(r['n_table_cells'] == 2 and r['n_table_rows'] == 1 for r in smol_budget_rows)
        assert re.search(r'GOT-OCR 2\.0\s+256\s+5\s+False', out) and re.search(r'SmolDocling\s+4096\s+7', out)


def multipage_tiff(path, frames):
    images = [Image.new('RGB', (64, 96), colour) for colour in ('white', 'black', 'gray')[:frames]]
    images[0].save(path, format='TIFF', save_all=True, append_images=images[1:])


@pytest.mark.parametrize('frames', [1, 2])
def test_multi_frame_tiff_is_refused_before_model_load(tmp_path, frames):
    ns, state = byod_setup(tmp_path)
    root = tmp_path/'inputs'
    root.mkdir()
    multipage_tiff(root/'scan.tiff', frames)
    ns.update(USE_BYOD=True, BYOD_PATH=str(root))
    if frames == 1:
        run_byod(ns)
        assert ns['byod_report']['images'] == 1 and state['loads'] == 2
    else:
        with pytest.raises(ValueError, match='2 pages/frames'):
            run_byod(ns)
        assert state['loads'] == 0


@pytest.mark.parametrize(('csv_text', 'message'), [
    ('file,text,text\npage.png,FIRST,SECOND\n', 'duplicate column names'),
    ('file,text,id\npage.png,hello\n', 'row 1 has 2 fields'),
    ('file,text\npage.png,hello,extra\n', 'row 1 has 3 fields'),
])
def test_ambiguous_transcripts_are_refused_before_model_load(tmp_path, csv_text, message):
    ns, state = byod_setup(tmp_path)
    root = byod_pages(tmp_path)
    (root/'transcripts.csv').write_text(csv_text)
    ns.update(USE_BYOD=True, BYOD_PATH=str(root))
    with pytest.raises(ValueError, match=message):
        run_byod(ns)
    assert state['loads'] == 0


def test_byod_run_previews_extracted_text(tmp_path, capsys):
    ns, _ = byod_setup(tmp_path)
    root = byod_pages(tmp_path)
    (root/'transcripts.csv').write_text('file,text\npage.png,hello\n')
    ns.update(USE_BYOD=True, BYOD_PATH=str(root))
    run_byod(ns)
    out = capsys.readouterr().out
    assert 'page.png · GOT-OCR 2.0 plain' in out and 'page.png · SmolDocling text' in out
    assert 'CER=0.0' in out
    with (Path(ns['byod_report']['output_directory'])/'results.csv').open() as handle:
        assert {r['text'] for r in csv.DictReader(handle)} == {'hello'}


def test_exports_record_reference_policy_and_new_count_columns(tmp_path):
    ns, _ = model_ns(tmp_path)
    text = STAGE_FILE.read_text(encoding='utf-8')
    source = next(ast.get_source_segment(text, n) for n in ast.parse(text).body
                  if isinstance(n, ast.FunctionDef) and n.name == 'stage_export')
    assert '"scored","expected_count"' in source and '"n_table_rows","n_otsl_tokens"' in source
    assert 'renderer v2 (draw-order references)' in source and '"reference_policy":REFERENCE_POLICY' in source
    assert ns['REFERENCE_POLICY'].startswith('Reference = every visible string in drawing order')
    digest = hashlib.sha256(ns['pages']['invoice']['reference_text'].encode('utf-8')).hexdigest()
    assert digest == hashlib.sha256(INVOICE.encode('utf-8')).hexdigest()
