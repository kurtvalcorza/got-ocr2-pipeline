"""Supplemental OCR notebook checks; no real models or hosted qualification.

Since 2026-10-03 the model, BYOD and export code runs in the uv isolated environment as stages of the
carried file tools/ocr_workshop.py; these checks call those stages in-process with model doubles.
"""
import csv
import hashlib
import io
import zipfile
from pathlib import Path

import pytest
from PIL import Image

NOTEBOOK = Path(__file__).resolve().parents[1] / 'tutorials/DIMER_OCR_Document_Extraction_Workshop.ipynb'


STAGE_FILE = Path(__file__).resolve().parents[1] / 'tools/ocr_workshop.py'


def stage_ns(tmp_path):
    """Globals of the carried stage file as one stage process sees them, with directories under tmp_path."""
    ns = {'__name__': 'ocr_workshop_under_test'}
    exec(compile(STAGE_FILE.read_text(encoding='utf-8'), str(STAGE_FILE), 'exec'), ns)
    ns.update(OUTPUT_DIR=str(tmp_path/'outputs'), WORK_DIR=str(tmp_path/'work'))
    return ns


def run_byod(ns):
    """The notebook's BYOD cell runs this stage; it returns the report it also saves as state."""
    ns['byod_report'] = ns['stage_byod']()


def setup(tmp_path):
    state = {'live': 0, 'loads': 0, 'fail': False}

    class Model:
        def __init__(self):
            assert state['live'] == 0
            state['live'] += 1
            state['loads'] += 1

        def __del__(self):
            state['live'] -= 1

    def load():
        return Model(), object(), 0, 0, 0

    def generate(model, processor, pad, stop, images, mode, budget):
        if state['fail']:
            raise RuntimeError('injected generation failure')
        return [
            dict(text='hello', doctags='<text>hello</text>', new_tokens=1, truncated=False)
            for _ in images
        ]

    ns = stage_ns(tmp_path)
    assert (ns['USE_BYOD'], ns['MIN_IMAGE_SIDE'], ns['MAX_IMAGE_SIDE'], ns['MAX_IMAGE_PIXELS']) == (
        False, 16, 16384, 4096**2)
    ns.update(BATCH_SIZE=8, GOT_PAGE_MAX_NEW_TOKENS=1024, SMOLDOC_PAGE_MAX_NEW_TOKENS=2048,
              SMOL_DEFAULT='convert', GOT_MANIFEST={'revision': 'got-fixed'},
              SMOL_MANIFEST={'revision': 'smol-fixed'}, RUNTIME={'test_double': True},
              load_got=load, load_smoldoc=load, got_generate=generate, smol_generate=generate,
              empty_device_cache=lambda: None)
    run_byod(ns)
    assert ns['byod_report'] is None
    return ns, state


def pages(tmp_path, names=('page.png',)):
    root = tmp_path/'inputs'
    root.mkdir(exist_ok=True)
    for name in names:
        Image.new('RGB', (32, 40), 'white').save(root/name, format='PNG')
    return root


@pytest.mark.parametrize('reference', [None, 'hello', ''])
def test_byod_pipeline_exports_and_blank_reference(tmp_path, reference):
    ns, state = setup(tmp_path)
    root = pages(tmp_path)
    if reference is not None:
        (root/'transcripts.csv').write_text('file,text,id\npage.png,'+reference+',custom\n')
    ns.update(USE_BYOD=True, BYOD_PATH=str(root))
    run_byod(ns)
    report = ns['byod_report']
    output = Path(report['output_directory'])
    assert report['evaluation_verdict'] == ('not-measurable' if reference is None else 'measured')
    assert report['input']['images']['page.png'] == hashlib.sha256((root/'page.png').read_bytes()).hexdigest()
    assert len(report['native_output_files']) == 4
    assert all((output/name).is_file() for name in report['native_output_files'])
    with (output/'results.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]['output_stem'] == 'page-0001'
    if reference == '':
        assert rows[0]['cer'] == '' and rows[0]['char_edits'] == '5'
        assert report['undefined_rate_rows'] == 2
    elif reference == 'hello':
        assert float(rows[0]['cer']) == 0
    assert state['live'] == 0 and state['loads'] == 2
    run_byod(ns)
    assert Path(ns['byod_report']['output_directory']) != output
    assert (output/'report.json').is_file()


@pytest.mark.parametrize(('rows', 'message'), [
    ('page.png,hello,../escape\n', 'Unsafe'),
    ('page.png,hello,x\npage.png,other,y\n', 'Duplicate'),
    ('extra.png,hello,x\n', 'missing='),
    ('page.png,hello,x\nextra.png,hello,y\n', 'extra='),
])
def test_transcript_validation_precedes_model_load(tmp_path, rows, message):
    ns, state = setup(tmp_path)
    root = pages(tmp_path)
    (root/'transcripts.csv').write_text('file,text,id\n'+rows)
    ns.update(USE_BYOD=True, BYOD_PATH=str(root))
    with pytest.raises(ValueError, match=message):
        run_byod(ns)
    assert state['loads'] == 0
    assert not (tmp_path/'escape.got_plain.txt').exists()


def test_duplicate_ids_and_unique_filename_defaults(tmp_path):
    ns, _ = setup(tmp_path)
    root = pages(tmp_path, ('page.png', 'page.jpg'))
    records, _, _ = ns['load_byod'](root)
    assert {r['id'] for r in records} == {'page.png', 'page.jpg'}
    (root/'transcripts.csv').write_text('file,text,id\npage.png,hello,x\npage.jpg,hello,x\n')
    with pytest.raises(ValueError, match='ids must be unique'):
        ns['load_byod'](root)


def test_zip_collision_and_fresh_staging(tmp_path):
    ns, _ = setup(tmp_path)
    stream = io.BytesIO()
    Image.new('RGB', (32, 40)).save(stream, format='PNG')
    bad = tmp_path/'bad.zip'
    with zipfile.ZipFile(bad, 'w') as z:
        z.writestr('a/page.png', stream.getvalue())
        z.writestr('b/page.png', stream.getvalue())
    with pytest.raises(ValueError, match='Duplicate ZIP basename'):
        ns['load_byod'](bad)
    for filename in ('one.png', 'two.png'):
        path = tmp_path/(filename+'.zip')
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr(filename, stream.getvalue())
        records, labelled, provenance = ns['load_byod'](path)
        assert [r['file'] for r in records] == [filename]
        assert not labelled and provenance['source_zip_sha256']


def test_generation_failure_releases_model_and_can_retry(tmp_path):
    ns, state = setup(tmp_path)
    root = pages(tmp_path)
    ns.update(USE_BYOD=True, BYOD_PATH=str(root))
    state['fail'] = True
    try:
        run_byod(ns)
    except RuntimeError as error:
        assert 'injected' in str(error)
        # Check while the traceback is still reachable, as in an interactive kernel.
        assert state['live'] == 0
    else:
        pytest.fail('generation failure was not propagated')
    assert state['live'] == 0
    state['fail'] = False
    run_byod(ns)
    assert state['live'] == 0 and ns['byod_report']['images'] == 1

