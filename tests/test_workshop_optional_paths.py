"""Supplemental OCR notebook checks; no real models or hosted qualification."""
import ast
import csv
import gc
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from packaging.version import InvalidVersion, Version
from PIL import Image

NOTEBOOK = Path(__file__).resolve().parents[1] / 'tutorials/DIMER_OCR_Document_Extraction_Workshop.ipynb'


def cell(index):
    return ''.join(json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells'][index]['source'])


def helpers(index, names, ns):
    tree = ast.parse(cell(index))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(tree, 'notebook-helper', 'exec'), ns)


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

    ns = dict(Path=Path, Image=Image, np=np, csv=csv, gc=gc, io=io, json=json, re=re,
              hashlib=hashlib, zipfile=zipfile, USE_BYOD=False,
              MIN_IMAGE_SIDE=16, MAX_IMAGE_SIDE=16384, MAX_IMAGE_PIXELS=4096**2,
              OUT_ROOT=tmp_path/'outputs', BATCH_SIZE=8, GOT_PAGE_MAX_NEW_TOKENS=1024,
              SMOLDOC_PAGE_MAX_NEW_TOKENS=2048, SMOL_DEFAULT='convert',
              GOT_MANIFEST={'revision': 'got-fixed'}, SMOL_MANIFEST={'revision': 'smol-fixed'},
              RUNTIME={'test_double': True}, load_got=load, load_smoldoc=load,
              got_generate=generate, smol_generate=generate,
              torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    helpers(9, {'sha256_file'}, ns)
    helpers(11, {'normalise_text'}, ns)
    helpers(13, {'edit_distance', 'words', 'one_metrics'}, ns)
    helpers(47, {'write_csv'}, ns)
    exec(cell(51), ns)
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
    exec(cell(51), ns)
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
    exec(cell(51), ns)
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
        exec(cell(51), ns)
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
        exec(cell(51), ns)
    except RuntimeError as error:
        assert 'injected' in str(error)
        # Check while the traceback is still reachable, as in an interactive kernel.
        assert state['live'] == 0
    else:
        pytest.fail('generation failure was not propagated')
    assert state['live'] == 0
    state['fail'] = False
    exec(cell(51), ns)
    assert state['live'] == 0 and ns['byod_report']['images'] == 1


@pytest.mark.parametrize(
    ('version', 'match'), [('2.14.0+cu128', True), ('2.14.0rc1', False), ('2.14.1', False)]
)
def test_runtime_version_matching(version, match):
    ns = dict(Version=Version, InvalidVersion=InvalidVersion)
    helpers(7, {'matches_public_version'}, ns)
    assert ns['matches_public_version'](version, '2.14.0') is match
