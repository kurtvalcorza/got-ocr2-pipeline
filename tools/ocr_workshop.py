"""Stages of tutorials/DIMER_OCR_Document_Extraction_Workshop.ipynb, run in its isolated uv environment.

The notebook carries this file byte-for-byte (cell ``uvcarrier``, written by
``tools/build_ocr_workshop_carrier.py``) and runs every stage as a separate process with the
interpreter of a hash-locked CPython 3.12 virtual environment, so nothing is installed into the
notebook kernel and Run all needs no restart. The kernel only displays what a stage prints and the
files it writes.

The functions are the former notebook cells' code, moved here unchanged in behaviour. Values that
used to live in kernel variables between cells are written to ``<work dir>/state/*.json`` by the
stage that computes them and read back by the stages that need them. Images handed between stages
(Belfort test lines, rendered pages) are written as lossless PNG files.

Usage (the notebook does this)::

    python ocr_workshop.py --config stage_config.json --stage <name> [--page <page id>]
"""

from __future__ import annotations

import argparse
import csv
import difflib
import gc
import hashlib
import io
import json
import random
import re
import sys
import tempfile
import time
import traceback
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------------------------
# Section 3: configuration. Defaults equal the notebook's Configuration cell; the notebook passes
# its current values in the stage config and ``configure`` replaces these module globals.
# ---------------------------------------------------------------------------------------------
USE_BYOD = False
BYOD_PATH = ""
RUN_BELFORT = True
RUN_PAGE_SUITE = True
RUN_TOKEN_BUDGET_EXPERIMENT = True
RUN_SHAPE_ROBUSTNESS = True
GOT_LINE_MAX_NEW_TOKENS = 128
GOT_PAGE_MAX_NEW_TOKENS = 1024
SMOLDOC_LINE_MAX_NEW_TOKENS = 160
SMOLDOC_PAGE_MAX_NEW_TOKENS = 2048
BATCH_SIZE = 8
MIN_IMAGE_SIDE = 16
MAX_IMAGE_SIDE = 16384
MAX_IMAGE_PIXELS = 4096 * 4096
OUTPUT_DIR = "outputs/document_extraction"
WORK_DIR = "work/document_extraction"
ENVIRONMENT: dict = {}
RUNTIME: dict = {}

CONFIG_KEYS = (
    "USE_BYOD", "BYOD_PATH", "RUN_BELFORT", "RUN_PAGE_SUITE", "RUN_TOKEN_BUDGET_EXPERIMENT",
    "RUN_SHAPE_ROBUSTNESS", "GOT_LINE_MAX_NEW_TOKENS", "GOT_PAGE_MAX_NEW_TOKENS",
    "SMOLDOC_LINE_MAX_NEW_TOKENS", "SMOLDOC_PAGE_MAX_NEW_TOKENS", "BATCH_SIZE", "MIN_IMAGE_SIDE",
    "MAX_IMAGE_SIDE", "MAX_IMAGE_PIXELS", "OUTPUT_DIR", "WORK_DIR", "ENVIRONMENT",
)

# A line starting with this marker asks the notebook kernel to render a text block or show an image.
DISPLAY_MARKER = "\x1edimer-display "


def configure(config: dict) -> None:
    unknown = sorted(set(config) - set(CONFIG_KEYS))
    if unknown:
        raise ValueError(f"unknown stage config keys: {unknown}")
    globals().update(config)


def work_dir() -> Path:
    return Path(WORK_DIR)


def state_path(name):
    return work_dir() / "state" / name


def save_state(name, value):
    path = state_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, ensure_ascii=False), encoding="utf-8")


def load_state(name):
    path = state_path(name)
    if not path.is_file():
        raise RuntimeError(f"{path} is missing: run the earlier cells of this notebook in order first")
    return json.loads(path.read_text(encoding="utf-8"))


def int_keys(mapping):
    return {int(k): v for k, v in mapping.items()}


# ---------------------------------------------------------------------------------------------
# Section 4: runtime and display helpers.
# ---------------------------------------------------------------------------------------------
def preview(text,limit=600,full_path=None):
    text=str(text)
    if len(text)<=limit: return text
    where=f"; full text: {full_path}" if full_path else ""
    return text[:limit]+f" … [+{len(text)-limit} chars{where}]"


def show_block(title,text,limit=1200,full_path=None):
    """Hand a titled text block to the notebook, which renders it in a fenced Markdown block."""
    body=preview(text,limit,full_path)
    print(DISPLAY_MARKER+json.dumps({"kind":"block","title":title,"body":body},ensure_ascii=False),flush=True)


def display_image(path):
    """Ask the notebook to show an image file this stage wrote."""
    print(DISPLAY_MARKER+json.dumps({"kind":"image","path":str(path)},ensure_ascii=False),flush=True)


def torch_runtime():
    import importlib.metadata as importlib_metadata

    import torch
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    runtime={"python":sys.version.split()[0],"torch":torch.__version__,
     "transformers":importlib_metadata.version("transformers"),
     "numpy":np.__version__,"pyarrow":importlib_metadata.version("pyarrow"),
     "device":device,"dtype":"float32"}
    if torch.cuda.is_available(): runtime["gpu_name"]=torch.cuda.get_device_name(0)
    if ENVIRONMENT: runtime["isolated_environment"]=dict(ENVIRONMENT)
    return runtime


def stage_environment():
    runtime=torch_runtime()
    save_state("runtime.json",runtime)
    print(runtime)


def device():
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------------------------
# Section 5: immutable model snapshots.
# ---------------------------------------------------------------------------------------------
GOT_MANIFEST = {
    "format": "dimer_hf_snapshot", "formatVersion": 1, "modelKey": "got-ocr-2.0-hf",
    "modelId": "stepfun-ai/GOT-OCR-2.0-hf", "revision": "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d",
    "files": [
        {"path": "README.md", "bytes": 12077,
         "sha256": "706d6fe217d2f047ca68f47bdcf44ace4a0832491da396972c2c917944e84c18"},
        {"path": "config.json", "bytes": 608,
         "sha256": "cbe8aacd6cd84a2d58eafcd0045c6ac40e02e3a448f24b8cee51cc81d8bdccf2"},
        {"path": "generation_config.json", "bytes": 74,
         "sha256": "31915c5a692f43c5765a20cfc5f9403bcd250f5721a0d931bb703169c08993b4"},
        {"path": "model.safetensors", "bytes": 1121114488,
         "sha256": "6175ac7868a4e75735f5d59f78c465081ad3427eb4f312d072a0f1d16b333ba4"},
        {"path": "preprocessor_config.json", "bytes": 439,
         "sha256": "ef9a0dc0935cac11f4230ca30d00a52bedfa52b6633e409e9fbd2ea56373aa7e"},
        {"path": "special_tokens_map.json", "bytes": 213,
         "sha256": "7c2368a3889fdfb37c24cabeb031b53f47934f357b54e56e8e389909a338ea47"},
        {"path": "tokenizer.json", "bytes": 18702549,
         "sha256": "36b382a3c48c9a143c30139dac6c8230ddfb0b46a3dc43082af6052abe99d9de"},
        {"path": "tokenizer_config.json", "bytes": 39228,
         "sha256": "8b0542937d32a67da8ea2d1288b870e325be383a962c65d201864299560a2b8e"},
    ],
    "totalBytes": 1139869676,
}
SMOL_MANIFEST = {
    "format": "dimer_hf_snapshot", "formatVersion": 1, "modelKey": "smoldocling-256m-preview",
    "modelId": "docling-project/SmolDocling-256M-preview", "revision": "ce51f56c4ebe36e0b1c3a55f67b261ba22a50bf8",
    "files": [
        {"path": "README.md", "bytes": 16108,
         "sha256": "9b82c4dd1b38340656da55d628d63bf319c5fc4700811148687b6d8070e7e493"},
        {"path": "added_tokens.json", "bytes": 3667,
         "sha256": "fc79a032b551636ad0fe6c0e16bfe38c43b5843895cbb0544a4a5919818472cc"},
        {"path": "chat_template.json", "bytes": 430,
         "sha256": "b585e3598909a5687f9f9d738d35223724dedef256b9b274e1cbfb32b13c74bf"},
        {"path": "config.json", "bytes": 3903,
         "sha256": "57af2810c65b9896a8d1d65c67aabdb9296d497aa10a6329f4bb2ddce623586f"},
        {"path": "generation_config.json", "bytes": 141,
         "sha256": "0758109c85e7f7d6b0202ebf643bb07c5625b3363e389854b414d9a701becc28"},
        {"path": "merges.txt", "bytes": 466391,
         "sha256": "0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510"},
        {"path": "model.safetensors", "bytes": 513028808,
         "sha256": "cdcdf5d823c5684029c7d8e52177cf10f9034b3aba6577549cfb1a9ce36ad0a2"},
        {"path": "preprocessor_config.json", "bytes": 486,
         "sha256": "6cb6e36d6fcb88ca1502c4a26750715dc3e7dedddc9a8f17b27d8d167d1457e7"},
        {"path": "processor_config.json", "bytes": 68,
         "sha256": "e7bff42da73ae9eec9042ef20e066e11f1ee20f025358ff79131e3c0fb549b46"},
        {"path": "special_tokens_map.json", "bytes": 1069,
         "sha256": "aa0ff906077086dfa9734a7f97f68c825877a48f9468807be65504495cdeef09"},
        {"path": "tokenizer.json", "bytes": 3547443,
         "sha256": "7c5cf6233a3dc8b9e54fb729ee6e771bdf5f0d65fd9075b5e60bed837959deee"},
        {"path": "tokenizer_config.json", "bytes": 27362,
         "sha256": "b38f39506a4fa7d2604015a6c303df320675cfeb7ba1f5969e1c44032727107b"},
        {"path": "vocab.json", "bytes": 800662,
         "sha256": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79"},
    ],
    "totalBytes": 517896538,
}
SNAPSHOT_ROOT = Path("weights/document-extraction")


def sha256_file(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def stage_snapshot(manifest):
    from huggingface_hub import hf_hub_download
    root=SNAPSHOT_ROOT/manifest["modelKey"]; root.mkdir(parents=True,exist_ok=True)
    (root/"dimer-base-manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    t=time.perf_counter()
    for e in manifest["files"]:
        p=root/e["path"]
        if not p.is_file():
            p.parent.mkdir(parents=True,exist_ok=True)
            hf_hub_download(repo_id=manifest["modelId"],filename=e["path"],
                            revision=manifest["revision"],local_dir=str(root))
    for e in manifest["files"]:
        p=root/e["path"]
        if p.stat().st_size!=e["bytes"] or sha256_file(p)!=e["sha256"]:
            raise RuntimeError(f"snapshot verification failed: {manifest['modelKey']}/{e['path']}")
    return root,time.perf_counter()-t


def stage_snapshots():
    got_dir,got_verify_seconds=stage_snapshot(GOT_MANIFEST)
    smol_dir,smol_verify_seconds=stage_snapshot(SMOL_MANIFEST)
    save_state("snapshots.json",{"got":{"dir":str(got_dir),"verify_seconds":got_verify_seconds},
                                 "smoldocling":{"dir":str(smol_dir),"verify_seconds":smol_verify_seconds}})
    for manifest in (GOT_MANIFEST,SMOL_MANIFEST):
        print(f"{manifest['modelId']} @ {manifest['revision'][:12]}: {len(manifest['files'])} files, "
              f"{manifest['totalBytes']:,} bytes")
    print("Snapshots verified.")


def snapshot_dir(key):
    return Path(load_state("snapshots.json")[key]["dir"])


# ---------------------------------------------------------------------------------------------
# Section 6: Belfort-line provenance.
# ---------------------------------------------------------------------------------------------
ROW_GROUP_PINS = {
    0: ("1dc3141e4809ea628b17c3ca7b81d64e6ca92bce18dd5765ecd618bfc7867954", 5481145),
    1: ("c9d3b52013933c803f4886edbce68da0ae483347a4a6ff3e0f5ced1db4a7e653", 5465901),
    2: ("6e0578a90a07a9e25e65b765881d3fa33d6a797624425e01026980d7287f0bf6", 5379166),
    3: ("00cdfb7aabe924f31b9f1bb1ba4849040051e5619567b68bf99fdcbcab15131a", 5821303),
    4: ("fc063442fb20e7a60c2533ab44dcc69a22ad59f5ce24921fe6af5f53ceab7e1a", 5163559),
    5: ("2d7e29331bd4e93e0c8a1caa9a83b8f1ede9b17af6dae9377b83f56c56f05689", 4713140),
    6: ("49423d91780cb184c4b0069630e85acccb314a113256ced9692a5138c4782ef1", 5069794),
    7: ("3a010831456f185399579b4ecf9d46222f95368c3cdbbc2ff103a258b9c16e2f", 5309891),
}
CORPUS_REPO="Teklia/Belfort-line"
CORPUS_REVISION="c4a74bbd39f2df314752e7e6026649a39d365cbb"
CORPUS_FILE="default/test/0000.parquet"
CORPUS_BYTES=210_579_166
CORPUS_ROWS=3_819
CORPUS_URL=f"https://huggingface.co/datasets/{CORPUS_REPO}/resolve/{CORPUS_REVISION}/{CORPUS_FILE}"
BELFORT_CACHE=Path("weights/belfort")
SAMPLE_SEED=42
SAMPLE_SPLIT={"train":600,"validation":60,"test":140}
SAMPLE_DIGEST="b7e1dd684691a0eedb63a609311f4964e7732e5c1a8d254fe4e1293a8cd0964d"


def normalise_text(text): return " ".join(str(text).split())


class HttpRangeFile(io.RawIOBase):
    def __init__(self,url,size): self.url,self.size,self.pos=url,size,0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=max(0,{0:0,1:self.pos,2:self.size}[whence]+offset); return self.pos
    def read(self,n=-1):
        if n is None or n<0: n=self.size-self.pos
        if n<=0 or self.pos>=self.size: return b""
        end=min(self.size,self.pos+n)-1
        req=urllib.request.Request(self.url,headers={"Range":f"bytes={self.pos}-{end}",
                                                     "User-Agent":"dimer-doc-extract/1.0"})
        with urllib.request.urlopen(req,timeout=300) as response:
            if response.status!=206: raise ValueError(f"server ignored Range request: {response.status}")
            data=response.read()
        self.pos+=len(data); return data
    def readinto(self,buffer):
        data=self.read(len(buffer)); buffer[:len(data)]=data; return len(data)


def declared_size(url):
    req=urllib.request.Request(url,method="HEAD",headers={"User-Agent":"dimer-doc-extract/1.0"})
    with urllib.request.urlopen(req,timeout=60) as response: value=response.headers.get("Content-Length")
    if value is None: raise ValueError("missing Content-Length")
    return int(value)


def group_digest(rows):
    h=hashlib.sha256(); total=0
    for row in rows:
        b=row["image"]["bytes"]; t=str(row["text"]).encode("utf-8")
        h.update(b); h.update(t); total+=len(b)+len(t)
    return h.hexdigest(),total


def fetch_groups():
    import pyarrow.parquet as pq
    BELFORT_CACHE.mkdir(parents=True,exist_ok=True)
    if declared_size(CORPUS_URL)!=CORPUS_BYTES: raise RuntimeError("Belfort shard size drift")
    reader=pq.ParquetFile(HttpRangeFile(CORPUS_URL,CORPUS_BYTES))
    if reader.metadata.num_rows!=CORPUS_ROWS: raise RuntimeError("Belfort row-count drift")
    out={}
    for group in sorted(ROW_GROUP_PINS):
        local=BELFORT_CACHE/f"test-rg{group}.parquet"; rows=None
        if local.is_file():
            candidate=pq.read_table(local).to_pylist()
            if group_digest(candidate)==ROW_GROUP_PINS[group]: rows=candidate
        if rows is None:
            table=reader.read_row_group(group,columns=["image","text"])
            rows=table.to_pylist()
            if group_digest(rows)!=ROW_GROUP_PINS[group]: raise RuntimeError(f"row group {group} digest drift")
            pq.write_table(table,local)
        out[group]=rows
    return out


def image_digest(image):
    rgb=image.convert("RGB")
    return hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode()+rgb.tobytes()).hexdigest()


def read_corpus(groups):
    out=[]
    for group in sorted(groups):
        for idx,row in enumerate(groups[group]):
            text=normalise_text(row["text"])
            if not text: continue
            image=Image.open(io.BytesIO(row["image"]["bytes"])); image.load()
            out.append({"id":f"belfort-test-{group*100+idx}","image":image.convert("RGB"),
                         "text":text,"source_row_group":group})
    return out


def dataset_digest(records):
    parts=sorted(f"{r['id']}:{image_digest(r['image'])}:{normalise_text(r['text'])}" for r in records)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def build_split(records):
    pool=[dict(r) for r in records]; random.Random(SAMPLE_SEED).shuffle(pool)
    out={}; cursor=0
    for name,count in SAMPLE_SPLIT.items():
        out[name]=pool[cursor:cursor+count]; cursor+=count
    return out


def belfort_image_dir():
    return work_dir()/"belfort"/"test"


def stage_belfort():
    belfort_splits=build_split(read_corpus(fetch_groups()))
    combined=[r for part in belfort_splits.values() for r in part]
    if dataset_digest(combined)!=SAMPLE_DIGEST: raise RuntimeError("Belfort sample digest mismatch")
    seen={}
    for name,part in belfort_splits.items():
        for r in part:
            d=image_digest(r["image"])
            if d in seen and seen[d]!=name: raise RuntimeError("cross-split image leakage")
            seen[d]=name
    # The test lines are handed to the model stages as lossless PNG files (identical RGB pixels).
    image_dir=belfort_image_dir(); image_dir.mkdir(parents=True,exist_ok=True)
    for r in belfort_splits["test"]:
        r["image"].save(image_dir/f"{r['id']}.png")
    save_state("belfort.json",{
        "digest":dataset_digest(combined),
        "image_digests":{r["id"]:image_digest(r["image"]) for r in belfort_splits["test"]},
        "splits":{name:[{"id":r["id"],"text":r["text"],"source_row_group":r["source_row_group"]} for r in part]
                  for name,part in belfort_splits.items()}})
    print({"split_sizes":{k:len(v) for k,v in belfort_splits.items()},"digest":dataset_digest(combined)})


def load_belfort_split(name,images=False):
    state=load_state("belfort.json"); records=[dict(r) for r in state["splits"][name]]
    if images:
        for r in records:
            with Image.open(belfort_image_dir()/f"{r['id']}.png") as decoded:
                r["image"]=decoded.convert("RGB")
            if image_digest(r["image"])!=state["image_digests"][r["id"]]:
                raise RuntimeError(f"{r['id']}: staged line image differs from the verified sample")
    return records


# ---------------------------------------------------------------------------------------------
# Section 7: common OCR metrics and baselines.
# ---------------------------------------------------------------------------------------------
def edit_distance(reference,hypothesis):
    previous=list(range(len(hypothesis)+1))
    for i,a in enumerate(reference,1):
        current=[i]
        for j,b in enumerate(hypothesis,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1]


def words(text): return normalise_text(text).lower().split()


def one_metrics(reference,hypothesis):
    ref=normalise_text(reference); hyp=normalise_text(hypothesis)
    ce=edit_distance(ref,hyp); rw,hw=words(ref),words(hyp); we=edit_distance(rw,hw)
    return {"cer":ce/len(ref),"wer":we/len(rw),"exact_match":hyp==ref,
            "reference_chars":len(ref),"hypothesis_chars":len(hyp),
            "length_ratio":len(hyp)/len(ref),"char_edits":ce,"word_edits":we,
            "reference_words":len(rw)}


def word_diff(reference,hypothesis,limit=10):
    ref,hyp=normalise_text(reference).split(),normalise_text(hypothesis).split()
    out=[]
    for op,i1,i2,j1,j2 in difflib.SequenceMatcher(a=ref,b=hyp,autojunk=False).get_opcodes():
        if op!="equal": out.append({"op":op,"reference":" ".join(ref[i1:i2]),"model":" ".join(hyp[j1:j2])})
    return out[:limit]


def corpus_metrics(hypotheses,records):
    rows=[{"id":r["id"],**one_metrics(r["text"],h)}
          for h,r in zip(hypotheses,records,strict=True)]
    return {"n":len(rows),
            "cer":sum(x["char_edits"] for x in rows)/sum(x["reference_chars"] for x in rows),
            "wer":sum(x["word_edits"] for x in rows)/sum(x["reference_words"] for x in rows),
            "cer_macro":float(np.mean([x["cer"] for x in rows])),
            "wer_macro":float(np.mean([x["wer"] for x in rows])),
            "exact_match":float(np.mean([x["exact_match"] for x in rows])),
            "median_cer":float(np.median([x["cer"] for x in rows])),
            "p90_cer":float(np.quantile([x["cer"] for x in rows],0.9)),
            "ref_chars":sum(x["reference_chars"] for x in rows),
            "hyp_chars":sum(x["hypothesis_chars"] for x in rows),
            "length_ratio":sum(x["hypothesis_chars"] for x in rows)/sum(x["reference_chars"] for x in rows),
            "rows":rows}


def medoid_transcript(train,pool=120):
    texts=[normalise_text(r["text"]) for r in train[:pool] if normalise_text(r["text"])]
    return min(texts,key=lambda c:sum(edit_distance(other,c)/len(other) for other in texts))


def stage_baselines():
    test_records=load_belfort_split("test")
    empty_metrics=corpus_metrics([""]*len(test_records),test_records)
    constant_text=medoid_transcript(load_belfort_split("train"))
    constant_metrics=corpus_metrics([constant_text]*len(test_records),test_records)
    save_state("baselines.json",{"empty_metrics":empty_metrics,"constant_metrics":constant_metrics,
                                 "constant_text":constant_text})
    print({"empty_cer":empty_metrics["cer"],"constant_cer":constant_metrics["cer"],"constant":constant_text})


# ---------------------------------------------------------------------------------------------
# Section 8: deterministic rendered pages.
# ---------------------------------------------------------------------------------------------
REFERENCE_POLICY=("Reference = every visible string in drawing order: top to bottom, left to right, tables row "
                  "by row (header first). Punctuation is kept as drawn; bullets are shapes, not text; whitespace "
                  "is collapsed. Expected counts cover only the listed (scored) element types.")


def page_dir():
    return Path(OUTPUT_DIR)/"pages"


def fnt(size): return ImageFont.load_default(size=size)


def missing_glyphs(text,size):
    font=fnt(size); box=font.getmask("\ue000"); box=(box.size,bytes(box))
    return sorted({ch for ch in str(text) if not ch.isspace()
                   and (font.getmask(ch).size,bytes(font.getmask(ch)))==box})


def wrapped(text,n):
    out=[]; line=[]
    for w in text.split():
        if len(" ".join(line+[w]))>n and line: out.append(" ".join(line)); line=[w]
        else: line.append(w)
    if line: out.append(" ".join(line))
    return out


class PageText:
    """Draws page text and records each visible string in the same call, in drawing order."""
    def __init__(self,image): self.draw=ImageDraw.Draw(image); self.parts=[]
    def text(self,xy,value,size,bullet=False):
        missing=missing_glyphs(value,size)
        if missing: raise ValueError(f"default font cannot draw {missing} in {value!r}")
        x,y=xy
        if bullet:
            r=max(2,size//6); cy=y+int(size*0.6)
            self.draw.ellipse((x+2,cy-r,x+2+2*r,cy+r),fill="black"); x+=size
        self.draw.text((x,y),str(value),fill="black",font=fnt(size)); self.parts.append(str(value))
    def reference(self): return normalise_text(" ".join(self.parts))


def block(page,text,x,y,width,size=22):
    for line in wrapped(text,width):
        page.text((x,y),line,size); y+=size+8
    return y


def table(page,x,y,widths,rows,row_h=40,size=18):
    draw=page.draw; W=sum(widths); H=len(rows)*row_h
    draw.rectangle((x,y,x+W,y+H),outline="black",width=2)
    xx=x
    for w in widths[:-1]: xx+=w; draw.line((xx,y,xx,y+H),fill="black")
    for i in range(1,len(rows)): draw.line((x,y+i*row_h,x+W,y+i*row_h),fill="black")
    for i,row in enumerate(rows):
        xx=x
        for j,val in enumerate(row):
            page.text((xx+5,y+i*row_h+8),val,size); xx+=widths[j]
    return y+H


def notice():
    im=Image.new("RGB",(1000,1300),"white"); pg=PageText(im)
    pg.text((70,50),"PUBLIC NOTICE",42)
    pg.text((70,115),"Document Processing Workshop - 26 September 2026",21)
    ps=["The records office will conduct a scheduled document digitization activity this Friday.",
        "Participants should bring one sample page and verify all extracted text against the original document.",
        "Generated OCR output may contain omissions, substitutions, repetitions, or invented text and must be reviewed."]
    y=200
    for p in ps: y=block(pg,p,70,y,72,23)+20
    bullets=["Prepare the source document.","Check the recognized text.","Review the exported structure."]
    for b in bullets: pg.text((95,y),b,23,bullet=True); y+=42
    pg.text((70,1200),"DIMER Document Intelligence Workshop",20)
    return im,pg.reference(),{"section_header_level_1":1,"text":4,"list_item":3,"page_footer":1}


def report():
    im=Image.new("RGB",(1100,1500),"white"); pg=PageText(im)
    pg.text((70,45),"DOCUMENT EXTRACTION REPORT",38)
    p1="This report summarizes a controlled document conversion exercise using synthetic text and deterministic tables."
    p2="The page is generated locally so every visible word and table cell has a known reference."
    y=120; y=block(pg,p1,70,y,82,22)+18; y=block(pg,p2,70,y,82,22)+30
    r1=[["Item","Q1","Q2","Q3","Total"]]+[[f"Series {i}",i,i+1,i+2,3*i+3] for i in range(1,8)]
    y=table(pg,70,y,[210,130,130,130,150],r1,38)+40
    r2=[["Code","Count","Status"]]+[[f"A{i}",i*3,"Ready" if i%2 else "Review"] for i in range(1,5)]
    y=table(pg,70,y,[240,170,300],r2,42)+35
    block(pg,"End of report. Verify extracted text and structural markup before downstream use.",70,y,82,22)
    return im,pg.reference(),{"section_header_level_1":1,"text":3,"otsl":2}


def invoice():
    im=Image.new("RGB",(1000,1350),"white"); pg=PageText(im)
    pg.text((70,45),"INVOICE",44)
    fields=[("Invoice Number","INV-2026-0926"),("Date","26 September 2026"),
            ("Customer","Sample Research Office"),("Address","C.P. Garcia Avenue, Diliman, Quezon City")]
    y=130
    for k,v in fields: pg.text((70,y),f"{k}: {v}",21); y+=42
    rows=[["Description","Qty","Unit Price","Amount"],["Document scan",3,"120.00","360.00"],
          ["OCR review",2,"250.00","500.00"],["Structure check",1,"400.00","400.00"]]
    y=table(pg,70,y+20,[360,100,170,170],rows,44)+35
    totals=[("Subtotal","1,260.00"),("Tax","151.20"),("Total","1,411.20")]
    for k,v in totals: pg.text((560,y),f"{k}: {v}",23); y+=42
    block(pg,"Payment note: This synthetic invoice is for OCR and document extraction testing only.",70,y+30,74,21)
    return im,pg.reference(),{"section_header_level_1":1,"text":8,"otsl":1,"key_value_region":1}


def technical():
    im=Image.new("RGB",(1050,1300),"white"); pg=PageText(im)
    pg.text((70,45),"TECHNICAL NOTE",40)
    intro="The following expressions are rendered as plain text for deterministic extraction testing."
    y=block(pg,intro,70,125,76,22)+35
    pg.text((100,y),"Energy: E = m * c^2",29); y+=70
    pg.text((100,y),"Mean: x_bar = (x1 + x2 + x3) / 3",29); y+=85
    y=block(pg,"Measured value: 12.5 kJ. Tolerance: plus or minus 0.2 kJ.",70,y,76,22)+35
    block(pg,"The formulas above are instructional content, not production measurements.",70,y,76,22)
    return im,pg.reference(),{"section_header_level_1":1,"text":3,"formula":2}


builders={"notice":("Notice",notice),"report":("Report with tables",report),
          "invoice":("Invoice-style document",invoice),"technical_note":("Formula / technical note",technical)}


def render_pages():
    page_root=page_dir(); page_root.mkdir(parents=True,exist_ok=True)
    pages={}
    for pid,(kind,builder) in builders.items():
        image,reference,expected=builder()
        path=page_root/f"{pid}.png"; image.save(path)
        pages[pid]={"id":pid,"kind":kind,"image":image,"reference_text":reference,
                    "expected_counts":expected,"path":str(path)}
    return pages


def stage_pages():
    pages=render_pages()
    save_state("pages.json",{pid:{k:v for k,v in page.items() if k!="image"} for pid,page in pages.items()})
    print({k:v["image"].size for k,v in pages.items()})
    print(REFERENCE_POLICY)
    sheet=Image.new("RGB",(4*260,370),"white")
    for i,page in enumerate(pages.values()):
        thumb=page["image"].copy(); thumb.thumbnail((250,360)); sheet.paste(thumb,(i*260+5,5))
    sheet_path=work_dir()/"pages_contact_sheet.png"; sheet_path.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(sheet_path); display_image(sheet_path)
    for pid,page in pages.items(): print(f"{pid}: {preview(page['reference_text'],160)}")


def load_pages():
    pages=load_state("pages.json")
    for page in pages.values():
        with Image.open(page["path"]) as decoded: page["image"]=decoded.convert("RGB")
    return pages


# ---------------------------------------------------------------------------------------------
# Section 9: native structure and supplemental text helpers.
# ---------------------------------------------------------------------------------------------
ELEMENT_TAGS=("section_header_level_1","section_header_level_2","section_header_level_3",
"text","paragraph","list_item","ordered_list","unordered_list","otsl","picture","caption",
"formula","code","page_header","page_footer","footnote","chart","key_value_region")
TAG_RE=re.compile(r"</?([a-z_]+(?:_[0-9]+)?)>")
LOC_RE=re.compile(r"<loc_([0-9]+)>")
OTSL_RE=re.compile(r"<(?:fcel|ecel|ched|rhed|srow|lcel|ucel|xcel|nl)>")
OTSL_TYPES=("fcel","ecel","ched","rhed","srow","lcel","ucel","xcel","nl")
OTSL_CELL_START=("fcel","ecel","ched","rhed","srow")
OTSL_MERGE_CONTINUATION=("lcel","ucel","xcel")


def doctags_to_text(x):
    x=LOC_RE.sub(" ",x); x=OTSL_RE.sub(" ",x); x=TAG_RE.sub(" ",x)
    return normalise_text(x)


def doctags_summary(x):
    opened=Counter(m.group(1) for m in TAG_RE.finditer(x) if not m.group(0).startswith("</"))
    locs=[int(v) for v in LOC_RE.findall(x)]
    if any(v<0 or v>500 for v in locs): raise RuntimeError("loc token outside 0..500")
    otsl={t:len(re.findall(fr"<{t}>",x)) for t in OTSL_TYPES}
    return {"counts":{t:opened.get(t,0) for t in ELEMENT_TAGS},
            "n_elements":sum(opened.get(t,0) for t in ELEMENT_TAGS),
            "n_loc_tokens":len(locs),"n_location_groups":len(locs)//4,
            "loc_tokens_divisible_by_4":len(locs)%4==0,
            "n_table_cells":sum(otsl[t] for t in OTSL_CELL_START),
            "n_table_rows":otsl["nl"],
            "n_table_grid_positions":sum(otsl[t] for t in OTSL_CELL_START+OTSL_MERGE_CONTINUATION),
            "n_otsl_tokens":sum(otsl.values()),
            "otsl_token_counts":otsl,
            "wrapped_in_doctag":x.lstrip().startswith("<doctag>") and x.rstrip().endswith("</doctag>"),
            "n_chars":len(x)}


def structure_rows(page_id,expected_counts,summary):
    rows=[]
    for tag in sorted(set(expected_counts)|{t for t,n in summary["counts"].items() if n}):
        observed=int(summary["counts"].get(tag,0)); scored=tag in expected_counts
        expected=int(expected_counts[tag]) if scored else None
        rows.append({"page_id":page_id,"element_type":tag,"scored":scored,"expected_count":expected,
                     "observed_count":observed,"absolute_count_error":abs(expected-observed) if scored else None})
    return rows


def token_overlap(reference,hypothesis):
    a,b=Counter(words(reference)),Counter(words(hypothesis))
    overlap=sum((a&b).values())
    p=overlap/sum(b.values()) if b else 0.0
    r=overlap/sum(a.values()) if a else 0.0
    f=2*p*r/(p+r) if p+r else 0.0
    return p,r,f


def page_row(model,page,item,text):
    m=one_metrics(page["reference_text"],text); p,r,f=token_overlap(page["reference_text"],text)
    return {"page_id":page["id"],"page_kind":page["kind"],"model":model,
            "cer":m["cer"],"wer":m["wer"],"token_precision":p,"token_recall":r,"token_f1":f,
            "reference_chars":m["reference_chars"],"hypothesis_chars":m["hypothesis_chars"],
            "length_ratio":m["length_ratio"],"new_tokens":item["new_tokens"],
            "truncated":item["truncated"],"inference_seconds":item.get("inference_seconds")}


def error_category(cer,ratio,empty,truncated):
    if truncated:return "truncated"
    if empty:return "empty"
    if ratio>1.5:return "runaway_or_hallucinated"
    if cer==0:return "exact"
    if cer<0.2:return "mostly_correct"
    if cer<0.8:return "partial"
    return "severe"


# ---------------------------------------------------------------------------------------------
# Section 10: GOT-OCR frozen inference.
# ---------------------------------------------------------------------------------------------
GOT_STOP="<|im_end|>"
GOT_MAX_NEW_TOKENS=4096


def load_got():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    got_dir=snapshot_dir("got"); DEVICE=device(); DTYPE=torch.float32
    t=time.perf_counter()
    p=AutoProcessor.from_pretrained(str(got_dir),local_files_only=True,trust_remote_code=False)
    m=AutoModelForImageTextToText.from_pretrained(
        str(got_dir),local_files_only=True,trust_remote_code=False,dtype=DTYPE).eval().to(DEVICE)
    for x in m.parameters(): x.requires_grad_(False)
    pad=p.tokenizer.pad_token_id; stop=p.tokenizer.convert_tokens_to_ids(GOT_STOP)
    original=m.model.get_image_features
    def chunked(pixel_values,**kwargs):
        if pixel_values.shape[0]<=1:return original(pixel_values=pixel_values,**kwargs)
        return torch.cat([original(pixel_values=pixel_values[i:i+1],**kwargs)
                          for i in range(pixel_values.shape[0])],dim=0)
    m.model.get_image_features=chunked
    return m,p,pad,stop,time.perf_counter()-t


def got_generate(model,processor,pad_id,stop_id,images,mode,budget):
    import torch
    if mode not in ("plain","format"): raise ValueError(mode)
    if not 1<=budget<=GOT_MAX_NEW_TOKENS: raise ValueError(budget)
    inputs=processor(list(images),return_tensors="pt",padding=True,format=(mode=="format"))
    if not bool(inputs["attention_mask"].all()): raise RuntimeError("GOT batch prompts differ in length")
    inputs=inputs.to(device())
    if torch.cuda.is_available():torch.cuda.synchronize()
    t=time.perf_counter()
    with torch.inference_mode():
        generated=model.generate(**inputs,do_sample=False,tokenizer=processor.tokenizer,
                                 stop_strings=GOT_STOP,max_new_tokens=budget)
    if torch.cuda.is_available():torch.cuda.synchronize()
    elapsed=time.perf_counter()-t; prompt_len=int(inputs["input_ids"].shape[1]); out=[]
    for row in generated:
        ids=row[prompt_len:].tolist(); n=len(ids)
        for pos,token in enumerate(ids):
            if token in (stop_id,pad_id):
                n=pos+(token==stop_id); break
        text=processor.decode(row[prompt_len:prompt_len+n],skip_special_tokens=True)
        out.append({"text":str(text).replace(GOT_STOP,"").strip(),"mode":mode,"new_tokens":int(n),
                    "truncated":n>=budget,"inference_seconds":elapsed/len(images)})
    return out


def stage_got_load():
    got_model,_,_,_,got_load_seconds=load_got()
    got_parameter_count=sum(p.numel() for p in got_model.parameters())
    save_state("got_load.json",{"load_seconds":got_load_seconds,"parameters":got_parameter_count})
    print({"load_seconds":got_load_seconds,"parameters":got_parameter_count})


# Section 11: GOT-OCR on held-out Belfort lines.
def stage_got_belfort():
    import torch
    got_belfort_items=[]; got_belfort_metrics=None; got_belfort_seconds=0.0
    if RUN_BELFORT:
        test_records=load_belfort_split("test",images=True)
        got_model,got_processor,got_pad,got_stop,_=load_got()
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        t=time.perf_counter()
        for start in range(0,len(test_records),BATCH_SIZE):
            batch=test_records[start:start+BATCH_SIZE]
            got_belfort_items.extend(got_generate(got_model,got_processor,got_pad,got_stop,
                                                   [r["image"] for r in batch],"plain",GOT_LINE_MAX_NEW_TOKENS))
            if (start+BATCH_SIZE)%40==0: print("GOT",min(start+BATCH_SIZE,len(test_records)),"/",len(test_records))
        got_belfort_seconds=time.perf_counter()-t
        got_belfort_metrics=corpus_metrics([x["text"] for x in got_belfort_items],test_records)
        got_belfort_metrics["truncated"]=sum(x["truncated"] for x in got_belfort_items)
        got_belfort_metrics["empty_hypotheses"]=sum(not normalise_text(x["text"]) for x in got_belfort_items)
        got_peak_gpu=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        print({k:got_belfort_metrics[k] for k in ("cer","wer","cer_macro","exact_match","length_ratio","truncated")})
    else:
        got_peak_gpu=None
    save_state("got_belfort.json",{"items":got_belfort_items,"metrics":got_belfort_metrics,
                                   "seconds":got_belfort_seconds,"peak_gpu":got_peak_gpu})


# Section 12: GOT-OCR rendered-page capabilities.
def stage_got_pages():
    got_page_plain={}; got_page_rows=[]; got_format_outputs={}
    if RUN_PAGE_SUITE:
        pages=load_pages(); PAGE_DIR=page_dir()
        got_model,got_processor,got_pad,got_stop,_=load_got()
        for pid,page in pages.items():
            item=got_generate(got_model,got_processor,got_pad,got_stop,[page["image"]],"plain",GOT_PAGE_MAX_NEW_TOKENS)[0]
            got_page_plain[pid]=item; got_page_rows.append(page_row("GOT-OCR 2.0",page,item,item["text"]))
        for pid in ("report","invoice","technical_note"):
            item=got_generate(got_model,got_processor,got_pad,got_stop,[pages[pid]["image"]],"format",GOT_PAGE_MAX_NEW_TOKENS)[0]
            text=item["text"]
            got_format_outputs[pid]={**item,"characters":len(text),"lines":len(text.splitlines()),
                                     "format_character_counts":{c:text.count(c) for c in ("|","#","$","\\","_","*")}}
        for r in got_page_rows:
            print(r["page_id"],round(r["cer"],3),round(r["wer"],3),"|",preview(got_page_plain[r["page_id"]]["text"],160))
        for pid,item in got_format_outputs.items():
            show_block(f"GOT format · {pid} ({item['new_tokens']} tokens, truncated={item['truncated']})",
                       item["text"],800,PAGE_DIR/pid/"got_format.txt")
    save_state("got_pages.json",{"plain":got_page_plain,"rows":got_page_rows,"format_outputs":got_format_outputs})


def probe_images():
    """The blank page and the seeded noise image used by both models' hallucination probes."""
    blank=Image.new("RGB",(900,1200),"white")
    rng=np.random.default_rng(42)
    noise=Image.fromarray(rng.integers(0,256,size=(600,800,3),dtype=np.uint8),"RGB")
    return blank,noise


def shape_variants(base):
    return {"portrait_original":base,
            "wide":base.resize((1600,700),Image.Resampling.BILINEAR),
            "tall":base.resize((650,1800),Image.Resampling.BILINEAR),
            "low_resolution":base.resize((base.width//2,base.height//2),Image.Resampling.BILINEAR).resize(base.size,Image.Resampling.BILINEAR)}


# Section 13: GOT token-budget, shape and hallucination probes.
def stage_got_probes():
    got_budget_rows=[]; got_shape_rows=[]; got_blank_noise={}
    pages=load_pages()
    got_model,got_processor,got_pad,got_stop,_=load_got()
    if RUN_TOKEN_BUDGET_EXPERIMENT:
        page=pages["report"]
        for budget in (256,512,1024,2048):
            item=got_generate(got_model,got_processor,got_pad,got_stop,[page["image"]],"plain",budget)[0]
            m=one_metrics(page["reference_text"],item["text"])
            got_budget_rows.append({"model":"GOT-OCR 2.0","page_id":"report","token_budget":budget,
              "generated_tokens":item["new_tokens"],"truncated":item["truncated"],"cer":m["cer"],"wer":m["wer"],
              "text_chars":len(normalise_text(item["text"])),"n_elements":None,"n_table_cells":None})
    if RUN_SHAPE_ROBUSTNESS:
        for name,image in shape_variants(pages["notice"]["image"]).items():
            item=got_generate(got_model,got_processor,got_pad,got_stop,[image],"plain",GOT_PAGE_MAX_NEW_TOKENS)[0]
            m=one_metrics(pages["notice"]["reference_text"],item["text"])
            got_shape_rows.append({"model":"GOT-OCR 2.0","variant":name,"cer":m["cer"],"wer":m["wer"],
                                   "length_ratio":m["length_ratio"],"new_tokens":item["new_tokens"],
                                   "truncated":item["truncated"]})
    blank,noise=probe_images()
    for name,image in (("blank",blank),("noise",noise)):
        item=got_generate(got_model,got_processor,got_pad,got_stop,[image],"plain",256)[0]
        got_blank_noise[name]={"text":item["text"],"characters":len(item["text"]),
                               "new_tokens":item["new_tokens"],"truncated":item["truncated"]}
    save_state("got_probes.json",{"budget_rows":got_budget_rows,"shape_rows":got_shape_rows,
                                  "blank_noise":got_blank_noise})
    print({k:{x:v[x] for x in ("characters","new_tokens","truncated")} for k,v in got_blank_noise.items()})


# ---------------------------------------------------------------------------------------------
# Section 15: SmolDocling frozen inference.
# ---------------------------------------------------------------------------------------------
SMOL_INSTRUCTIONS=("Convert this page to docling.","Convert chart to table.","Convert formula to LaTeX.",
"Convert code to text.","Convert table to OTSL.","Find all 'text' elements on the page, retrieve all section headers.",
"Detect footer elements on the page.")
SMOL_DEFAULT=SMOL_INSTRUCTIONS[0]
SMOL_MAX_NEW_TOKENS=8192
SMOL_TERMINATORS=("<end_of_utterance>","<|im_end|>")


def messages(instruction):
    return [{"role":"user","content":[{"type":"image"},{"type":"text","text":instruction}]}]


def load_smoldoc():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    smol_dir=snapshot_dir("smoldocling"); DEVICE=device(); DTYPE=torch.float32
    t=time.perf_counter()
    p=AutoProcessor.from_pretrained(str(smol_dir),local_files_only=True,trust_remote_code=False)
    m=AutoModelForImageTextToText.from_pretrained(
        str(smol_dir),local_files_only=True,trust_remote_code=False,dtype=DTYPE).eval().to(DEVICE)
    for x in m.parameters():x.requires_grad_(False)
    return m,p,p.tokenizer.convert_tokens_to_ids("<end_of_utterance>"),p.tokenizer.pad_token_id,time.perf_counter()-t


def smol_generate(model,processor,end_id,pad_id,images,instruction,budget):
    import torch
    if instruction not in SMOL_INSTRUCTIONS:raise ValueError(instruction)
    if not 1<=budget<=SMOL_MAX_NEW_TOKENS:raise ValueError(budget)
    prompt=processor.apply_chat_template(messages(instruction),add_generation_prompt=True)
    processor.tokenizer.padding_side="left"
    inputs=processor(text=[prompt]*len(images),images=[[im.convert("RGB")] for im in images],
                     return_tensors="pt",padding=True).to(device())
    if torch.cuda.is_available():torch.cuda.synchronize()
    t=time.perf_counter()
    with torch.inference_mode(): generated=model.generate(**inputs,max_new_tokens=budget,do_sample=False)
    if torch.cuda.is_available():torch.cuda.synchronize()
    elapsed=time.perf_counter()-t; prompt_len=int(inputs["input_ids"].shape[1]); out=[]
    for row in generated[:,prompt_len:]:
        ids=row.tolist(); n=len(ids)
        for pos,token in enumerate(ids):
            if token in (end_id,pad_id):
                n=pos+(token==end_id);break
        raw=processor.tokenizer.decode(ids[:n],skip_special_tokens=False)
        for term in SMOL_TERMINATORS:raw=raw.replace(term,"")
        raw=raw.strip()
        out.append({"doctags":raw,"text":doctags_to_text(raw),"instruction":instruction,
                    "new_tokens":int(n),"truncated":n>=budget,"inference_seconds":elapsed/len(images)})
    return out


def stage_smol_load():
    smol_model,_,_,_,smol_load_seconds=load_smoldoc()
    smol_parameter_count=sum(p.numel() for p in smol_model.parameters())
    save_state("smol_load.json",{"load_seconds":smol_load_seconds,"parameters":smol_parameter_count})
    print({"load_seconds":smol_load_seconds,"parameters":smol_parameter_count})


# Section 16: SmolDocling on held-out Belfort lines.
def stage_smol_belfort():
    import torch
    smol_belfort_items=[]; smol_belfort_metrics=None; smol_belfort_seconds=0.0
    if RUN_BELFORT:
        test_records=load_belfort_split("test",images=True)
        smol_model,smol_processor,smol_end,smol_pad,_=load_smoldoc()
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        t=time.perf_counter()
        for start in range(0,len(test_records),BATCH_SIZE):
            batch=test_records[start:start+BATCH_SIZE]
            smol_belfort_items.extend(smol_generate(smol_model,smol_processor,smol_end,smol_pad,
              [r["image"] for r in batch],SMOL_DEFAULT,SMOLDOC_LINE_MAX_NEW_TOKENS))
            if (start+BATCH_SIZE)%40==0:print("SmolDocling",min(start+BATCH_SIZE,len(test_records)),"/",len(test_records))
        smol_belfort_seconds=time.perf_counter()-t
        smol_belfort_metrics=corpus_metrics([x["text"] for x in smol_belfort_items],test_records)
        smol_belfort_metrics["truncated"]=sum(x["truncated"] for x in smol_belfort_items)
        smol_belfort_metrics["empty_hypotheses"]=sum(not normalise_text(x["text"]) for x in smol_belfort_items)
        smol_peak_gpu=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
        print({k:smol_belfort_metrics[k] for k in ("cer","wer","cer_macro","exact_match","length_ratio","truncated")})
    else:
        smol_peak_gpu=None
    save_state("smol_belfort.json",{"items":smol_belfort_items,"metrics":smol_belfort_metrics,
                                    "seconds":smol_belfort_seconds,"peak_gpu":smol_peak_gpu})


# Section 17: SmolDocling rendered-page text and structure.
def stage_smol_pages():
    smol_page_results={}; smol_page_rows=[]; smol_structure_rows=[]; smol_summaries={}
    if RUN_PAGE_SUITE:
        pages=load_pages(); PAGE_DIR=page_dir()
        smol_model,smol_processor,smol_end,smol_pad,_=load_smoldoc()
        for pid,page in pages.items():
            item=smol_generate(smol_model,smol_processor,smol_end,smol_pad,[page["image"]],SMOL_DEFAULT,SMOLDOC_PAGE_MAX_NEW_TOKENS)[0]
            smol_page_results[pid]=item; smol_page_rows.append(page_row("SmolDocling",page,item,item["text"]))
            summary=doctags_summary(item["doctags"]); smol_summaries[pid]=summary
            smol_structure_rows.extend(structure_rows(pid,page["expected_counts"],summary))
            print(pid,{"CER":round(smol_page_rows[-1]["cer"],3),"WER":round(smol_page_rows[-1]["wer"],3),
                       "elements":summary["n_elements"],"table_cells":summary["n_table_cells"],
                       "table_rows":summary["n_table_rows"],"OTSL_tokens":summary["n_otsl_tokens"]})
        print(f"\n{'page':<16}{'scored element':<24}{'expected':>9}{'observed':>9}{'abs err':>8}")
        for r in smol_structure_rows:
            if r["scored"]:
                print(f"{r['page_id']:<16}{r['element_type']:<24}{r['expected_count']:>9}{r['observed_count']:>9}{r['absolute_count_error']:>8}")
        print("Unscored observed types:",{f"{r['page_id']}:{r['element_type']}":r["observed_count"]
                                         for r in smol_structure_rows if not r["scored"]})
        show_block("SmolDocling native DocTags · report",smol_page_results["report"]["doctags"],1500,
                   PAGE_DIR/"report"/"smoldocling.doctags")
    save_state("smol_pages.json",{"results":smol_page_results,"rows":smol_page_rows,
                                  "structure_rows":smol_structure_rows,"summaries":smol_summaries})


# Section 18: native SmolDocling instructions.
def stage_smol_specialized():
    specialized_outputs={}
    if RUN_PAGE_SUITE:
        pages=load_pages()
        smol_model,smol_processor,smol_end,smol_pad,_=load_smoldoc()
        demos=[("report","Convert table to OTSL."),
               ("technical_note","Convert formula to LaTeX."),
               ("report","Find all 'text' elements on the page, retrieve all section headers."),
               ("notice","Detect footer elements on the page.")]
        for pid,instruction in demos:
            item=smol_generate(smol_model,smol_processor,smol_end,smol_pad,
                               [pages[pid]["image"]],instruction,SMOLDOC_PAGE_MAX_NEW_TOKENS)[0]
            specialized_outputs[f"{pid}::{instruction}"]=item
            print(pid,instruction,item["new_tokens"],item["truncated"])
            show_block(f"{pid} · {instruction}",item["doctags"],800,
                       Path(OUTPUT_DIR)/"specialized_instruction_outputs.json")
    save_state("smol_specialized.json",specialized_outputs)


# Section 19: SmolDocling token-budget, shape and hallucination probes.
def stage_smol_probes():
    smol_budget_rows=[]; smol_shape_rows=[]; smol_blank_noise={}
    pages=load_pages()
    smol_model,smol_processor,smol_end,smol_pad,_=load_smoldoc()
    if RUN_TOKEN_BUDGET_EXPERIMENT:
        page=pages["report"]
        for budget in (256,512,1024,2048,4096):
            item=smol_generate(smol_model,smol_processor,smol_end,smol_pad,[page["image"]],SMOL_DEFAULT,budget)[0]
            m=one_metrics(page["reference_text"],item["text"]); s=doctags_summary(item["doctags"])
            smol_budget_rows.append({"model":"SmolDocling","page_id":"report","token_budget":budget,
              "generated_tokens":item["new_tokens"],"truncated":item["truncated"],"cer":m["cer"],"wer":m["wer"],
              "text_chars":len(normalise_text(item["text"])),"n_elements":s["n_elements"],"n_table_cells":s["n_table_cells"],
              "n_table_rows":s["n_table_rows"],"n_otsl_tokens":s["n_otsl_tokens"]})
    if RUN_SHAPE_ROBUSTNESS:
        for name,image in shape_variants(pages["notice"]["image"]).items():
            item=smol_generate(smol_model,smol_processor,smol_end,smol_pad,[image],SMOL_DEFAULT,SMOLDOC_PAGE_MAX_NEW_TOKENS)[0]
            m=one_metrics(pages["notice"]["reference_text"],item["text"])
            smol_shape_rows.append({"model":"SmolDocling","variant":name,"cer":m["cer"],"wer":m["wer"],
                                    "length_ratio":m["length_ratio"],"new_tokens":item["new_tokens"],
                                    "truncated":item["truncated"]})
    blank,noise=probe_images()
    for name,image in (("blank",blank),("noise",noise)):
        item=smol_generate(smol_model,smol_processor,smol_end,smol_pad,[image],SMOL_DEFAULT,256)[0]
        smol_blank_noise[name]={"text":item["text"],"doctags":item["doctags"],
                                "characters":len(item["text"]),"new_tokens":item["new_tokens"],
                                "truncated":item["truncated"],"structure":doctags_summary(item["doctags"])}
    save_state("smol_probes.json",{"budget_rows":smol_budget_rows,"shape_rows":smol_shape_rows,
                                   "blank_noise":smol_blank_noise})
    print({k:{x:v[x] for x in ("characters","new_tokens","truncated")} for k,v in smol_blank_noise.items()})
    got_budget_rows=load_state("got_probes.json")["budget_rows"]
    if got_budget_rows or smol_budget_rows:
        print("\nPaired token-budget comparison · report page")
        print(f"{'model':<13}{'budget':>7}{'generated':>10}{'truncated':>10}{'CER':>8}{'WER':>8}{'cells':>7}{'rows':>6}")
        for r in got_budget_rows+smol_budget_rows:
            cells,rows=(r.get(k) if r.get(k) is not None else "-" for k in ("n_table_cells","n_table_rows"))
            print(f"{r['model']:<13}{r['token_budget']:>7}{r['generated_tokens']:>10}{str(r['truncated']):>10}"
                  f"{r['cer']:>8.3f}{r['wer']:>8.3f}{cells:>7}{rows:>6}")


# ---------------------------------------------------------------------------------------------
# Section 20: cross-model comparison.
# ---------------------------------------------------------------------------------------------
def summary_row(system,m,truncated=0,empty=0):
    return {"system":system,"cer":m["cer"],"cer_macro":m["cer_macro"],"wer":m["wer"],"wer_macro":m["wer_macro"],
            "exact_match":m["exact_match"],"median_cer":m["median_cer"],"p90_cer":m["p90_cer"],
            "reference_chars":m["ref_chars"],"hypothesis_chars":m["hyp_chars"],
            "length_ratio":m["length_ratio"],"empty_hypotheses":empty,"truncated":truncated}


def stage_compare():
    belfort_result_rows=[]; belfort_metric_rows=[]
    if RUN_BELFORT:
        test_records=load_belfort_split("test")
        baselines=load_state("baselines.json"); got=load_state("got_belfort.json"); smol=load_state("smol_belfort.json")
        empty_metrics,constant_metrics=baselines["empty_metrics"],baselines["constant_metrics"]
        got_belfort_metrics,got_belfort_items=got["metrics"],got["items"]
        smol_belfort_metrics,smol_belfort_items=smol["metrics"],smol["items"]
        belfort_metric_rows=[summary_row("Empty baseline",empty_metrics),summary_row("Constant baseline",constant_metrics),
          summary_row("GOT-OCR 2.0",got_belfort_metrics,got_belfort_metrics["truncated"],got_belfort_metrics["empty_hypotheses"]),
          summary_row("SmolDocling",smol_belfort_metrics,smol_belfort_metrics["truncated"],smol_belfort_metrics["empty_hypotheses"])]
        groups=Counter()
        for i,r in enumerate(test_records):
            g,s=got_belfort_items[i],smol_belfort_items[i]
            gm,sm=one_metrics(r["text"],g["text"]),one_metrics(r["text"],s["text"])
            if gm["cer"]<.2 and sm["cer"]<.2: cat="both_good"
            elif gm["cer"]<.2: cat="got_only_good"
            elif sm["cer"]<.2: cat="smoldocling_only_good"
            else: cat="both_poor"
            groups[cat]+=1
            for model,item,m in (("GOT-OCR 2.0",g,gm),("SmolDocling",s,sm)):
                belfort_result_rows.append({"sample_id":r["id"],"reference":r["text"],"model":model,
                 "hypothesis":item["text"],"cer":m["cer"],"wer":m["wer"],"exact_match":m["exact_match"],
                 "reference_chars":m["reference_chars"],"hypothesis_chars":m["hypothesis_chars"],
                 "length_ratio":m["length_ratio"],"new_tokens":item["new_tokens"],"truncated":item["truncated"],
                 "inference_seconds":item["inference_seconds"],
                 "error_category":error_category(m["cer"],m["length_ratio"],not normalise_text(item["text"]),item["truncated"]),
                 "disagreement_category":cat})
        print(dict(groups))
        for row in belfort_metric_rows: print(row["system"],round(row["cer"],3),round(row["wer"],3))
    page_text_rows=load_state("got_pages.json")["rows"]+load_state("smol_pages.json")["rows"]
    save_state("compare.json",{"belfort_result_rows":belfort_result_rows,"belfort_metric_rows":belfort_metric_rows,
                               "page_text_rows":page_text_rows})


# Section 21: try it yourself.
def stage_inspect(INSPECT_PAGE):
    pages=load_state("pages.json"); PAGE_DIR=page_dir()
    if INSPECT_PAGE not in pages: raise ValueError(f"INSPECT_PAGE must be one of {sorted(pages)}")
    page_text_rows=load_state("compare.json")["page_text_rows"]
    got_page_plain=load_state("got_pages.json")["plain"]; smol_page_results=load_state("smol_pages.json")["results"]
    for row in page_text_rows:
        print(row["page_id"],row["model"],
              "CER",round(row["cer"],3),"WER",round(row["wer"],3),"token_F1",round(row["token_f1"],3))
    if page_text_rows:
        page=pages[INSPECT_PAGE]
        with Image.open(page["path"]) as source: view=source.convert("RGB")
        view.thumbnail((700,1000)); view_path=work_dir()/f"inspect_{INSPECT_PAGE}.png"
        view.save(view_path); display_image(view_path)
        show_block(f"Reference · {INSPECT_PAGE}",page["reference_text"],1500,PAGE_DIR/INSPECT_PAGE/"reference.txt")
        for label,store,filename in (("GOT-OCR 2.0 plain",got_page_plain,"got_plain.txt"),
                                     ("SmolDocling text",smol_page_results,"smoldocling.txt")):
            if INSPECT_PAGE not in store: continue
            text=store[INSPECT_PAGE]["text"]
            show_block(f"{label} · {INSPECT_PAGE}",text,1500,PAGE_DIR/INSPECT_PAGE/filename)
            print(f"First word-level differences, {label} vs reference:")
            for d in word_diff(page["reference_text"],text,8) or [{"op":"none","reference":"","model":""}]:
                print(f"  {d['op']:<8} reference={d['reference']!r}  model={d['model']!r}")
        if INSPECT_PAGE in smol_page_results and INSPECT_PAGE!="report":
            show_block(f"SmolDocling native DocTags · {INSPECT_PAGE}",smol_page_results[INSPECT_PAGE]["doctags"],
                       1500,PAGE_DIR/INSPECT_PAGE/"smoldocling.doctags")


# Section 22: resource comparison.
def stage_resources():
    snapshots=load_state("snapshots.json"); test_count=len(load_state("belfort.json")["splits"]["test"])
    got_load,smol_load=load_state("got_load.json"),load_state("smol_load.json")
    got,smol=load_state("got_belfort.json"),load_state("smol_belfort.json")
    got_page_plain=load_state("got_pages.json")["plain"]; smol_page_results=load_state("smol_pages.json")["results"]
    resource_rows=[
     {"model":"GOT-OCR 2.0","parameters":got_load["parameters"],
      "weight_bytes":next(x["bytes"] for x in GOT_MANIFEST["files"] if x["path"]=="model.safetensors"),
      "input_strategy":"1024x1024 square; aspect ratio not preserved",
      "snapshot_verify_seconds":snapshots["got"]["verify_seconds"],"load_seconds":got_load["load_seconds"],
      "belfort_seconds":got["seconds"],
      "mean_line_latency_s":got["seconds"]/test_count if RUN_BELFORT else None,
      "mean_page_latency_s":float(np.mean([x["inference_seconds"] for x in got_page_plain.values()])) if got_page_plain else None,
      "peak_gpu_memory_bytes":got["peak_gpu"]},
     {"model":"SmolDocling","parameters":smol_load["parameters"],
      "weight_bytes":next(x["bytes"] for x in SMOL_MANIFEST["files"] if x["path"]=="model.safetensors"),
      "input_strategy":"longest edge 2048; 512px tiles + global view",
      "snapshot_verify_seconds":snapshots["smoldocling"]["verify_seconds"],"load_seconds":smol_load["load_seconds"],
      "belfort_seconds":smol["seconds"],
      "mean_line_latency_s":smol["seconds"]/test_count if RUN_BELFORT else None,
      "mean_page_latency_s":float(np.mean([x["inference_seconds"] for x in smol_page_results.values()])) if smol_page_results else None,
      "peak_gpu_memory_bytes":smol["peak_gpu"]}]
    save_state("resources.json",resource_rows)
    for r in resource_rows:print(r)


# ---------------------------------------------------------------------------------------------
# Section 24: machine-readable exports.
# ---------------------------------------------------------------------------------------------
def write_csv(path,rows,fields):
    with open(path,"w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows:w.writerow({k:r.get(k) for k in fields})


def compact(m):
    if not isinstance(m,dict):return m
    return {k:v for k,v in m.items() if k!="rows"}


def stage_export():
    OUT_ROOT=Path(OUTPUT_DIR); OUT_ROOT.mkdir(parents=True,exist_ok=True); PAGE_DIR=page_dir()
    compare=load_state("compare.json"); baselines=load_state("baselines.json")
    got_pages,smol_pages=load_state("got_pages.json"),load_state("smol_pages.json")
    got_probes,smol_probes=load_state("got_probes.json"),load_state("smol_probes.json")
    belfort_result_rows,belfort_metric_rows=compare["belfort_result_rows"],compare["belfort_metric_rows"]
    page_text_rows=compare["page_text_rows"]
    smol_structure_rows,smol_summaries=smol_pages["structure_rows"],smol_pages["summaries"]
    got_budget_rows,smol_budget_rows=got_probes["budget_rows"],smol_probes["budget_rows"]
    got_shape_rows,smol_shape_rows=got_probes["shape_rows"],smol_probes["shape_rows"]
    got_blank_noise,smol_blank_noise=got_probes["blank_noise"],smol_probes["blank_noise"]
    resource_rows=load_state("resources.json"); specialized_outputs=load_state("smol_specialized.json")
    got_format_outputs,got_page_plain,smol_page_results=got_pages["format_outputs"],got_pages["plain"],smol_pages["results"]
    got_belfort_metrics=load_state("got_belfort.json")["metrics"]; smol_belfort_metrics=load_state("smol_belfort.json")["metrics"]
    got_parameter_count=load_state("got_load.json")["parameters"]; smol_parameter_count=load_state("smol_load.json")["parameters"]
    pages=load_pages()

    write_csv(OUT_ROOT/"belfort_results.csv",belfort_result_rows,
     ["sample_id","reference","model","hypothesis","cer","wer","exact_match","reference_chars",
      "hypothesis_chars","length_ratio","new_tokens","truncated","inference_seconds","error_category","disagreement_category"])
    write_csv(OUT_ROOT/"belfort_metrics.csv",belfort_metric_rows,
     ["system","cer","cer_macro","wer","wer_macro","exact_match","median_cer","p90_cer",
      "reference_chars","hypothesis_chars","length_ratio","empty_hypotheses","truncated"])
    write_csv(OUT_ROOT/"page_text_metrics.csv",page_text_rows,
     ["page_id","page_kind","model","cer","wer","token_precision","token_recall","token_f1",
      "reference_chars","hypothesis_chars","length_ratio","new_tokens","truncated","inference_seconds"])
    write_csv(OUT_ROOT/"smoldocling_structure.csv",smol_structure_rows,
     ["page_id","element_type","scored","expected_count","observed_count","absolute_count_error"])
    write_csv(OUT_ROOT/"token_budget.csv",got_budget_rows+smol_budget_rows,
     ["model","page_id","token_budget","generated_tokens","truncated","cer","wer","text_chars","n_elements","n_table_cells",
      "n_table_rows","n_otsl_tokens"])
    write_csv(OUT_ROOT/"shape_robustness.csv",got_shape_rows+smol_shape_rows,
     ["model","variant","cer","wer","length_ratio","new_tokens","truncated"])
    write_csv(OUT_ROOT/"resource_metrics.csv",resource_rows,
     ["model","parameters","weight_bytes","input_strategy","snapshot_verify_seconds","load_seconds",
      "belfort_seconds","mean_line_latency_s","mean_page_latency_s","peak_gpu_memory_bytes"])

    (OUT_ROOT/"got_format_outputs.json").write_text(json.dumps(got_format_outputs,indent=2,ensure_ascii=False),encoding="utf-8")
    (OUT_ROOT/"smoldocling_summary.json").write_text(json.dumps(smol_summaries,indent=2,ensure_ascii=False),encoding="utf-8")
    (OUT_ROOT/"specialized_instruction_outputs.json").write_text(json.dumps(specialized_outputs,indent=2,ensure_ascii=False),encoding="utf-8")
    (OUT_ROOT/"blank_noise_probes.json").write_text(json.dumps({"got":got_blank_noise,"smoldocling":smol_blank_noise},
                                                             indent=2,ensure_ascii=False),encoding="utf-8")

    # One folder per synthetic page with raw native outputs.
    for pid,page in pages.items():
        pdir=PAGE_DIR/pid; pdir.mkdir(exist_ok=True)
        page["image"].save(pdir/"source.png")
        (pdir/"reference.txt").write_text(page["reference_text"],encoding="utf-8")
        if pid in got_page_plain:(pdir/"got_plain.txt").write_text(got_page_plain[pid]["text"],encoding="utf-8")
        if pid in got_format_outputs:(pdir/"got_format.txt").write_text(got_format_outputs[pid]["text"],encoding="utf-8")
        if pid in smol_page_results:
            (pdir/"smoldocling.txt").write_text(smol_page_results[pid]["text"],encoding="utf-8")
            (pdir/"smoldocling.doctags").write_text(smol_page_results[pid]["doctags"],encoding="utf-8")

    metrics_export={
     "baselines":{"empty":compact(baselines["empty_metrics"]),"constant":compact(baselines["constant_metrics"])},
     "belfort":{"got":compact(got_belfort_metrics),"smoldocling":compact(smol_belfort_metrics)},
     "rendered_pages":page_text_rows,"smoldocling_structure":smol_summaries,
     "token_budget":got_budget_rows+smol_budget_rows,"shape_robustness":got_shape_rows+smol_shape_rows,
     "resources":resource_rows}
    (OUT_ROOT/"metrics.json").write_text(json.dumps(metrics_export,indent=2,ensure_ascii=False),encoding="utf-8")

    provenance={
     "notebook_spec":"2.1","profile":"TASK-INFERENCE","pedagogical_mode":"WORKSHOP","standalone":True,
     "got":{"model_id":GOT_MANIFEST["modelId"],"revision":GOT_MANIFEST["revision"],"license":"Apache-2.0",
            "manifest":GOT_MANIFEST,"parameter_count":got_parameter_count,
            "preprocessing":"1024x1024 square; aspect ratio not preserved","decoding":"greedy"},
     "smoldocling":{"model_id":SMOL_MANIFEST["modelId"],"revision":SMOL_MANIFEST["revision"],
            "license":"CDLA-Permissive-2.0","manifest":SMOL_MANIFEST,"parameter_count":smol_parameter_count,
            "preprocessing":"longest edge 2048; 512px tiles + global view","decoding":"greedy"},
     "belfort":{"repo":CORPUS_REPO,"revision":CORPUS_REVISION,"file":CORPUS_FILE,
            "row_group_pins":ROW_GROUP_PINS,"sample_digest":SAMPLE_DIGEST,
            "split_seed":SAMPLE_SEED,"split_sizes":SAMPLE_SPLIT},
     "rendered_pages":{"generator":"Pillow deterministic renderer v2 (draw-order references)",
            "reference_policy":REFERENCE_POLICY,
            "pages":{k:{"kind":v["kind"],"expected_counts":v["expected_counts"],
                        "reference_sha256":hashlib.sha256(v["reference_text"].encode("utf-8")).hexdigest()}
                     for k,v in pages.items()}},
     "runtime":RUNTIME}
    (OUT_ROOT/"provenance.json").write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding="utf-8")
    print("Exports written.")


# ---------------------------------------------------------------------------------------------
# Section 26: BYOD, labelled or unlabelled.
# ---------------------------------------------------------------------------------------------
BYOD_EXTENSIONS={".png",".jpg",".jpeg",".webp",".tif",".tiff",".bmp"}


def safe_identifier(value):
    value=str(value).strip()
    if not value or value in (".","..") or any(ch in value for ch in ("/", "\\", ":", "\x00")):
        raise ValueError(f"Unsafe or empty BYOD id: {value!r}")
    return value


def load_byod(path):
    src=Path(path)
    source_zip_sha256=None
    if src.is_file() and src.suffix.lower()==".zip":
        source_zip_sha256=sha256_file(src)
        with zipfile.ZipFile(src) as z:
            total=0; members=[]; seen=set()
            for info in z.infolist():
                name=info.filename.replace("\\","/"); parts=[p for p in name.split("/") if p]
                if info.is_dir(): continue
                if name.startswith("/") or ".." in parts or any(":" in part for part in parts):
                    raise ValueError(f"unsafe ZIP member {name}")
                mode=(info.external_attr>>16)&0o170000
                if mode==0o120000: raise ValueError(f"symlink refused {name}")
                total+=info.file_size
                if total>1_000_000_000: raise ValueError("ZIP exceeds 1 GB expanded")
                basename=Path(name).name
                if basename.casefold() in seen: raise ValueError(f"Duplicate ZIP basename: {basename}")
                if Path(basename).suffix.lower() not in BYOD_EXTENSIONS and basename!="transcripts.csv":
                    raise ValueError(f"Unsupported BYOD ZIP file: {basename}; supply images and optional transcripts.csv only")
                seen.add(basename.casefold()); members.append((info,basename))
            root=Path(tempfile.mkdtemp(prefix="ocr-byod-"))
            for info,basename in members: (root/basename).write_bytes(z.read(info))
    elif src.is_dir(): root=src
    else: raise ValueError("BYOD_PATH must be directory or ZIP")
    unexpected=[p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() not in BYOD_EXTENSIONS and p.name!="transcripts.csv"]
    if unexpected: raise ValueError(f"Unsupported BYOD files: {unexpected}; supply images and optional transcripts.csv only")
    if any(p.is_dir() for p in root.iterdir()): raise ValueError("BYOD directories must contain page images directly, not nested folders")
    images=sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in BYOD_EXTENSIONS)
    if not 1<=len(images)<=500: raise ValueError("BYOD supports 1..500 images")
    filenames=[p.name for p in images]
    if len({name.casefold() for name in filenames})!=len(filenames):
        raise ValueError("Duplicate BYOD image filenames")
    transcript_file=root/"transcripts.csv"; refs={}; ids={}; transcript_sha256=None
    labelled=transcript_file.is_file()
    if labelled:
        payload=transcript_file.read_bytes(); transcript_sha256=hashlib.sha256(payload).hexdigest()
        table=[values for values in csv.reader(io.StringIO(payload.decode("utf-8-sig"))) if values]
        header=table[0] if table else []
        repeated=sorted({h for h in header if header.count(h)>1})
        if repeated: raise ValueError(f"transcripts.csv has duplicate column names: {repeated}")
        if len(table)<2 or not {"file","text"}.issubset(header):
            raise ValueError("transcripts.csv requires file,text and exactly one row per image")
        rows=[]
        for number,values in enumerate(table[1:],1):
            if len(values)!=len(header):
                raise ValueError(f"transcripts.csv row {number} has {len(values)} fields but the header has "
                                 f"{len(header)}; use an explicit empty CSV field for a blank page")
            rows.append(dict(zip(header,values)))
        for r in rows:
            name=(r.get("file") or "").strip()
            if not name or name in refs: raise ValueError(f"Duplicate or empty transcript filename: {name!r}")
            refs[name]=normalise_text(r["text"])
            ids[name]=safe_identifier(r.get("id") or name)
        missing=sorted(set(filenames)-set(refs)); extra=sorted(set(refs)-set(filenames))
        if missing or extra: raise ValueError(f"Transcript files must match images exactly; missing={missing}, extra={extra}")
    effective_ids=[ids.get(name,name) for name in filenames]
    if len({value.casefold() for value in effective_ids})!=len(effective_ids):
        raise ValueError("BYOD ids must be unique")
    out=[]
    for p,record_id in zip(images,effective_ids,strict=True):
        payload=p.read_bytes()
        with Image.open(io.BytesIO(payload)) as decoded:
            frames=getattr(decoded,"n_frames",1)
            if frames>1:
                raise ValueError(f"{p.name}: {frames} pages/frames; BYOD accepts single-page images only. "
                                 "Save one file per page.")
            decoded.load(); image=decoded.convert("RGB")
        if min(image.size)<MIN_IMAGE_SIDE or max(image.size)>MAX_IMAGE_SIDE or image.width*image.height>MAX_IMAGE_PIXELS:
            raise ValueError(f"{p.name}: outside image ceilings")
        out.append({"id":record_id,"file":p.name,"image":image,"text":refs.get(p.name),
                    "source_sha256":hashlib.sha256(payload).hexdigest()})
    return out,labelled,{"source_zip_sha256":source_zip_sha256,"transcripts_sha256":transcript_sha256,
                         "images":{r["file"]:r["source_sha256"] for r in out}}


def byod_text_metrics(reference,hypothesis):
    if normalise_text(reference): return one_metrics(reference,hypothesis)
    hyp=normalise_text(hypothesis)
    return {"cer":None,"wer":None,"length_ratio":None,"exact_match":hyp=="",
            "reference_chars":0,"reference_words":0,"hypothesis_chars":len(hyp),
            "char_edits":len(hyp),"word_edits":len(words(hyp)),
            "rate_reason":"Empty reference: CER/WER and length ratio have zero denominators"}


def empty_device_cache():
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def infer_byod(records):
    gm=gp=None
    try:
        gm,gp,gpad,gstop,_=load_got(); gplain=[]; gformat=[]
        for start in range(0,len(records),BATCH_SIZE):
            batch=records[start:start+BATCH_SIZE]
            gplain.extend(got_generate(gm,gp,gpad,gstop,[r["image"] for r in batch],"plain",GOT_PAGE_MAX_NEW_TOKENS))
        for r in records:
            gformat.append(got_generate(gm,gp,gpad,gstop,[r["image"]],"format",GOT_PAGE_MAX_NEW_TOKENS)[0])
    except Exception as exc:
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        gm=gp=None; gc.collect()
        empty_device_cache()
    sm=sp=None
    try:
        sm,sp,se,spd,_=load_smoldoc(); sitems=[]
        for start in range(0,len(records),BATCH_SIZE):
            batch=records[start:start+BATCH_SIZE]
            sitems.extend(smol_generate(sm,sp,se,spd,[r["image"] for r in batch],SMOL_DEFAULT,SMOLDOC_PAGE_MAX_NEW_TOKENS))
    except Exception as exc:
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        sm=sp=None; gc.collect()
        empty_device_cache()
    for label,items in (("GOT plain",gplain),("GOT format",gformat),("SmolDocling",sitems)):
        if len(items)!=len(records): raise RuntimeError(f"{label}: output count does not match input images")
    return gplain,gformat,sitems


def stage_byod():
    OUT_ROOT=Path(OUTPUT_DIR)
    state_path("byod.json").unlink(missing_ok=True)
    byod_report=None
    if not USE_BYOD:
        print("BYOD disabled.")
        return None
    records,labelled,byod_input=load_byod(BYOD_PATH)
    gplain,gformat,sitems=infer_byod(records)
    byod_parent=OUT_ROOT/"byod"; byod_parent.mkdir(parents=True,exist_ok=True)
    byod_dir=Path(tempfile.mkdtemp(prefix="run-",dir=byod_parent))
    rows=[]; output_files=[]
    for i,r in enumerate(records):
        stem=f"page-{i+1:04d}"
        for model,item in (("GOT-OCR 2.0",gplain[i]),("SmolDocling",sitems[i])):
            row={"id":r["id"],"file":r["file"],"output_stem":stem,"model":model,"text":item["text"],
                 "new_tokens":item["new_tokens"],"truncated":item["truncated"]}
            if labelled: row.update(byod_text_metrics(r["text"],item["text"]))
            rows.append(row)
        for suffix,text in (("got_plain.txt",gplain[i]["text"]),("got_format.txt",gformat[i]["text"]),
                            ("smoldocling.txt",sitems[i]["text"]),("smoldocling.doctags",sitems[i]["doctags"])):
            output=byod_dir/f"{stem}.{suffix}"; output.write_text(text,encoding="utf-8"); output_files.append(output.name)
    fields=["id","file","output_stem","model","text","new_tokens","truncated"]
    if labelled: fields += ["cer","wer","exact_match","reference_chars","hypothesis_chars","length_ratio","char_edits","word_edits","reference_words","rate_reason"]
    write_csv(byod_dir/"results.csv",rows,fields)
    byod_report={"images":len(records),"evaluation_verdict":"measured" if labelled else "not-measurable",
                 "undefined_rate_rows":sum(r.get("rate_reason") is not None for r in rows),
                 "output_directory":str(byod_dir),"native_output_files":output_files,
                 "input":byod_input,"runtime":RUNTIME,
                 "models":{"got":GOT_MANIFEST,"smoldocling":SMOL_MANIFEST},
                 "generation":{"got_page_max_new_tokens":GOT_PAGE_MAX_NEW_TOKENS,
                               "smoldocling_page_max_new_tokens":SMOLDOC_PAGE_MAX_NEW_TOKENS,"decoding":"greedy"}}
    (byod_dir/"report.json").write_text(json.dumps(byod_report,indent=2,ensure_ascii=False),encoding="utf-8")
    save_state("byod.json",byod_report)
    print({k:byod_report[k] for k in ("images","evaluation_verdict","undefined_rate_rows","output_directory")})
    print("Preview of the first inputs; full outputs are the files listed in results.csv:")
    for i,r in enumerate(records[:3]):
        stem=f"page-{i+1:04d}"
        for label,item,suffix in (("GOT-OCR 2.0 plain",gplain[i],"got_plain.txt"),
                                  ("SmolDocling text",sitems[i],"smoldocling.txt")):
            show_block(f"{r['file']} · {label}",item["text"],600,byod_dir/f"{stem}.{suffix}")
        if labelled:
            for row in rows[2*i:2*i+2]:
                print(f"  {row['model']}: CER={row['cer']} WER={row['wer']} exact={row['exact_match']}")
    return byod_report


# ---------------------------------------------------------------------------------------------
# Section 28: terminal summary.
# ---------------------------------------------------------------------------------------------
def stage_summary():
    OUT_ROOT=Path(OUTPUT_DIR)
    required=[OUT_ROOT/"belfort_results.csv",OUT_ROOT/"belfort_metrics.csv",OUT_ROOT/"page_text_metrics.csv",
     OUT_ROOT/"smoldocling_structure.csv",OUT_ROOT/"token_budget.csv",OUT_ROOT/"shape_robustness.csv",
     OUT_ROOT/"resource_metrics.csv",OUT_ROOT/"got_format_outputs.json",OUT_ROOT/"smoldocling_summary.json",
     OUT_ROOT/"specialized_instruction_outputs.json",OUT_ROOT/"metrics.json",OUT_ROOT/"provenance.json"]
    if USE_BYOD:
        byod_report=load_state("byod.json"); byod_dir=Path(byod_report["output_directory"])
        required.extend([byod_dir/"results.csv",byod_dir/"report.json"])
        required.extend(byod_dir/name for name in byod_report["native_output_files"])
    missing=[str(p) for p in required if not p.is_file()]
    if missing:raise RuntimeError(f"missing outputs: {missing}")

    compare=load_state("compare.json"); belfort_metric_rows=compare["belfort_metric_rows"]
    page_text_rows=compare["page_text_rows"]; smol_summaries=load_state("smol_pages.json")["summaries"]
    print("DIMER OCR & Document Extraction Notebook")
    print("-"*44)
    print(f"Belfort held-out lines: {len(load_state('belfort.json')['splits']['test'])}")
    if RUN_BELFORT:
        print(f"{'System':<22} {'CER':>8} {'WER':>8} {'Exact':>8}")
        for r in belfort_metric_rows:
            print(f"{r['system']:<22} {r['cer']:>8.3f} {r['wer']:>8.3f} {r['exact_match']:>8.3f}")
    print(f"Rendered pages: {len(load_state('pages.json'))}")
    for model in ("GOT-OCR 2.0","SmolDocling"):
        rows=[r for r in page_text_rows if r["model"]==model]
        if rows:
            print(model, "mean CER",round(float(np.mean([r["cer"] for r in rows])),3),
                  "mean WER",round(float(np.mean([r["wer"] for r in rows])),3),
                  "truncated",sum(r["truncated"] for r in rows))
    if smol_summaries:print("SmolDocling table cells / rows (logical, row ends excluded):",
                            sum(x["n_table_cells"] for x in smol_summaries.values()),"/",
                            sum(x["n_table_rows"] for x in smol_summaries.values()))
    print("Outputs:",OUT_ROOT)


# ---------------------------------------------------------------------------------------------
# Command line.
# ---------------------------------------------------------------------------------------------
STAGES = {
    "environment": stage_environment, "snapshots": stage_snapshots, "belfort": stage_belfort,
    "baselines": stage_baselines, "pages": stage_pages, "got-load": stage_got_load,
    "got-belfort": stage_got_belfort, "got-pages": stage_got_pages, "got-probes": stage_got_probes,
    "smol-load": stage_smol_load, "smol-belfort": stage_smol_belfort, "smol-pages": stage_smol_pages,
    "smol-specialized": stage_smol_specialized, "smol-probes": stage_smol_probes, "compare": stage_compare,
    "inspect": stage_inspect, "resources": stage_resources, "export": stage_export, "byod": stage_byod,
    "summary": stage_summary,
}


def main(argv=None):
    global RUNTIME
    parser = argparse.ArgumentParser(description="Run one stage of the OCR workshop notebook.")
    parser.add_argument("--config", required=True, help="stage config JSON written by the notebook")
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--page", help="page id for the inspect stage")
    args = parser.parse_args(argv)
    configure(json.loads(Path(args.config).read_text(encoding="utf-8")))
    if args.stage != "environment" and state_path("runtime.json").is_file():
        RUNTIME = load_state("runtime.json")
    if args.stage == "inspect":
        stage_inspect(args.page)
    else:
        STAGES[args.stage]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
