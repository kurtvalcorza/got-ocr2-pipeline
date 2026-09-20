"""Labelled text-line datasets for the adaptation contract: the digest-pinned Belfort sample, the record contract
and its structural validation, image-disjoint splitting, and the BYOD loader.

A record is ``{id, image, text}`` where ``image`` is a PIL image of one text line (or any region whose transcript is
known; sides within the pipeline's ceilings) and ``text`` its transcript — the string the model is asked to
generate for that image. Whitespace runs in ``text`` are collapsed; case and punctuation are kept.

The default sample is drawn from the Belfort-line dataset (Teklia; the minutes of the Belfort municipal council,
19th–20th century French handwriting transcribed by a crowdsourcing campaign; Tarride et al. 2023, **MIT**) as
converted to parquet by the Hugging Face Hub at an immutable revision: the first ``CORPUS_ROW_GROUPS`` row groups of
the test shard are read with HTTPS range requests (about 5.5 MB each; the shard's declared size is checked first and
every row group's decoded content is refused unless its SHA-256 matches the pin). Every line image is 128 px tall;
widths run from about 180 to 9,000 px. The domain gap to the model's printed-text training distribution is the point
of the sample: the frozen model reads almost none of it.
"""
# ruff: noqa: E501  -- record and pin literals are kept on single lines

from __future__ import annotations

import csv
import hashlib
import io
import random
import re
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .pipeline import MODEL_ID, normalise_text, validate_image

CORPUS_NAME = "Belfort-line (test split), first four parquet row groups"
CORPUS_REPO = "Teklia/Belfort-line"
CORPUS_REVISION = "c4a74bbd39f2df314752e7e6026649a39d365cbb"  # refs/convert/parquet commit on the Hub
CORPUS_FILE = "default/test/0000.parquet"
CORPUS_BYTES = 210_579_166
CORPUS_ROWS = 3_819
CORPUS_ROW_GROUPS = 4  # of 39; 100 lines each
CORPUS_LICENSE = "MIT (Teklia; Belfort municipal council minutes, Zenodo record 8041668; Tarride et al. 2023, https://doi.org/10.1145/3604951.3605517)"
CORPUS_LANGUAGE = "fr"
CORPUS_URL = f"https://huggingface.co/datasets/{CORPUS_REPO}/resolve/{CORPUS_REVISION}/{CORPUS_FILE}"
# SHA-256 over the concatenated image bytes + UTF-8 transcript of each row group, in row order, and that byte total.
ROW_GROUP_PINS: dict[int, tuple[str, int]] = {
    0: ("1dc3141e4809ea628b17c3ca7b81d64e6ca92bce18dd5765ecd618bfc7867954", 5_481_145),
    1: ("c9d3b52013933c803f4886edbce68da0ae483347a4a6ff3e0f5ced1db4a7e653", 5_465_901),
    2: ("6e0578a90a07a9e25e65b765881d3fa33d6a797624425e01026980d7287f0bf6", 5_379_166),
    3: ("00cdfb7aabe924f31b9f1bb1ba4849040051e5619567b68bf99fdcbcab15131a", 5_821_303),
}
DEFAULT_CACHE_DIR = Path("weights") / "belfort"

SAMPLE_SEED = 42
SAMPLE_SPLIT = {"train": 280, "validation": 40, "test": 80}  # of the 400 lines the four row groups hold
SAMPLE_DIGEST = "10aece2bc2b1465de7e4b9b06e76825f3743eec688659ba72aba2e197fa3497e"  # dataset_digest over the three default splits together; tests pin it
MIN_RECORDS = 8
MAX_RECORDS = 5_000
MIN_TEXT_CHARS = 1
MAX_TEXT_CHARS = 512
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _HttpRangeFile(io.RawIOBase):
    """A seekable read-only view of one HTTPS object served with `Range` requests (what `pyarrow` needs to read a
    parquet footer and a few row groups without downloading the file)."""

    def __init__(self, url: str, size: int) -> None:
        self.url, self.size, self.pos = url, size, 0
        self.fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "got-ocr2-pipeline"})
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 (pinned https URL)
            if response.status != 206:
                raise ValueError(f"{self.url}: server ignored the Range request (HTTP {response.status})")
            data = response.read()
        self.fetched += len(data)
        self.pos += len(data)
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _declared_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "got-ocr2-pipeline"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 (pinned https URL)
        length = response.headers.get("Content-Length")
    if length is None:
        raise ValueError(f"{url}: no Content-Length in the HEAD response")
    return int(length)


def _group_digest(rows: Sequence[Mapping[str, Any]]) -> tuple[str, int]:
    digest, total = hashlib.sha256(), 0
    for row in rows:
        data = row["image"]["bytes"]
        text = str(row["text"]).encode("utf-8")
        digest.update(data)
        digest.update(text)
        total += len(data) + len(text)
    return digest.hexdigest(), total


def fetch_corpus(
    *, cache_dir: str | Path | None = None, groups: Sequence[int] | None = None, opener: Any = None
) -> dict[int, list[dict[str, Any]]]:
    """Return the pinned row groups as lists of `{image, text}` (JPEG bytes, transcript), from the cache (one parquet
    file per row group) or the Hub (footer + the row groups it needs, over range requests). Every row group's decoded
    content is refused unless its SHA-256 and byte total match `ROW_GROUP_PINS`; a fresh fetch also checks the shard's
    declared size and row count."""
    import pyarrow.parquet as pq

    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    wanted = list(groups) if groups is not None else sorted(ROW_GROUP_PINS)
    out: dict[int, list[dict[str, Any]]] = {}
    reader = None
    for group in wanted:
        if group not in ROW_GROUP_PINS:
            raise ValueError(f"row group {group} has no pin; pinned groups are {sorted(ROW_GROUP_PINS)}")
        local = cache / f"test-rg{group}.parquet"
        rows: list[dict[str, Any]] | None = None
        if local.is_file():
            rows = pq.read_table(local).to_pylist()
            if _group_digest(rows) != ROW_GROUP_PINS[group]:
                rows = None  # stale or corrupt cache: refetch
        if rows is None:
            if reader is None:
                if opener is not None:
                    reader = pq.ParquetFile(opener(CORPUS_URL))
                else:
                    declared = _declared_size(CORPUS_URL)
                    if declared != CORPUS_BYTES:
                        raise ValueError(f"{CORPUS_FILE}: declared size {declared} != pinned {CORPUS_BYTES}")
                    reader = pq.ParquetFile(_HttpRangeFile(CORPUS_URL, CORPUS_BYTES))
                if reader.metadata.num_rows != CORPUS_ROWS:
                    raise ValueError(f"{CORPUS_FILE}: {reader.metadata.num_rows} rows, pinned {CORPUS_ROWS}")
            table = reader.read_row_group(group, columns=["image", "text"])
            rows = table.to_pylist()
            digest, total = _group_digest(rows)
            if (digest, total) != ROW_GROUP_PINS[group]:
                raise ValueError(f"{CORPUS_FILE} row group {group}: sha256 {digest} / {total} bytes != pinned {ROW_GROUP_PINS[group]}")
            pq.write_table(table, local)
        out[group] = [{"image": r["image"]["bytes"], "text": str(r["text"])} for r in rows]
    return out


def read_corpus(groups: Mapping[int, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    """Decode the verified row groups into records (one per line; lines with an empty transcript are skipped)."""
    out = []
    for group in sorted(groups):
        for index, row in enumerate(groups[group]):
            text = normalise_text(row["text"])
            if not text:
                continue
            image = Image.open(io.BytesIO(row["image"]))
            image.load()
            out.append({"id": f"belfort-test-{group * 100 + index}", "image": image.convert("RGB"), "text": text, "source_row_group": group})
    return out


def build_sample_dataset(
    records: Sequence[Mapping[str, Any]], *, seed: int = SAMPLE_SEED, sizes: Mapping[str, int] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Seeded line-level draw: shuffle the records and cut `sizes` (train / validation / test) in order."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    pool = [dict(r) for r in records]
    random.Random(seed).shuffle(pool)
    needed = sum(sizes.values())
    if len(pool) < needed:
        raise ValueError(f"only {len(pool)} records available, need {needed}")
    out, cursor = {}, 0
    for name, count in sizes.items():
        out[name] = pool[cursor : cursor + count]
        cursor += count
    return out


def fetch_sample_dataset(*, cache_dir: str | Path | None = None, seed: int = SAMPLE_SEED) -> dict[str, list[dict[str, Any]]]:
    return build_sample_dataset(read_corpus(fetch_corpus(cache_dir=cache_dir)), seed=seed)


# ---------------------------------------------------------------------------------------------------------
# Record contract
# ---------------------------------------------------------------------------------------------------------


def _open(image: Any, where: str) -> Image.Image:
    if isinstance(image, str | Path):
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"{where}: image file not found: {path}")
        image = Image.open(path)
        image.load()
    if not isinstance(image, Image.Image):
        raise ValueError(f"{where}: must be a PIL.Image.Image or a file path")
    return image


def _check_record(record: Any, index: int) -> dict[str, Any]:
    where = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{where} must be a mapping with id/image/text")
    for key in ("id", "image", "text"):
        if key not in record:
            raise ValueError(f"{where} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{where}: id must match {_ID_RE.pattern}")
    try:
        image = validate_image(_open(record["image"], f"{where}.image"))
    except TypeError as exc:
        raise ValueError(f"{where}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc
    if not isinstance(record["text"], str):
        raise ValueError(f"{where}: text must be a str")
    text = normalise_text(record["text"])
    if not MIN_TEXT_CHARS <= len(text) <= MAX_TEXT_CHARS:
        raise ValueError(f"{where}: text has {len(text)} characters after whitespace normalisation; {MIN_TEXT_CHARS}..{MAX_TEXT_CHARS} are required")
    item = {"id": rid, "image": image, "text": text}
    if "source_row_group" in record:
        item["source_row_group"] = record["source_row_group"]
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a text-line dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, image, text} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked, ids = [], set()
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        checked.append(item)
    chars = [len(r["text"]) for r in checked]
    words = [len(r["text"].split()) for r in checked]
    widths = [r["image"].width for r in checked]
    heights = [r["image"].height for r in checked]
    return {
        "records": checked,
        "n_records": len(checked),
        "text_chars": {"min": min(chars), "max": max(chars), "total": sum(chars)},
        "text_words": {"total": sum(words)},
        "image_width": {"min": min(widths), "max": max(widths)},
        "image_height": {"min": min(heights), "max": max(heights)},
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def image_digest(image: Image.Image) -> str:
    """SHA-256 of the decoded RGB pixels (size-prefixed) — the identity a split is made disjoint on."""
    rgb = image.convert("RGB")
    return _sha256_bytes(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes())


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Order-independent SHA-256 over (id, image digest, normalised text)."""
    parts = sorted(f"{r['id']}:{image_digest(r['image'])}:{normalise_text(r['text'])}" for r in records)
    return _sha256_bytes("\n".join(parts).encode("utf-8"))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no image (by decoded-pixel digest) appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = image_digest(record["image"])
            if key in seen and seen[key] != name:
                raise ValueError(f"image {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]], *, val_fraction: float = 0.15, test_fraction: float = 0.2, seed: int = 0
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train/validation/test after de-duplicating images."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    unique = []
    for record in checked:
        key = image_digest(record["image"])
        if key not in seen:
            seen.add(key)
            unique.append(record)
    random.Random(seed).shuffle(unique)
    n = len(unique)
    n_test = max(1, round(n * test_fraction))
    n_val = round(n * val_fraction)
    if n - n_test - n_val < 1:
        raise ValueError(f"{n} distinct images are too few to split into train/validation/test")
    return {"test": unique[:n_test], "validation": unique[n_test : n_test + n_val], "train": unique[n_test + n_val :]}


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Records from a directory or zip holding line images and a `transcripts.csv` with the columns `file` and `text`
    (and optionally `id`); every image file must have a transcript row and every row an image."""
    source = Path(path)
    members: dict[str, bytes] = {}
    if source.is_dir():
        for file in sorted(source.rglob("*")):
            if file.is_file():
                members[file.name] = file.read_bytes()
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    members[Path(info.filename).name] = archive.read(info)  # flattened; no extractall
    else:
        raise ValueError(f"{source} is neither a directory nor a zip file")
    if "transcripts.csv" not in members:
        raise ValueError("BYOD data must include transcripts.csv with the columns file and text")
    rows = list(csv.DictReader(io.StringIO(members["transcripts.csv"].decode("utf-8-sig"))))
    if not rows or "file" not in rows[0] or "text" not in rows[0]:
        raise ValueError("transcripts.csv must have the columns file and text")
    out = []
    for row in rows:
        name = Path(str(row.get("file", "")).strip()).name
        if name not in members:
            raise ValueError(f"transcripts.csv names a missing image: {name}")
        try:
            image = Image.open(io.BytesIO(members[name]))
            image.load()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"BYOD file is not a decodable image: {name}") from exc
        rid = str(row.get("id", "") or "").strip()
        out.append({"id": rid or re.sub(r"[^A-Za-z0-9_.:-]", "_", Path(name).stem)[:64], "image": image.convert("RGB"), "text": str(row.get("text", ""))})
    listed = {Path(str(r.get("file", "")).strip()).name for r in rows}
    unlisted = [n for n in members if n != "transcripts.csv" and n not in listed]
    if unlisted:
        raise ValueError(f"{len(unlisted)} image file(s) have no transcripts.csv row, e.g. {unlisted[0]}")
    return out


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """A summary table (id, image size, transcript length, transcript, provenance) in the BYOD `transcripts.csv`
    column layout plus extras (`file` names the id; the images themselves are not written)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "file", "width", "height", "chars", "words", "text", "source_row_group"])
        for r in records:
            text = normalise_text(r["text"])
            writer.writerow([r["id"], f"{r['id']}.jpg", r["image"].width, r["image"].height, len(text), len(text.split()), text, r.get("source_row_group", "")])
    return out
