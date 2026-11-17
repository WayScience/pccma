#!/usr/bin/env python3
"""Package immutable pilot inputs, stage a pinned archive, and build current Pages code.

Only report-needed Parquet columns, the original compressed JPEG hash list, and
selected original JPEGs travel in the input archive. Public tables keep all rows
and unrounded values, but omit unused private source paths. HTML, JavaScript and
DuckDB are built from this checkout. Commands never replace existing outputs.
"""

import argparse
import ctypes
import gzip
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import tempfile
from urllib.error import URLError
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import pandas as pd

import build_report as report

INPUTS = {
    "umap": "inputs/single_cell_umap_coordinates_by_cancer_group.parquet",
    "crop_index": "inputs/crops_index.parquet",
    "warehouse": "warehouse",
    "source_hashes": "warehouse/jpg-sha256.txt.gz",
}
REQUIRED_FILES = frozenset(INPUTS[key] for key in ("umap", "crop_index", "source_hashes"))
MAX_EXPANDED_BYTES = 950_000_000
MAX_MEMBERS = 60_000
CHUNK = 1024 * 1024
DOWNLOAD_TIMEOUT = 30
DOWNLOAD_RETRIES = 2


def _identifier(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,255}", value):
        raise ValueError(f"Invalid {label}")
    return value


def _https_url(value):
    if not isinstance(value, str) or any(ord(c) <= 32 for c in value):
        raise ValueError("Archive URL must be HTTPS")
    parsed = urlsplit(value)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.fragment or "\\" in value):
        raise ValueError("Archive URL must be HTTPS without credentials or fragment")
    # Accessing port also rejects malformed/out-of-range port specifications.
    parsed.port
    return value


def _keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"Invalid {label} keys; expected {sorted(keys)}")


def load_config(path):
    """Validate the exact version-one schema, including local-archive workflows."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate config key: {key}")
            result[key] = value
        return result

    config = json.loads(Path(path).read_text(), object_pairs_hook=unique_object)
    _keys(config, {"schema_version", "plate", "run_id", "archive", "inputs", "expected"}, "config")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Unsupported schema_version")
    _identifier(config["plate"], "plate")
    _identifier(config["run_id"], "run_id")
    _keys(config["archive"], {"url", "sha256", "bytes"}, "archive")
    archive = config["archive"]
    _https_url(archive["url"])
    if not isinstance(archive["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", archive["sha256"]):
        raise ValueError("Archive sha256 must be 64 lowercase hex digits")
    if type(archive["bytes"]) is not int or not 0 < archive["bytes"] <= MAX_EXPANDED_BYTES:
        raise ValueError("Invalid archive bytes budget")
    if config["inputs"] != INPUTS:
        raise ValueError("Input paths must match the immutable pilot archive layout")
    _keys(config["expected"], {"cells", "assets"}, "expected")
    expected = config["expected"]
    if (any(type(expected[k]) is not int or expected[k] <= 0 for k in expected)
            or expected["assets"] != expected["cells"] * len(report.CHANNELS)
            or expected["assets"] + len(REQUIRED_FILES) > MAX_MEMBERS):
        raise ValueError("Expected counts must describe six unique assets per cell within member budget")
    return config


def _no_symlinks(path):
    """Reject symlink ancestors too, including a dangling output symlink."""
    path = Path(os.path.abspath(path))
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError(f"Unsafe symlink path: {candidate}")
    return path


def _regular(path, root=None):
    path = _no_symlinks(path)
    if root is not None and not path.is_relative_to(Path(root).resolve()):
        raise ValueError("Input file escapes its root")
    if not path.is_file():
        raise ValueError(f"Missing regular input file: {path.name}")
    return path


def _new_output(output):
    output = _no_symlinks(output)
    if output.exists():
        raise ValueError(f"Refusing to replace existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _no_symlinks(output)
    return output


def _publish_directory(source, output):
    """Atomically rename with no-replace semantics; never fall back to overwrite.

    macOS renamex_np(RENAME_EXCL) and Linux renameat2(RENAME_NOREPLACE)
    protect even against an empty directory appearing after the initial check.
    """
    _new_output(output)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        args = [os.fsencode(source), os.fsencode(output), 0x4]
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        args = [-100, os.fsencode(source), -100, os.fsencode(output), 1]
    else:
        raise OSError("Atomic no-replace directory publication requires macOS or Linux rename support")
    rename.restype = ctypes.c_int
    if rename(*args):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def _member_name(name):
    if (not isinstance(name, str) or not name or name.startswith("/")
            or "\\" in name or ":" in name or any(ord(c) < 32 or ord(c) == 127 for c in name)
            or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError(f"Unsafe archive member: {name!r}")
    if name in REQUIRED_FILES:
        return name
    if name.startswith("warehouse/crops/"):
        report.relative_crop_path(name.removeprefix("warehouse/"))
        return name
    raise ValueError(f"Unexpected archive member: {name}")


def _selection(umap, crop_index, warehouse, *, plate, run_id):
    payload = report.prepare_data(pd.read_parquet(umap), pd.read_parquet(crop_index),
                                  plate=plate, run_id=run_id)
    selected = {}
    for point in payload["points"]:
        if set(point["images"]) != set(report.CHANNELS):
            raise ValueError("Incomplete selected channels")
        for url in point["images"].values():
            relative = report.relative_crop_path(unquote(url.removeprefix("assets/")))
            if len(relative.parts) < 3 or relative.parts[1] != plate:
                raise ValueError("Selected JPEG is outside configured plate scope")
            name = _member_name(f"warehouse/{relative}")
            if name in selected:
                raise ValueError("Reused selected JPEG path")
            selected[name] = _regular(Path(warehouse) / relative, warehouse)
    return payload, selected


def _source_hashes(path, selected):
    """Read the ORIGINAL list, retaining only selected entries; reject gzip bombs."""
    needed = {name.removeprefix("warehouse/") for name in selected}
    hashes = {}
    total = 0
    with gzip.open(_regular(path), "rb") as stream:
        while True:
            line = stream.readline(16_385)
            if not line:
                break
            total += len(line)
            if len(line) > 16_384 or total > MAX_EXPANDED_BYTES:
                raise ValueError("Original source hash list exceeds expanded budget")
            match = re.fullmatch(rb"([0-9a-fA-F]{64}) [ *]([^\r\n]+)\n?", line)
            if not match:
                raise ValueError("Malformed original JPEG hash list")
            relative = match[2].decode("utf-8")
            report.relative_crop_path(relative)
            if relative in needed:
                if relative in hashes:
                    raise ValueError(f"Duplicate original JPEG hash: {relative}")
                hashes[relative] = match[1].decode("ascii").lower()
    if set(hashes) != needed:
        raise ValueError("Selected JPEG missing from original source hash list")
    return {f"warehouse/{name}": digest for name, digest in hashes.items()}


def _verify_jpegs(selected, source_hashes):
    hashes = _source_hashes(source_hashes, selected)
    for name, path in selected.items():
        if report.sha256(path) != hashes[name]:
            raise ValueError(f"Original JPEG hash mismatch: {name}")
    return hashes


def _read_exact(stream: gzip.GzipFile, count: int) -> bytes:
    value = stream.read(count)
    if len(value) != count:
        raise ValueError("Truncated gzip/tar archive")
    return value


def _read_archive(path, *, root=None, expected=None):
    """Read raw USTAR headers, never extractall or implicit PAX/GNU extensions.

    Reading headers ourselves prevents tarfile's transparent extension handling
    from allocating an unbounded PAX/longname payload before size/type checks.
    The complete gzip stream (including trailer/CRC) is consumed and checked.
    """
    members = {}
    expanded = 0
    with gzip.GzipFile(filename=path, mode="rb") as stream:
        while True:
            header = _read_exact(stream, 512)
            if header == bytes(512):
                if _read_exact(stream, 512) != bytes(512):
                    raise ValueError("Invalid tar end marker")
                tail_size = 1024
                while tail := stream.read(CHUNK):
                    tail_size += len(tail)
                    # Streaming tar writers can pad one additional record when
                    # their final two zero blocks straddle a record boundary.
                    if tail_size > 2 * tarfile.RECORDSIZE or any(tail):
                        raise ValueError("Unexpected trailing tar/gzip content")
                break
            info = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            name = _member_name(info.name)
            if not info.isreg() or info.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
                raise ValueError("Archive accepts regular files only (no links, directories or extensions)")
            if name in members:
                raise ValueError(f"Duplicate archive member: {name}")
            if len(members) >= MAX_MEMBERS:
                raise ValueError("Archive exceeds member count budget")
            expanded += info.size
            if info.size <= 0 or expanded > MAX_EXPANDED_BYTES:
                raise ValueError("Archive exceeds expanded bytes budget or has empty member")
            if expected is not None and name not in expected:
                raise ValueError(f"Unexpected read-back member: {name}")
            digest = hashlib.sha256()
            target = None
            try:
                if root is not None:
                    destination = Path(root) / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    target = destination.open("xb")
                remaining = info.size
                while remaining:
                    data = _read_exact(stream, min(CHUNK, remaining))
                    digest.update(data)
                    if target is not None:
                        target.write(data)
                    remaining -= len(data)
                if target is not None:
                    target.flush()
                padding = _read_exact(stream, (-info.size) % 512)
                if any(padding):
                    raise ValueError("Nonzero tar member padding")
            finally:
                if target is not None:
                    target.close()
            metadata = {"bytes": info.size, "sha256": digest.hexdigest()}
            if expected is not None and metadata != expected[name]:
                raise ValueError(f"Archive source-byte read-back mismatch: {name}")
            members[name] = metadata
    if not REQUIRED_FILES.issubset(members):
        raise ValueError("Archive lacks required original input members")
    if expected is not None and set(members) != set(expected):
        raise ValueError("Archive read-back member set mismatch")
    return members


def _public_parquet(path, columns, work):
    """Drop unused columns/attrs that could disclose private source filesystem paths."""
    frame = pd.read_parquet(path)
    omitted = sorted(set(frame.columns) - set(columns))
    if not omitted and not frame.attrs:
        return path, omitted
    public = frame.loc[:, columns].copy()
    public.attrs.clear()
    target = work / path.name
    public.to_parquet(target, index=False)
    return target, omitted


def package(*, umap, crop_index, warehouse, source_hashes, plate, run_id, output):
    """Package public build inputs, verifying original images and unchanged report values."""
    _identifier(plate, "plate")
    _identifier(run_id, "run_id")
    output = _new_output(output)
    sources = {INPUTS["umap"]: _regular(umap), INPUTS["crop_index"]: _regular(crop_index),
               INPUTS["source_hashes"]: _regular(source_hashes)}
    for name, path in sources.items():
        if path.name != Path(name).name:
            raise ValueError(f"Original input basename must be {Path(name).name}; got {path.name}")
    original = {name: {"bytes": path.stat().st_size, "sha256": report.sha256(path)}
                for name, path in sources.items()}
    payload, selected = _selection(umap, crop_index, warehouse, plate=plate, run_id=run_id)
    hashes = _verify_jpegs(selected, source_hashes)
    sources.update(selected)
    metadata = dict(original)
    metadata.update({name: {"bytes": path.stat().st_size, "sha256": hashes[name]}
                     for name, path in selected.items()})
    work = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    partial = work / "inputs.tar.gz"
    omitted = {}
    try:
        for name, expected in original.items():
            if report.sha256(sources[name]) != expected["sha256"]:
                raise ValueError(f"Source changed during packaging: {name}")
        # These are the exact report-required columns, retaining all rows and
        # unrounded values but never unused absolute TIFF source directories.
        common = report.POSITION_KEY + ["Metadata_cell_line", "Metadata_time_point"]
        columns = {
            "umap": common + ["cancer_group", "tissue_site", "umap_1", "umap_2", "cluster"],
            "crop_index": list(dict.fromkeys(common + report.OBJECT_KEY +
                           list(report.PATH_COLUMNS.values()) + ["Metadata_Nucleocentric_Crop_Size"])),
        }
        archive_sources = dict(sources)
        for key, required in columns.items():
            name = INPUTS[key]
            archive_sources[name], omitted[key] = _public_parquet(sources[name], required, work)
            path = archive_sources[name]
            metadata[name] = {"bytes": path.stat().st_size, "sha256": report.sha256(path)}
        checked = report.prepare_data(pd.read_parquet(archive_sources[INPUTS["umap"]]),
                                      pd.read_parquet(archive_sources[INPUTS["crop_index"]]),
                                      plate=plate, run_id=run_id)
        if checked != payload:
            raise ValueError("Public input projection changed report values or scope")
        if len(sources) > MAX_MEMBERS or sum(m["bytes"] for m in metadata.values()) > MAX_EXPANDED_BYTES:
            raise ValueError("Selected inputs exceed archive size/member budget")
        with partial.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode="w|", format=tarfile.USTAR_FORMAT,
                                  encoding="utf-8", errors="strict") as archive:
                    for name, path in sorted(archive_sources.items()):
                        info = tarfile.TarInfo(name)
                        info.size = metadata[name]["bytes"]
                        info.mode = 0o644
                        info.mtime = info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        with path.open("rb") as stream:
                            archive.addfile(info, stream)
            raw.flush()
            os.fsync(raw.fileno())
        archive_sha = report.sha256(partial)
        archive_bytes = partial.stat().st_size
        if archive_bytes > MAX_EXPANDED_BYTES:
            raise ValueError("Compressed archive exceeds size budget")
        _read_archive(partial, expected=metadata)
        for name, path in sources.items():
            expected = original[name] if name in original else metadata[name]
            if report.sha256(path) != expected["sha256"]:
                raise ValueError(f"Source changed during packaging: {name}")
        if report.sha256(partial) != archive_sha:
            raise ValueError("Full archive checksum read-back mismatch")
        _no_symlinks(output)
        os.link(partial, output)
    finally:
        shutil.rmtree(work)
    return {"schema_version": 1, "plate": plate, "run_id": run_id,
            "archive": {"sha256": archive_sha, "bytes": archive_bytes},
            "inputs": dict(INPUTS), "expected": {"cells": len(payload["points"]), "assets": len(selected)},
            "members": len(metadata), "expanded_bytes": sum(m["bytes"] for m in metadata.values()),
            "summary": payload["summary"], "sources": original,
            "archived_sources": {name: metadata[name] for name in original}, "omitted_columns": omitted}


class _HTTPSRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _https_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _copy_pinned(stream, target, pin):
    digest = hashlib.sha256()
    count = 0
    with target.open("xb") as output:
        while data := stream.read(min(CHUNK, pin["bytes"] - count + 1)):
            count += len(data)
            if count > pin["bytes"]:
                raise ValueError("Archive exceeds pinned bytes budget")
            digest.update(data)
            output.write(data)
    if count != pin["bytes"]:
        raise ValueError("Archive bytes differ from pinned size")
    if digest.hexdigest() != pin["sha256"]:
        raise ValueError("Archive sha256 checksum mismatch")


def _acquire_archive(config, target, local):
    pin = config["archive"]
    if local is not None:
        with _regular(local).open("rb") as source:
            _copy_pinned(source, target, pin)
        return
    opener = build_opener(_HTTPSRedirects())
    for attempt in range(DOWNLOAD_RETRIES + 1):
        try:
            request = Request(pin["url"], headers={"Accept-Encoding": "identity", "User-Agent": "pccma-pages/1"})
            with opener.open(request, timeout=DOWNLOAD_TIMEOUT) as response:
                _https_url(response.geturl())
                _copy_pinned(response, target, pin)
            return
        except (OSError, URLError, http.client.HTTPException):
            target.unlink(missing_ok=True)
            if attempt == DOWNLOAD_RETRIES:
                raise


def stage(*, config, output, archive=None):
    """Download/copy a pinned archive, hash before extraction, publish a new root."""
    config = load_config(config)
    output = _new_output(output)
    work = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        packed = work / "download.tar.gz"
        _acquire_archive(config, packed, archive)
        extracted = work / "extracted"
        extracted.mkdir()
        members = _read_archive(packed, root=extracted)
        if len(members) - len(REQUIRED_FILES) != config["expected"]["assets"]:
            raise ValueError("Extracted asset count differs from expected count")
        _publish_directory(extracted, output)
    finally:
        shutil.rmtree(work)
    return {"output": str(output), "archive": config["archive"], "members": len(members),
            "expanded_bytes": sum(m["bytes"] for m in members.values())}


def _input_tree(root):
    root = _no_symlinks(root)
    if not root.is_dir():
        raise ValueError("Extracted inputs root is missing")
    files = set()
    expanded = 0
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            _no_symlinks(Path(directory) / name)
        for name in names:
            path = _regular(Path(directory) / name, root)
            files.add(_member_name(path.relative_to(root).as_posix()))
            expanded += path.stat().st_size
            if len(files) > MAX_MEMBERS or expanded > MAX_EXPANDED_BYTES:
                raise ValueError("Extracted inputs exceed size/member budget")
    if not REQUIRED_FILES.issubset(files):
        raise ValueError("Extracted inputs lack required original input members")
    return root, files


def build(*, config, inputs, output):
    """Check exact selected scope and original hashes, then build current web code."""
    config = load_config(config)
    output = _new_output(output)
    root, files = _input_tree(inputs)
    umap, crop_index = root / INPUTS["umap"], root / INPUTS["crop_index"]
    source_hashes = root / INPUTS["source_hashes"]
    original = {name: report.sha256(root / name) for name in REQUIRED_FILES}
    payload, selected = _selection(umap, crop_index, root / INPUTS["warehouse"],
                                   plate=config["plate"], run_id=config["run_id"])
    counts = {"cells": len(payload["points"]), "assets": len(selected)}
    if counts != config["expected"]:
        raise ValueError(f"Prepared counts differ from expected counts: {counts} != {config['expected']}")
    if files != REQUIRED_FILES | set(selected):
        raise ValueError("Extracted JPEG member set differs from selected plotted pilot scope")
    hashes = _verify_jpegs(selected, source_hashes)
    work = Path(tempfile.mkdtemp(prefix=f".{output.name}.partial-", dir=output.parent))
    try:
        site = work / "site"
        built = report.build_report(umap_path=umap, index_path=crop_index,
                                    warehouse=root / INPUTS["warehouse"], output=site,
                                    plate=config["plate"], run_id=config["run_id"])
        if built["points"] != payload["points"] or built["summary"] != payload["summary"]:
            raise ValueError("Prepared scope changed during report build")
        manifest = json.loads((site / "manifest.json").read_text())
        copied = {"warehouse/" + a["path"].removeprefix("assets/"): a["sha256"] for a in manifest["assets"]}
        if copied != hashes or built["images"]["files"] != counts["assets"]:
            raise ValueError("Built JPEGs differ from original source hashes")
        for name, digest in original.items():
            if report.sha256(root / name) != digest:
                raise ValueError("Original inputs changed during build")
        for label in ("umap", "crop_index"):
            if built["sources"][label]["sha256"] != original[INPUTS[label]]:
                raise ValueError("Original input source hash was not preserved")
        _publish_directory(site, output)
    finally:
        shutil.rmtree(work)
    return {"output": str(output), "plate": config["plate"], "run_id": config["run_id"],
            "summary": built["summary"], "images": built["images"], "catalog": built["catalog"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packing = commands.add_parser("package", help="Package only original plotted pilot inputs")
    for option in ("umap", "crop-index", "warehouse", "source-hashes", "output"):
        packing.add_argument(f"--{option}", type=Path, required=True)
    for option in ("plate", "run-id"):
        packing.add_argument(f"--{option}", required=True)
    staging = commands.add_parser("stage", help="Verify a pinned archive before safe extraction")
    staging.add_argument("--config", type=Path, required=True)
    staging.add_argument("--output", type=Path, required=True)
    staging.add_argument("--archive", type=Path, help="Local archive; bypass network, never checksums")
    building = commands.add_parser("build", help="Build current report code from staged original inputs")
    building.add_argument("--config", type=Path, required=True)
    building.add_argument("--inputs", type=Path, required=True)
    building.add_argument("--output", type=Path, required=True)
    arguments = vars(parser.parse_args(argv))
    command = arguments.pop("command")
    try:
        result = {"package": package, "stage": stage, "build": build}[command](**arguments)
    except (ValueError, OSError, EOFError, tarfile.TarError, KeyError, TypeError,
            http.client.HTTPException, report.duckdb.Error) as error:
        parser.exit(1, f"Pages {command} failed: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
