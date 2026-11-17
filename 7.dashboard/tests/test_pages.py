"""Fixture-only CLI and hostile-input checks for immutable Pages pilot inputs."""

import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from urllib.error import URLError

import duckdb
import pandas as pd
from PIL import Image
import pytest

DASHBOARD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DASHBOARD))
SPEC = importlib.util.spec_from_file_location("pilot_pages", DASHBOARD / "pages.py")
assert SPEC is not None and SPEC.loader is not None
pages = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pages)
PLATE = "BR00150695"
RUN = "fixture-original-run"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def pilot(tmp_path):
    source = tmp_path / "original inputs"
    source.mkdir()
    warehouse = source / "warehouse"
    base = {
        "Metadata_Plate": PLATE, "Metadata_Well": "C03", "Metadata_Site": 1,
        "Metadata_Nuclei_Location_Center_X": 12.1234567890123,
        "Metadata_Nuclei_Location_Center_Y": 24.5, "Metadata_cell_line": "DAOY",
        "Metadata_time_point": 24,
    }
    umap = pd.DataFrame([
        dict(base, cancer_group="Medulloblastoma", tissue_site="CNS", umap_1=1.25, umap_2=-2.5, cluster=2),
        dict(base, Metadata_Plate="OTHER", cancer_group="Other", tissue_site="Other", umap_1=99., umap_2=99., cluster=1),
    ])
    row = dict(base, Metadata_ImageNumber=7, Metadata_Nuclei_Number_Object_Number=3,
               Metadata_Cells_Number_Object_Number=2, Metadata_Nucleocentric_Crop_Size=16)
    for channel, column in pages.report.PATH_COLUMNS.items():
        row[column] = f"crops/{PLATE}/DAOY/C03/f01/{PLATE}_C03_f01_nuc0003_cell0002_{channel}.jpg"
        image = warehouse / row[column]
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB" if channel == "composite" else "L", (16, 16), 80).save(image, "JPEG")
    # Full index includes an unplotted cell whose absent JPEGs must not be needed.
    unplotted = dict(row, Metadata_Nuclei_Location_Center_X=900.,
                     Metadata_Nuclei_Number_Object_Number=9, Metadata_Cells_Number_Object_Number=8)
    for column in pages.report.PATH_COLUMNS.values():
        unplotted[column] = row[column].replace("nuc0003_cell0002", "nuc0009_cell0008")
    index = pd.DataFrame([row, unplotted])
    umap_path = source / Path(pages.INPUTS["umap"]).name
    index_path = source / Path(pages.INPUTS["crop_index"]).name
    umap.to_parquet(umap_path)
    index.to_parquet(index_path)
    hashes = warehouse / "jpg-sha256.txt.gz"
    lines = [f"{digest(warehouse / row[column])}  {row[column]}\n"
             for column in pages.report.PATH_COLUMNS.values()]
    # Preserve unrelated original hashes verbatim, not a rewritten subset list.
    lines.append("0" * 64 + "  crops/OTHER/not-packaged.jpg\n")
    hashes.write_bytes(gzip.compress("".join(lines).encode(), mtime=12345))
    # Old generated results are irrelevant and must never enter the archive.
    (warehouse / "old-index.html").write_text("do not publish")
    (warehouse / "old.duckdb").write_bytes(b"not an input")
    return {"umap": umap_path, "crop_index": index_path, "warehouse": warehouse,
            "source_hashes": hashes, "plate": PLATE, "run_id": RUN, "output": tmp_path / "pilot-inputs.tar.gz"}


def make_config(tmp_path, archive, *, expected=None):
    config = {
        "schema_version": 1, "plate": PLATE, "run_id": RUN,
        "archive": {"url": "https://example.invalid/pilot-inputs.tar.gz", "sha256": digest(archive),
                    "bytes": archive.stat().st_size},
        "inputs": dict(pages.INPUTS), "expected": expected or {"cells": 1, "assets": 6},
    }
    path = tmp_path / "pages-config.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def packed(pilot, tmp_path):
    descriptor = pages.package(**pilot)
    config = make_config(tmp_path, pilot["output"])
    return pilot, descriptor, config


@pytest.fixture
def staged(packed, tmp_path):
    pilot, descriptor, config = packed
    root = tmp_path / "extracted"
    pages.stage(config=config, output=root, archive=pilot["output"])
    return pilot, descriptor, config, root


def run_cli(module, arguments, cwd):
    result = subprocess.run([sys.executable, str(module), *map(str, arguments)], cwd=cwd,
                            env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert not result.stderr
    return json.loads(result.stdout)


def test_real_cli_package_stage_build_uses_current_web_and_original_sources(pilot, tmp_path):
    descriptor = run_cli(DASHBOARD / "pages.py", ["package", *sum(
        (["--" + key.replace("_", "-"), value] for key, value in pilot.items()), [])], tmp_path)
    assert descriptor["expected"] == {"cells": 1, "assets": 6}
    assert descriptor["members"] == 9
    assert descriptor["archive"] == {"sha256": digest(pilot["output"]), "bytes": pilot["output"].stat().st_size}
    assert descriptor["inputs"] == pages.INPUTS
    assert descriptor["summary"]["excluded_umap_rows"] == 1
    assert descriptor["summary"]["warehouse_cells"] == 2
    config = make_config(tmp_path, pilot["output"])
    extracted = tmp_path / "staged inputs"
    stage_metadata = run_cli(DASHBOARD / "pages.py", ["stage", "--config", config,
                             "--archive", pilot["output"], "--output", extracted], tmp_path)
    assert stage_metadata["members"] == 9
    for key in ("umap", "crop_index", "source_hashes"):
        assert (extracted / pages.INPUTS[key]).read_bytes() == pilot[key].read_bytes()
    # Build from a fresh copy of current source; modify web AFTER packaging.
    # No writes to the repository's web files or real distribution directories.
    checkout = tmp_path / "current checkout"
    checkout.mkdir()
    for name in ("pages.py", "build_report.py"):
        shutil.copyfile(DASHBOARD / name, checkout / name)
    shutil.copytree(DASHBOARD / "web", checkout / "web")
    marker = "/* latest-source-after-input-packaging */"
    with (checkout / "web/app.js").open("a") as stream:
        stream.write("\n" + marker + "\n")
    site = tmp_path / "new site"
    result = run_cli(checkout / "pages.py", ["build", "--config", config,
                     "--inputs", extracted, "--output", site], tmp_path)
    assert marker in (site / "index.html").read_text()
    assert result["summary"] == descriptor["summary"]
    assert result["images"]["files"] == 6
    assert result["catalog"]["status"] == "bundled"
    manifest = json.loads((site / "manifest.json").read_text())
    assert manifest["code_sha256"]["app.js"] == digest(checkout / "web/app.js")
    for label in ("umap", "crop_index"):
        assert manifest["sources"][label] == {"name": pilot[label].name, "sha256": digest(pilot[label])}
    for asset in manifest["assets"]:
        original = pilot["warehouse"] / asset["path"].removeprefix("assets/")
        assert (site / asset["path"]).read_bytes() == original.read_bytes()
        assert asset["sha256"] == digest(original)
    assert (site / ".nojekyll").is_file()
    with duckdb.connect(str(site / "data/pilot.duckdb"), read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM cells").fetchall() == [(1,)]
        assert connection.execute("SELECT count(*) FROM image_assets").fetchall() == [(6,)]
        assert connection.execute("SELECT DISTINCT plate, run_id FROM cells").fetchall() == [(PLATE, RUN)]
        summary_row = connection.execute("SELECT value FROM report_metadata WHERE key='summary'").fetchone()
        assert summary_row is not None
        summary = json.loads(summary_row[0])
        assert summary == descriptor["summary"]
    assert str(pilot["warehouse"]) not in (site / "index.html").read_text()


@pytest.mark.parametrize("private_field", ["columns", "attrs", "both"])
def test_public_snapshot_omits_unused_private_paths_without_changing_report(pilot, tmp_path, private_field):
    private = "/private/scratch/example-person@example.invalid/source-images"
    original = {}
    for key in ("umap", "crop_index"):
        frame = pd.read_parquet(pilot[key])
        if private_field in ("columns", "both"):
            frame["Unused_Source_Path"] = private
        if private_field in ("attrs", "both"):
            frame.attrs["source_directory"] = private
        frame.to_parquet(pilot[key])
        original[key] = digest(pilot[key])
    descriptor = pages.package(**pilot)
    config = make_config(tmp_path, pilot["output"])
    extracted = tmp_path / "public-inputs"
    pages.stage(config=config, output=extracted, archive=pilot["output"])
    for key in ("umap", "crop_index"):
        public_path = extracted / pages.INPUTS[key]
        public = pd.read_parquet(public_path)
        assert "Unused_Source_Path" not in public.columns
        assert not public.attrs
        assert private.encode() not in public_path.read_bytes()
        assert digest(pilot[key]) == original[key], "Original files must remain untouched"
        assert descriptor["sources"][pages.INPUTS[key]]["sha256"] == original[key]
        assert descriptor["archived_sources"][pages.INPUTS[key]]["sha256"] == digest(public_path)
        assert len(public) == len(pd.read_parquet(pilot[key]))
    site = tmp_path / "site"
    result = pages.build(config=config, inputs=extracted, output=site)
    assert result["summary"] == descriptor["summary"]
    assert result["images"]["files"] == 6
    assert private not in (site / "index.html").read_text()


def test_archive_is_deterministic_sorted_regular_only_and_byte_verified(packed, tmp_path):
    pilot, descriptor, _ = packed
    second = tmp_path / "second.tar.gz"
    again = pages.package(**dict(pilot, output=second))
    assert second.read_bytes() == pilot["output"].read_bytes()
    assert again == descriptor
    with tarfile.open(second, "r:gz") as archive:
        members = archive.getmembers()
        names = [m.name for m in members]
        assert names == sorted(names)
        assert len(names) == len(set(names)) == 9
        assert set(names) == pages.REQUIRED_FILES | {
            "warehouse/" + str(p.relative_to(pilot["warehouse"])) for p in pilot["warehouse"].rglob("*.jpg")}
        for member in members:
            assert member.isreg()
            assert member.mtime == member.uid == member.gid == 0
            assert member.uname == member.gname == ""
            assert member.mode == 0o644
            assert archive.extractfile(member).read() == (
                pilot[{pages.INPUTS["umap"]: "umap", pages.INPUTS["crop_index"]: "crop_index",
                       pages.INPUTS["source_hashes"]: "source_hashes"}[member.name]].read_bytes()
                if member.name in pages.REQUIRED_FILES else
                (pilot["warehouse"] / member.name.removeprefix("warehouse/")).read_bytes())
    # gzip must omit timestamp and filename, so moving output cannot change bytes.
    header = second.read_bytes()[:10]
    assert header[3] & 8 == 0
    assert header[4:8] == bytes(4)


@pytest.mark.parametrize("problem", ["checksum", "short", "long", "gzip_crc", "gzip_truncated"])
def test_stage_rejects_tampered_pins_or_gzip_and_cleans_partial(packed, tmp_path, problem):
    pilot, _, config_path = packed
    config = json.loads(config_path.read_text())
    archive = pilot["output"]
    data = archive.read_bytes()
    if problem == "checksum":
        config["archive"]["sha256"] = "0" * 64
    elif problem == "short":
        config["archive"]["bytes"] += 1
    elif problem == "long":
        config["archive"]["bytes"] -= 1
    else:
        archive.write_bytes(data[:-8] + bytes([data[-8] ^ 1]) + data[-7:]
                            if problem == "gzip_crc" else data[:-5])
        config["archive"].update(sha256=digest(archive), bytes=archive.stat().st_size)
    config_path.write_text(json.dumps(config))
    output = tmp_path / "no-stage"
    with pytest.raises((ValueError, OSError, EOFError)):
        pages.stage(config=config_path, output=output, archive=archive)
    assert not output.exists()
    assert not list(tmp_path.glob(".no-stage.partial-*"))


def hostile_archive(tmp_path, members):
    archive = tmp_path / "hostile.tar.gz"
    # Raw headers allow declaring enormous files without allocating their data.
    with gzip.open(archive, "wb") as stream:
        for info, body in members:
            stream.write(info.tobuf(format=tarfile.USTAR_FORMAT))
            stream.write(body)
            stream.write(bytes((-len(body)) % 512))
        stream.write(bytes(1024))
    return archive


def member(name, body=b"x", *, kind=tarfile.REGTYPE, size=None):
    info = tarfile.TarInfo(name)
    info.type = kind
    info.size = len(body) if size is None else size
    info.linkname = "../../escape"
    return info, body


@pytest.mark.parametrize("name", [
    "/absolute.jpg", "../escape.jpg", "warehouse/crops/../escape.jpg", "warehouse/./crops/file.jpg",
    "warehouse//crops/file.jpg", "warehouse/crops/x\\escape.jpg", "warehouse/crops/C:escape.jpg",
    "warehouse/crops/file.jpg/", "index.html", "warehouse/pilot.duckdb", "inputs/other.parquet",
    "warehouse/crops/not-jpeg.png", "warehouse/crops/with\nnewline.jpg",
])
def test_stage_rejects_unsafe_or_unapproved_member_names(tmp_path, name):
    archive = hostile_archive(tmp_path, [member(name)])
    config = make_config(tmp_path, archive)
    with pytest.raises(ValueError, match="Unsafe|Unexpected|Invalid relative"):
        pages.stage(config=config, output=tmp_path / "bad", archive=archive)
    assert not (tmp_path / "bad").exists()
    assert not (tmp_path / "escape.jpg").exists()
    assert not list(tmp_path.glob(".bad.partial-*"))


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE,
                                    tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.FIFOTYPE,
                                    tarfile.XHDTYPE, tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_SPARSE])
def test_stage_rejects_links_directories_devices_and_extended_headers(tmp_path, kind):
    archive = hostile_archive(tmp_path, [member("warehouse/crops/unsafe.jpg", kind=kind)])
    config = make_config(tmp_path, archive)
    with pytest.raises(ValueError, match="regular files only"):
        pages.stage(config=config, output=tmp_path / "bad", archive=archive)
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("problem", ["duplicate", "oversized", "missing_inputs", "truncated", "empty"])
def test_stage_bounds_members_and_requires_complete_input_layout(tmp_path, problem):
    name = "warehouse/crops/a.jpg"
    members = [member(name)]
    if problem == "duplicate":
        members *= 2
    elif problem == "oversized":
        members = [member(name, body=b"", size=pages.MAX_EXPANDED_BYTES + 1)]
    elif problem == "truncated":
        members = [member(name, body=b"", size=100_000)]
    elif problem == "empty":
        members = [member(name, body=b"")]
    archive = hostile_archive(tmp_path, members)
    config = make_config(tmp_path, archive)
    with pytest.raises((ValueError, OSError, EOFError)):
        pages.stage(config=config, output=tmp_path / "bad", archive=archive)
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".bad.partial-*"))


def test_stage_bounds_aggregate_size_and_member_count(packed, tmp_path, monkeypatch):
    pilot, _, config = packed
    # Keep configured six assets valid while reducing read-time member budget.
    original_load = pages.load_config
    validated = original_load(config)
    monkeypatch.setattr(pages, "load_config", lambda _: validated)
    monkeypatch.setattr(pages, "MAX_MEMBERS", 2)
    with pytest.raises(ValueError, match="member count budget"):
        pages.stage(config=config, output=tmp_path / "bad-count", archive=pilot["output"])
    monkeypatch.setattr(pages, "MAX_MEMBERS", 60_000)
    monkeypatch.setattr(pages, "MAX_EXPANDED_BYTES", 1000)
    with pytest.raises(ValueError, match="expanded bytes budget"):
        pages.stage(config=config, output=tmp_path / "bad-size", archive=pilot["output"])
    assert not (tmp_path / "bad-count").exists()
    assert not (tmp_path / "bad-size").exists()


@pytest.mark.parametrize("problem", ["missing", "hash_mismatch", "hash_missing", "hash_duplicate", "symlink", "parent_symlink"])
def test_package_rejects_missing_unsafe_or_untrusted_original_jpegs(pilot, problem, tmp_path):
    image = next(pilot["warehouse"].rglob("*.jpg"))
    relative = image.relative_to(pilot["warehouse"]).as_posix()
    hashes = gzip.decompress(pilot["source_hashes"].read_bytes()).decode()
    if problem == "missing":
        image.unlink()
    elif problem == "hash_mismatch":
        Image.new("L", (16, 16), 23).save(image, "JPEG")
    elif problem == "hash_missing":
        pilot["source_hashes"].write_bytes(gzip.compress("".join(
            line + "\n" for line in hashes.splitlines() if not line.endswith(relative)).encode()))
    elif problem == "hash_duplicate":
        pilot["source_hashes"].write_bytes(gzip.compress((hashes + f"{digest(image)}  {relative}\n").encode()))
    elif problem == "symlink":
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(image.read_bytes())
        image.unlink()
        image.symlink_to(outside)
    else:
        parent = image.parent
        moved = parent.with_name("real-images")
        parent.rename(moved)
        parent.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="Missing|hash|symlink"):
        pages.package(**pilot)
    assert not pilot["output"].exists()
    assert not list(tmp_path.glob(".pilot-inputs.tar.gz.partial-*"))


def test_package_enforces_original_basenames(pilot):
    wrong = pilot["umap"].with_name("renamed.parquet")
    shutil.copyfile(pilot["umap"], wrong)
    with pytest.raises(ValueError, match="Original input basename"):
        pages.package(**dict(pilot, umap=wrong))


@pytest.mark.parametrize("problem", ["wrong_expected", "wrong_plate", "missing", "hash_mismatch", "extra_image", "symlink"])
def test_build_fails_before_publication_for_count_scope_or_hash_problems(staged, tmp_path, problem):
    _, _, config_path, root = staged
    config = json.loads(config_path.read_text())
    image = next((root / "warehouse/crops").rglob("*.jpg"))
    if problem == "wrong_expected":
        config["expected"] = {"cells": 2, "assets": 12}
    elif problem == "wrong_plate":
        config["plate"] = "WRONG"
    elif problem == "missing":
        image.unlink()
    elif problem == "hash_mismatch":
        image.write_bytes(image.read_bytes() + b"changed")
    elif problem == "extra_image":
        (image.parent / "extra.jpg").write_bytes(image.read_bytes())
    else:
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(image.read_bytes())
        image.unlink()
        image.symlink_to(outside)
    config_path.write_text(json.dumps(config))
    output = tmp_path / "bad-site"
    with pytest.raises(ValueError):
        pages.build(config=config_path, inputs=root, output=output)
    assert not output.exists()
    assert not list(tmp_path.glob(".bad-site.partial-*"))


@pytest.mark.parametrize("command", ["package", "stage", "build"])
@pytest.mark.parametrize("existing", ["file", "directory", "dangling_symlink", "parent_symlink"])
def test_all_commands_refuse_existing_or_symlink_outputs(staged, tmp_path, command, existing):
    pilot, _, config, root = staged
    output = tmp_path / "keep"
    if existing == "file":
        output.write_text("untouched")
    elif existing == "directory":
        output.mkdir()
        (output / "sentinel").write_text("untouched")
    elif existing == "dangling_symlink":
        output.symlink_to(tmp_path / "missing-target")
    else:
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        output.symlink_to(real_parent, target_is_directory=True)
        output = output / "new-output"
    with pytest.raises(ValueError, match="Refusing|symlink"):
        if command == "package":
            pages.package(**dict(pilot, output=output))
        elif command == "stage":
            pages.stage(config=config, output=output, archive=pilot["output"])
        else:
            pages.build(config=config, inputs=root, output=output)
    if existing == "file":
        assert output.read_text() == "untouched"
    elif existing == "directory":
        assert (output / "sentinel").read_text() == "untouched"


@pytest.mark.parametrize("mutation", ["schema", "unknown", "bool_size", "http", "file_url", "credentials",
                                       "wrong_path", "zero_cells", "not_six", "bad_sha", "fragment"])
def test_exact_config_schema_rejects_unsupported_inputs(packed, mutation):
    _, _, path = packed
    config = json.loads(path.read_text())
    if mutation == "schema":
        config["schema_version"] = True
    elif mutation == "unknown":
        config["private_cache"] = "/not/allowed"
    elif mutation == "bool_size":
        config["archive"]["bytes"] = True
    elif mutation == "http":
        config["archive"]["url"] = "http://example.invalid/input.tar.gz"
    elif mutation == "file_url":
        config["archive"]["url"] = "file:///input.tar.gz"
    elif mutation == "credentials":
        config["archive"]["url"] = "https://user:secret@example.invalid/input.tar.gz"
    elif mutation == "wrong_path":
        config["inputs"]["umap"] = "../umap.parquet"
    elif mutation == "zero_cells":
        config["expected"] = {"cells": 0, "assets": 0}
    elif mutation == "not_six":
        config["expected"]["assets"] = 5
    elif mutation == "bad_sha":
        config["archive"]["sha256"] = "a" * 63
    else:
        config["archive"]["url"] += "#fragment"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        pages.load_config(path)


def test_duplicate_json_keys_are_rejected(packed):
    _, _, path = packed
    path.write_text(path.read_text().replace('"schema_version": 1', '"schema_version": 1, "schema_version": 1'))
    with pytest.raises(ValueError, match="Duplicate config key"):
        pages.load_config(path)


def test_streamed_network_download_retries_twice_and_checks_redirects(packed, tmp_path, monkeypatch):
    pilot, _, config = packed
    calls = []
    reads = []

    class Response(io.BytesIO):
        def geturl(self):
            return "https://release-assets.example.invalid/immutable-inputs.tar.gz"

        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    class Opener:
        def open(self, request, *, timeout):
            calls.append((request.full_url, timeout, request.get_header("Accept-encoding")))
            if len(calls) < 3:
                raise URLError("fixture transient failure")
            return Response(pilot["output"].read_bytes())

    monkeypatch.setattr(pages, "build_opener", lambda handler: Opener())
    result = pages.stage(config=config, output=tmp_path / "network-stage")
    assert result["members"] == 9
    assert len(calls) == 3
    assert all(call == ("https://example.invalid/pilot-inputs.tar.gz", 30, "identity") for call in calls)
    assert reads and all(0 < size <= pages.CHUNK for size in reads)
    with pytest.raises(ValueError, match="HTTPS"):
        pages._HTTPSRedirects().redirect_request(None, None, 302, "", {}, "http://unsafe.example/input")


def test_failed_network_exhausts_retries_and_cleans_staging(packed, tmp_path, monkeypatch):
    _, _, config = packed
    calls = []

    class Opener:
        def open(self, *args, **kwargs):
            calls.append(1)
            raise URLError("fixture offline")

    monkeypatch.setattr(pages, "build_opener", lambda _: Opener())
    with pytest.raises(URLError):
        pages.stage(config=config, output=tmp_path / "offline")
    assert len(calls) == 3
    assert not (tmp_path / "offline").exists()
    assert not list(tmp_path.glob(".offline.partial-*"))


def test_cli_errors_are_nonzero_and_do_not_emit_success_json(packed, tmp_path):
    pilot, _, config = packed
    value = json.loads(config.read_text())
    value["archive"]["sha256"] = "0" * 64
    config.write_text(json.dumps(value))
    result = subprocess.run([sys.executable, str(DASHBOARD / "pages.py"), "stage", "--config", str(config),
                             "--archive", str(pilot["output"]), "--output", str(tmp_path / "bad")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 1
    assert "sha256 checksum mismatch" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


def test_archive_source_byte_readback_precedes_publication(pilot, monkeypatch, tmp_path):
    real_readback = pages._read_archive

    def corrupt(path, **kwargs):
        expected = dict(kwargs["expected"])
        name = next(iter(expected))
        expected[name] = dict(expected[name], sha256="0" * 64)
        return real_readback(path, expected=expected)

    monkeypatch.setattr(pages, "_read_archive", corrupt)
    with pytest.raises(ValueError, match="source-byte read-back mismatch"):
        pages.package(**pilot)
    assert not pilot["output"].exists()
    assert not list(tmp_path.glob(".pilot-inputs.tar.gz.partial-*"))


def test_atomic_directory_publish_cannot_replace_empty_directory(tmp_path, monkeypatch):
    source = tmp_path / "partial"
    source.mkdir()
    (source / "sentinel").write_text("new")
    output = tmp_path / "existing"
    output.mkdir()
    # Bypass the preliminary existence check to exercise the OS no-replace
    # primitive itself, as if another writer won the publication race.
    monkeypatch.setattr(pages, "_new_output", lambda value: Path(value))
    with pytest.raises(OSError):
        pages._publish_directory(source, output)
    assert not list(output.iterdir())
    assert (source / "sentinel").read_text() == "new"


@pytest.mark.parametrize("size", [6657, 7169, 16897])
def test_valid_streaming_tar_record_boundary_padding_is_accepted(tmp_path, size):
    path = tmp_path / "record-boundary.tar.gz"
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            with tarfile.open(fileobj=stream, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
                for index, name in enumerate(sorted(pages.REQUIRED_FILES)):
                    data = b"x" * (size if index == 0 else 1)
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
    assert set(pages._read_archive(path)) == pages.REQUIRED_FILES


@pytest.mark.parametrize("problem", ["negative_size", "header_checksum", "trailing_data", "concatenated_gzip"])
def test_stage_checks_raw_headers_and_complete_gzip_stream(tmp_path, problem):
    info, body = member("warehouse/crops/a.jpg")
    if problem == "negative_size":
        info.size = -1
        body = b""
    header = bytearray(info.tobuf(format=tarfile.GNU_FORMAT))
    if problem == "header_checksum":
        header[0] ^= 1
    uncompressed = bytes(header) + body + bytes((-len(body)) % 512) + bytes(1024)
    if problem == "trailing_data":
        uncompressed += b"not part of the tar"
    compressed = gzip.compress(uncompressed)
    if problem == "concatenated_gzip":
        compressed += gzip.compress(b"hidden second stream")
    archive = tmp_path / "invalid-header.tar.gz"
    archive.write_bytes(compressed)
    config = make_config(tmp_path, archive)
    with pytest.raises((ValueError, OSError, tarfile.TarError)):
        pages.stage(config=config, output=tmp_path / "bad", archive=archive)
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".bad.partial-*"))


def test_stage_checksum_is_verified_before_extraction(packed, tmp_path, monkeypatch):
    pilot, _, path = packed
    config = json.loads(path.read_text())
    config["archive"]["sha256"] = "0" * 64
    path.write_text(json.dumps(config))

    def must_not_extract(*args, **kwargs):
        pytest.fail("Extraction started before archive checksum validation")

    monkeypatch.setattr(pages, "_read_archive", must_not_extract)
    with pytest.raises(ValueError, match="sha256 checksum mismatch"):
        pages.stage(config=path, output=tmp_path / "bad", archive=pilot["output"])
    assert not (tmp_path / "bad").exists()


def test_package_detects_original_input_mutation_before_publish(pilot, monkeypatch):
    selection = pages._selection

    def mutate_after_read(*args, **kwargs):
        result = selection(*args, **kwargs)
        with pilot["umap"].open("ab") as stream:
            stream.write(b"source mutation after selecting original rows")
        return result

    monkeypatch.setattr(pages, "_selection", mutate_after_read)
    with pytest.raises(ValueError, match="Source changed during packaging"):
        pages.package(**pilot)
    assert not pilot["output"].exists()


def test_package_atomic_publish_loses_race_without_overwrite(pilot, monkeypatch):
    link = os.link

    def racing_link(source, target):
        Path(target).write_bytes(b"concurrent writer wins")
        link(source, target)

    monkeypatch.setattr(pages.os, "link", racing_link)
    with pytest.raises(FileExistsError):
        pages.package(**pilot)
    assert pilot["output"].read_bytes() == b"concurrent writer wins"


def test_download_retries_after_partial_stream_without_reusing_bytes(packed, tmp_path, monkeypatch):
    pilot, _, config = packed
    attempts = []
    archive_bytes = pilot["output"].read_bytes()

    class Response(io.BytesIO):
        def __init__(self, fail):
            super().__init__(archive_bytes)
            self.fail = fail

        def geturl(self):
            return "https://example.invalid/pilot-inputs.tar.gz"

        def read(self, size=-1):
            if self.fail and self.tell():
                raise TimeoutError("fixture midstream timeout")
            return super().read(min(size, 17))

    class Opener:
        def open(self, *args, **kwargs):
            attempts.append(1)
            return Response(fail=len(attempts) < 3)

    monkeypatch.setattr(pages, "build_opener", lambda _: Opener())
    output = tmp_path / "retried-stage"
    result = pages.stage(config=config, output=output)
    assert len(attempts) == 3
    assert result["members"] == 9
    assert (output / pages.INPUTS["umap"]).read_bytes() == pilot["umap"].read_bytes()
