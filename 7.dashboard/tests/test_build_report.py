"""Tests for the static report's exact cell-to-crop join."""

import importlib.util
import hashlib
import json
import re
from pathlib import Path

from PIL import Image
import pandas as pd
import pytest

MODULE = Path(__file__).parents[1] / "build_report.py"


def builder():
    assert MODULE.is_file(), "The static dashboard exporter has not been implemented"
    spec = importlib.util.spec_from_file_location("build_report", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tables():
    umap = pd.DataFrame([
        {"Metadata_Plate": "P1", "Metadata_Well": "C03", "Metadata_Site": 1,
         "Metadata_Nuclei_Location_Center_X": 12.1234567890123,
         "Metadata_Nuclei_Location_Center_Y": 24.5, "Metadata_cell_line": "DAOY",
         "Metadata_time_point": 24, "cancer_group": "Medulloblastoma",
         "tissue_site": "Central nervous system", "umap_1": 1.25, "umap_2": -2.5,
         "cluster": 2},
        {"Metadata_Plate": "P2", "Metadata_Well": "C03", "Metadata_Site": 1,
         "Metadata_Nuclei_Location_Center_X": 90.,
         "Metadata_Nuclei_Location_Center_Y": 90., "Metadata_cell_line": "OTHER",
         "Metadata_time_point": 48, "cancer_group": "Other", "tissue_site": "Other",
         "umap_1": 999., "umap_2": 999., "cluster": 1},
    ])
    index = umap.iloc[[0]].drop(columns=["umap_1", "umap_2", "cluster", "cancer_group", "tissue_site"]).copy()
    index["Metadata_ImageNumber"] = 7
    index["Metadata_Nuclei_Number_Object_Number"] = 3
    index["Metadata_Cells_Number_Object_Number"] = 2
    index["Metadata_Nucleocentric_Crop_Size"] = 224
    for channel in ["DNA", "ER", "RNA", "AGP", "Mito", "composite"]:
        column = channel if channel == "composite" else f"grayscale_{channel}"
        index[column] = f"crops/P1/DAOY/C03/f01/P1_C03_f01_nuc0003_cell0002_{channel}.jpg"
    return umap, index


def test_exact_join_scopes_plate_and_preserves_coordinates(tables):
    umap, index = tables
    report = builder().prepare_data(umap, index, plate="P1", run_id="pilot")
    assert report["summary"]["umap_cells"] == 1
    assert report["summary"]["excluded_umap_rows"] == 1
    point = report["points"][0]
    assert point["cx"] == 12.1234567890123
    assert point["id"] == "P1_C03_f01_nuc0003_cell0002"
    assert (point["x"], point["y"], point["line"], point["cluster"]) == (1.25, -2.5, "DAOY", 2)
    assert point["images"]["DNA"] == "assets/" + index.iloc[0].grayscale_DNA


@pytest.mark.parametrize("problem", [
    "missing_column", "null_key", "duplicate_index", "duplicate_umap", "missing_match",
    "rounded_coordinate", "nonfinite", "line_mismatch", "time_mismatch", "empty_plate",
    "absolute_path", "traversal", "url_path", "backslash", "duplicate_object",
])
def test_invalid_join_inputs_fail_closed(tables, problem):
    umap, index = tables
    if problem == "missing_column":
        umap = umap.drop(columns="cancer_group")
    elif problem == "null_key":
        index.loc[0, "Metadata_Well"] = None
    elif problem == "duplicate_index":
        index = pd.concat([index, index], ignore_index=True)
    elif problem == "duplicate_umap":
        umap = pd.concat([umap, umap.iloc[[0]]], ignore_index=True)
    elif problem == "missing_match":
        index.loc[0, "Metadata_Well"] = "C04"
    elif problem == "rounded_coordinate":
        index.loc[0, "Metadata_Nuclei_Location_Center_X"] = 12.123456789
    elif problem == "nonfinite":
        umap.loc[0, "umap_1"] = float("inf")
    elif problem == "line_mismatch":
        index.loc[0, "Metadata_cell_line"] = "WRONG"
    elif problem == "time_mismatch":
        index.loc[0, "Metadata_time_point"] = 48
    elif problem == "empty_plate":
        umap = umap.iloc[[1]]
    elif problem == "duplicate_object":
        second = index.copy()
        second["Metadata_Nuclei_Location_Center_X"] = 91.
        index = pd.concat([index, second], ignore_index=True)
    else:
        index.loc[0, "composite"] = {
            "absolute_path": "/etc/passwd", "traversal": "crops/../../private.jpg",
            "url_path": "https://other.example/crop.jpg", "backslash": "crops\\\\private.jpg",
        }[problem]
    with pytest.raises(ValueError):
        builder().prepare_data(umap, index, plate="P1", run_id="pilot")


@pytest.fixture
def build_inputs(tables, tmp_path):
    umap, index = tables
    umap_path, index_path = tmp_path / "umap.parquet", tmp_path / "index.parquet"
    umap.to_parquet(umap_path)
    index.to_parquet(index_path)
    warehouse = tmp_path / "warehouse"
    for col in ["composite"] + [f"grayscale_{ch}" for ch in ["DNA", "ER", "RNA", "AGP", "Mito"]]:
        path = warehouse / index.iloc[0][col]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB" if col == "composite" else "L", (224, 224), 80).save(path, "JPEG")
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    (web_dir / "template.html").write_text(
        '<!doctype html><style>__REPORT_CSS__</style>'
        '<script type="application/json" id="report-data">__REPORT_DATA__</script>'
        '<script>__REPORT_JS__</script>')
    (web_dir / "style.css").write_text("body {color: white}")
    (web_dir / "app.js").write_text("console.log('fixture only');")
    return {"umap_path": umap_path, "index_path": index_path, "warehouse": warehouse,
            "output": tmp_path / "site", "plate": "P1", "run_id": "pilot", "web_dir": web_dir}


def test_export_bundles_original_bytes_and_records_provenance(build_inputs):
    payload = builder().build_report(**build_inputs)
    site = build_inputs["output"]
    assert (site / ".nojekyll").is_file()
    assert payload["images"]["status"] == "bundled"
    assert payload["images"]["files"] == 6
    html = (site / "index.html").read_text()
    embedded = json.loads(re.search(r'id="report-data">(.*?)</script>', html, re.S)[1])
    assert embedded == payload
    assert str(build_inputs["warehouse"]) not in html
    manifest = json.loads((site / "manifest.json").read_text())
    assert len(manifest["assets"]) == 6
    for entry in manifest["assets"]:
        copied = (site / entry["path"]).read_bytes()
        original = (build_inputs["warehouse"] / entry["path"].removeprefix("assets/")).read_bytes()
        assert copied == original
        assert hashlib.sha256(copied).hexdigest() == entry["sha256"]
    assert payload["sources"]["umap"]["sha256"] == hashlib.sha256(build_inputs["umap_path"].read_bytes()).hexdigest()


@pytest.mark.parametrize("problem", ["missing", "corrupt", "wrong_size", "wrong_mode", "symlink"])
def test_incomplete_or_invalid_warehouse_leaves_no_published_output(build_inputs, problem):
    index = pd.read_parquet(build_inputs["index_path"])
    path = build_inputs["warehouse"] / index.iloc[0].composite
    if problem == "missing":
        path.unlink()
    elif problem == "corrupt":
        path.write_bytes(b"not a jpeg")
    elif problem == "wrong_size":
        Image.new("RGB", (160, 160)).save(path, "JPEG")
    elif problem == "wrong_mode":
        Image.new("L", (224, 224)).save(path, "JPEG")
    elif problem == "symlink":
        outside = build_inputs["output"].parent / "outside.jpg"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    with pytest.raises((ValueError, OSError)):
        builder().build_report(**build_inputs)
    assert not build_inputs["output"].exists()
    assert not build_inputs["output"].with_name(".site.partial").exists()


def test_unavailable_warehouse_requires_explicit_diagnostic_opt_in(build_inputs):
    build_inputs["warehouse"] = None
    with pytest.raises(ValueError, match="Warehouse is unavailable"):
        builder().build_report(**build_inputs)
    payload = builder().build_report(**build_inputs, allow_missing_images=True)
    assert payload["images"]["status"] == "unavailable"
    assert payload["images"]["files"] == 0
    assert all(point["images"] == {} for point in payload["points"])
    assert not (build_inputs["output"] / "assets").exists()


def test_diagnostic_opt_in_does_not_hide_partial_warehouse(build_inputs):
    index = pd.read_parquet(build_inputs["index_path"])
    (build_inputs["warehouse"] / index.iloc[0].composite).unlink()
    with pytest.raises(ValueError, match="Missing or unsafe"):
        builder().build_report(**build_inputs, allow_missing_images=True)


def test_existing_output_is_never_overwritten(build_inputs):
    build_inputs["output"].mkdir()
    sentinel = build_inputs["output"] / "keep.txt"
    sentinel.write_text("keep me")
    with pytest.raises(ValueError, match="Refusing to replace"):
        builder().build_report(**build_inputs)
    assert sentinel.read_text() == "keep me"


def test_script_text_in_metadata_cannot_escape_json_element(build_inputs):
    text = '</script><script>window.injected=true</script> __REPORT_JS__ & < >'
    for name in ["umap_path", "index_path"]:
        frame = pd.read_parquet(build_inputs[name])
        frame.loc[0, "Metadata_cell_line"] = text
        frame.to_parquet(build_inputs[name])
    payload = builder().build_report(**build_inputs)
    html = (build_inputs["output"] / "index.html").read_text()
    assert text not in html
    assert 'window.injected=true</script>' not in html
    embedded = json.loads(re.search(r'id="report-data">(.*?)</script>', html, re.S)[1])
    assert embedded["points"][0]["line"] == text
    assert embedded == payload


def test_cli_builds_from_absolute_inputs_and_reports_missing_assets(build_inputs):
    import subprocess
    import sys

    command = [sys.executable, str(MODULE), "--umap", str(build_inputs["umap_path"]),
               "--crop-index", str(build_inputs["index_path"]), "--plate", "P1",
               "--run-id", "pilot", "--output", str(build_inputs["output"]),
               "--allow-missing-images", "--web-dir", str(build_inputs["web_dir"])]
    result = subprocess.run(command, cwd=build_inputs["output"].parent, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert (build_inputs["output"] / "index.html").exists()
    assert json.loads(result.stdout)["images"]["status"] == "unavailable"
    failed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert failed.returncode != 0
    assert "Refusing to replace" in failed.stderr
    assert "Traceback" not in failed.stderr


def test_oversized_static_site_fails_before_publication(build_inputs, monkeypatch):
    module = builder()
    monkeypatch.setattr(module, "MAX_SITE_BYTES", 100, raising=False)
    with pytest.raises(ValueError, match="size budget"):
        module.build_report(**build_inputs)
    assert not build_inputs["output"].exists()


