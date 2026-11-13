"""Run the real notebook scripts on explicitly synthetic TIFFs and parquet inputs."""

import os
import pathlib
import shutil
import subprocess
import sys

import nbformat
import numpy as np
import pandas as pd
import pytest
import tifffile
from nbconvert import PythonExporter
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[2]
MODULE = "5a.object_image_crops"
NOTEBOOKS = ("0.generate_object_crops", "1.build_crop_index", "2.validate_joins")
FINAL_KEY = ["Metadata_Plate", "Metadata_Well", "Metadata_Site",
             "Metadata_ImageNumber", "Metadata_Nuclei_Number_Object_Number",
             "Metadata_Cells_Number_Object_Number"]
RENAME = {f"Image_Metadata_{name}": f"Metadata_{name}" for name in ("Plate", "Well", "Site")}


@pytest.fixture
def atlas(tmp_path):
    """Three synthetic objects: two pass QC, including one near an image edge."""
    root = tmp_path / "synthetic atlas"
    root.mkdir()
    (root / ".git").mkdir()
    script_dir = root / MODULE / "nbconverted"
    script_dir.mkdir(parents=True)
    for name in NOTEBOOKS:
        shutil.copyfile(REPO / MODULE / f"{name}.ipynb", script_dir.parent / f"{name}.ipynb")
        notebook = nbformat.read(REPO / MODULE / f"{name}.ipynb", as_version=4)
        script, _ = PythonExporter().from_notebook_node(notebook)
        (script_dir / f"{name}.py").write_text(script)
    rows = []
    image_dir = root / "source images" / "BR00150698__measurement" / "Images"
    image_dir.mkdir(parents=True)
    illum_dir = root / "2.illumination_correction/illum_directory/BR00150698"
    illum_dir.mkdir(parents=True)
    yy, xx = np.mgrid[:256, :256]
    for channel, number in zip(("DNA", "ER", "RNA", "AGP", "Mito"), (5, 2, 6, 3, 4)):
        image = (100 + number * xx + yy * 8).astype(np.uint16)
        image[125:130, 125:130] = 65535
        tifffile.imwrite(image_dir / f"ch{number}.tiff", image)
        np.save(illum_dir / f"BR00150698_Illum{channel}.npy", np.full((256, 256), 1.25, np.float32))
    for obj, center in enumerate((128.25, 8.5, 150.75), start=1):
        row = {"Image_Metadata_Plate": "BR00150698", "Image_Metadata_Well": "C05",
               "Image_Metadata_Site": 2, "Metadata_ImageNumber": 7,
               "Metadata_Nuclei_Number_Object_Number": obj,
               "Metadata_Cells_Number_Object_Number": obj,
               "Metadata_Nuclei_Location_Center_X": center,
               "Metadata_Nuclei_Location_Center_Y": center}
        for channel, number in zip(("DNA", "ER", "RNA", "AGP", "Mito"), (5, 2, 6, 3, 4)):
            row[f"Image_FileName_Orig{channel}"] = f"ch{number}.tiff"
            row[f"Image_PathName_Orig{channel}"] = str(image_dir)
        rows.append(row)
    converted = pd.DataFrame(rows)
    converted_dir = root / "4.preprocess_features/data/converted_profiles/A673"
    converted_dir.mkdir(parents=True)
    converted.to_parquet(converted_dir / "BR00150698_A673_converted.parquet", index=False)
    qc_dir = root / "4.preprocess_features/qc_results"
    qc_dir.mkdir(parents=True)
    qc = converted[["Image_Metadata_Plate", "Image_Metadata_Well", "Image_Metadata_Site",
                    "Metadata_Cells_Number_Object_Number"]].copy()
    qc["area_is_outlier"] = [False, False, True]
    qc["blur_is_outlier"] = [False, False, False]
    qc.to_parquet(qc_dir / "A673_qc_annotations.parquet", index=False)
    platemaps = root / "0.download_data/metadata/platemaps"
    platemaps.mkdir(parents=True)
    pd.DataFrame([{"plate_barcode": "BR00150698", "well_position": "C05",
                   "cell_line": "A673", "time_point": 24}]).to_csv(
                       platemaps / "BR00150698_platemap.csv", index=False)
    passing = converted.iloc[:2].rename(columns=RENAME)
    for directory, filename in (
        ("4.preprocess_features/data/single_cell_profiles", "BR00150698_sc_feature_selected.parquet"),
        ("5.morphem_featurization/data/morphem_single_cell_profiles", "BR00150698_morphem_sc_feature_selected.parquet"),
    ):
        target = root / directory
        target.mkdir(parents=True)
        passing.to_parquet(target / filename, index=False)
    umap_dir = root / "6.profile_analysis/results/3.single_cell_embedding"
    umap_dir.mkdir(parents=True)
    position = ["Metadata_Plate", "Metadata_Well", "Metadata_Site",
                "Metadata_Nuclei_Location_Center_X", "Metadata_Nuclei_Location_Center_Y"]
    umap = passing[position].copy()
    umap["umap_1"], umap["umap_2"] = [0.1, 0.2], [0.3, 0.4]
    umap.to_parquet(umap_dir / "single_cell_umap_coordinates_by_cancer_group.parquet", index=False)
    umap.iloc[:1].to_parquet(umap_dir / "single_cell_umap_example_cells.parquet", index=False)
    return root


def run_stage(root, number, *args, ok=True):
    env = dict(os.environ, PYTHONPATH=str(REPO / "utils"))
    result = subprocess.run(
        [sys.executable, str(root / MODULE / "nbconverted" / f"{NOTEBOOKS[number]}.py"), *args],
        cwd=root, env=env, capture_output=True, text=True, timeout=45,
    )
    output = result.stdout + result.stderr
    if ok:
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0, output
    return output


def generate(root, *args, ok=True):
    return run_stage(root, 0, "--plate", "BR00150698", "--cell_line", "A673", *args, ok=ok)


def warehouse(root):
    return root / MODULE / "data/object_image_crops"


def test_notebooks_end_to_end(atlas):
    generate(atlas)
    run_stage(atlas, 1)
    output = run_stage(atlas, 2)
    index = pd.read_parquet(warehouse(atlas) / "crops_index.parquet")
    assert len(index) == 2
    assert index.columns.is_unique
    assert index["Metadata_cell_line"].tolist() == ["A673", "A673"]
    assert index["Metadata_time_point"].tolist() == [24, 24]
    assert not index.duplicated(FINAL_KEY).any()
    assert len(list(warehouse(atlas).rglob("*.jpg"))) == 12
    for row in index.to_dict("records"):
        for name in ("grayscale_DNA", "grayscale_ER", "grayscale_RNA", "grayscale_AGP", "grayscale_Mito", "composite"):
            with Image.open(warehouse(atlas) / row[name]) as image:
                assert image.size == (224, 224)
                assert image.mode == ("RGB" if name == "composite" else "L")
    assert "MorphEm" in output
    assert "UMAP" in output


def test_missing_qc_flags_fails_before_writing(atlas):
    qc_path = atlas / "4.preprocess_features/qc_results/A673_qc_annotations.parquet"
    qc = pd.read_parquet(qc_path).drop(columns=["area_is_outlier", "blur_is_outlier"])
    qc.to_parquet(qc_path, index=False)
    output = generate(atlas, ok=False)
    assert "QC" in output
    assert not list(warehouse(atlas).rglob("*.jpg"))


def test_zero_passing_cells_have_empty_manifest(atlas):
    qc_path = atlas / "4.preprocess_features/qc_results/A673_qc_annotations.parquet"
    qc = pd.read_parquet(qc_path)
    qc["area_is_outlier"] = True
    qc.to_parquet(qc_path, index=False)
    generate(atlas)
    manifest = pd.read_parquet(warehouse(atlas) / "manifests/BR00150698_A673_manifest.parquet")
    assert manifest.empty
    assert "composite" in manifest.columns
    assert not list(warehouse(atlas).rglob("*.jpg"))


def test_index_rejects_missing_grayscale(atlas):
    generate(atlas)
    next(warehouse(atlas).rglob("*_DNA.jpg")).unlink()
    assert "missing" in run_stage(atlas, 1, ok=False).lower()


def test_validation_rejects_missing_post_qc_cell(atlas):
    generate(atlas)
    run_stage(atlas, 1)
    path = warehouse(atlas) / "crops_index.parquet"
    pd.read_parquet(path).iloc[:1].to_parquet(path, index=False)
    assert "profile" in run_stage(atlas, 2, ok=False).lower()


def test_no_profile_files_cannot_report_success(atlas):
    generate(atlas)
    run_stage(atlas, 1)
    for path in (atlas / "4.preprocess_features/data/single_cell_profiles").glob("*.parquet"):
        path.unlink()
    assert "profile" in run_stage(atlas, 2, ok=False).lower()


def test_same_size_wrong_image_is_detected(atlas):
    generate(atlas)
    run_stage(atlas, 1)
    next_path = next(warehouse(atlas).rglob("*_DNA.jpg"))
    Image.fromarray(np.zeros((224, 224), np.uint8)).save(next_path, quality=95)
    assert "Wrong crop pixels" in run_stage(atlas, 2, ok=False)


def test_missing_umap_is_reported_as_skipped(atlas):
    generate(atlas)
    run_stage(atlas, 1)
    for path in (atlas / "6.profile_analysis/results/3.single_cell_embedding").glob("*.parquet"):
        path.unlink()
    assert "skipped" in run_stage(atlas, 2).lower()


def test_generator_skips_fresh_job_and_recovers_missing_crop(atlas):
    generate(atlas)
    path = warehouse(atlas) / "manifests/BR00150698_A673_manifest.parquet"
    mtime = path.stat().st_mtime_ns
    assert "Skipping" in generate(atlas)
    assert path.stat().st_mtime_ns == mtime
    next(warehouse(atlas).rglob("*_DNA.jpg")).unlink()
    assert "Skipping" not in generate(atlas)
    assert len(list(warehouse(atlas).rglob("*.jpg"))) == 12


def test_overwrite_removes_crops_of_cells_that_now_fail_qc(atlas):
    generate(atlas)
    qc_path = atlas / "4.preprocess_features/qc_results/A673_qc_annotations.parquet"
    qc = pd.read_parquet(qc_path)
    qc.loc[1, "area_is_outlier"] = True
    qc.to_parquet(qc_path, index=False)
    generate(atlas, "--overwrite")
    assert len(list(warehouse(atlas).rglob("*.jpg"))) == 6


def test_changed_source_tiff_forces_regeneration(atlas):
    generate(atlas)
    next(atlas.glob("source images/*/Images/*.tiff")).touch()
    assert "Skipping" not in generate(atlas)


def test_runner_end_to_end_with_existing_python(atlas):
    shutil.copyfile(REPO / MODULE / "run_crop_pipeline.sh", atlas / MODULE / "run_crop_pipeline.sh")
    (atlas / "utils").symlink_to(REPO / "utils", target_is_directory=True)
    env = dict(os.environ, PCCMA_PYTHON=sys.executable)
    result = subprocess.run(["bash", str(atlas / MODULE / "run_crop_pipeline.sh")],
        cwd=atlas, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (warehouse(atlas) / "validation.json").is_file()
    assert "No files have been synced" in result.stdout


def test_missing_qc_label_fails_closed(atlas):
    path = atlas / "4.preprocess_features/qc_results/A673_qc_annotations.parquet"
    pd.read_parquet(path).iloc[1:].to_parquet(path, index=False)
    assert "Missing or invalid QC flags" in generate(atlas, ok=False)


def test_duplicate_positions_are_rejected(atlas):
    path = atlas / "4.preprocess_features/data/converted_profiles/A673/BR00150698_A673_converted.parquet"
    cells = pd.read_parquet(path)
    for column in ("Metadata_Nuclei_Location_Center_X", "Metadata_Nuclei_Location_Center_Y"):
        cells.loc[1, column] = cells.loc[0, column]
    cells.to_parquet(path, index=False)
    assert "Post-QC positions" in generate(atlas, ok=False)


def test_missing_job_manifest_blocks_index(atlas):
    generate(atlas)
    source = atlas / "4.preprocess_features/data/converted_profiles/A673/BR00150698_A673_converted.parquet"
    shutil.copyfile(source, source.with_name("BR00150699_A673_converted.parquet"))
    assert "Missing or stale job manifests" in run_stage(atlas, 1, ok=False)


def test_actual_tracked_umap_key_contract():
    from crop_warehouse import PATH_COLUMNS, POSITION_KEY, join_crop_paths
    path = REPO / "6.profile_analysis/results/3.single_cell_embedding/single_cell_umap_coordinates_by_cancer_group.parquet"
    table = pd.read_parquet(path)
    # This checks key preservation only, not real-image availability.
    position_index = table[POSITION_KEY].copy()
    for column in PATH_COLUMNS:
        position_index[column] = [f"key-test/{i}/{column}.jpg" for i in range(len(table))]
    joined = join_crop_paths(table, position_index)
    assert len(joined) == len(table)
    pd.testing.assert_frame_equal(joined[POSITION_KEY], table[POSITION_KEY])


def test_source_relocation_preserves_measurement(atlas):
    from crop_warehouse import resolve_image_path
    directory = atlas / "source images"
    record = {"Image_Metadata_Plate": "BR00150698",
              "Image_PathName_OrigDNA": "/unmounted/PCCMA_atlas_data/BR00150698__measurement/Images",
              "Image_FileName_OrigDNA": "ch5.tiff"}
    expected = directory / "BR00150698__measurement/Images/ch5.tiff"
    assert resolve_image_path(record, "DNA", directory) == expected
    record["Image_PathName_OrigDNA"] = "/unmounted/BR00150698__other_measurement/Images"
    with pytest.raises(FileNotFoundError):
        resolve_image_path(record, "DNA", directory)


def add_other_plate_inputs(root):
    """An unprocessed second plate must not expand an explicitly plate-scoped run."""
    converted = root / "4.preprocess_features/data/converted_profiles/A673/BR00150698_A673_converted.parquet"
    cells = pd.read_parquet(converted)
    cells["Image_Metadata_Plate"] = "BR00150699"
    cells.to_parquet(converted.with_name("BR00150699_A673_converted.parquet"), index=False)
    for directory, filename in (
        ("4.preprocess_features/data/single_cell_profiles", "BR00150698_sc_feature_selected.parquet"),
        ("5.morphem_featurization/data/morphem_single_cell_profiles", "BR00150698_morphem_sc_feature_selected.parquet"),
    ):
        path = root / directory / filename
        table = pd.read_parquet(path)
        table["Metadata_Plate"] = "BR00150699"
        table.to_parquet(path.with_name(filename.replace("BR00150698", "BR00150699")), index=False)
    for path in (root / "6.profile_analysis/results/3.single_cell_embedding").glob("*.parquet"):
        table = pd.read_parquet(path)
        other = table.copy()
        other["Metadata_Plate"] = "BR00150699"
        pd.concat([table, other], ignore_index=True).to_parquet(path, index=False)


def test_one_plate_runner_does_not_process_or_require_other_plates(atlas, monkeypatch):
    import json
    add_other_plate_inputs(atlas)
    monkeypatch.setenv("PCCMA_PLATE", "BR00150698")
    shutil.copyfile(REPO / MODULE / "run_crop_pipeline.sh", atlas / MODULE / "run_crop_pipeline.sh")
    (atlas / "utils").symlink_to(REPO / "utils", target_is_directory=True)
    env = dict(os.environ, PCCMA_PYTHON=sys.executable)
    result = subprocess.run(["bash", str(atlas / MODULE / "run_crop_pipeline.sh")],
        cwd=atlas, env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    index = pd.read_parquet(warehouse(atlas) / "crops_index.parquet")
    assert set(index["Metadata_Plate"]) == {"BR00150698"}
    assert not (warehouse(atlas) / "crops/BR00150699").exists()
    report = json.loads((warehouse(atlas) / "validation.json").read_text())
    assert report["plate_scope"] == "BR00150698"
    assert report["umap_rows_outside_scope"]["single_cell_umap_coordinates_by_cancer_group.parquet"] == 2


def test_plate_scope_rejects_mixed_output_directory(atlas, monkeypatch):
    generate(atlas)
    manifest_dir = warehouse(atlas) / "manifests"
    mixed = manifest_dir / "BR00150699_A673_manifest.parquet"
    mixed.write_bytes(b"unrelated existing output")
    monkeypatch.setenv("PCCMA_PLATE", "BR00150698")
    assert "separate output" in generate(atlas, ok=False).lower()
    assert mixed.read_bytes() == b"unrelated existing output"


def test_plate_scope_rejects_mismatched_job(atlas, monkeypatch):
    monkeypatch.setenv("PCCMA_PLATE", "BR00150699")
    assert "scope" in generate(atlas, ok=False).lower()
    assert not list(warehouse(atlas).rglob("*.jpg"))
