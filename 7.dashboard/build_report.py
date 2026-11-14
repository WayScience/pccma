#!/usr/bin/env python3
"""Export existing single-cell UMAPs and their exact crop-index matches to a static page."""

from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote
import argparse
import hashlib
import io
import json
import math
import re
import shutil

import pandas as pd
from PIL import Image

# Leave headroom below a 1 GB published-site budget; never silently ship an oversized bundle.
MAX_SITE_BYTES = 950_000_000

POSITION_KEY = [
    "Metadata_Plate", "Metadata_Well", "Metadata_Site",
    "Metadata_Nuclei_Location_Center_X", "Metadata_Nuclei_Location_Center_Y",
]
OBJECT_KEY = POSITION_KEY[:3] + [
    "Metadata_ImageNumber", "Metadata_Nuclei_Number_Object_Number",
    "Metadata_Cells_Number_Object_Number",
]
CHANNELS = ["composite", "DNA", "ER", "RNA", "AGP", "Mito"]
PATH_COLUMNS = {ch: ch if ch == "composite" else f"grayscale_{ch}" for ch in CHANNELS}


def require_table(frame, columns, label, *, unique=None):
    """Reject absent/null fields and ambiguous keys before attempting a join."""
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label}: missing columns: {sorted(missing)}")
    if frame[columns].isna().any().any():
        raise ValueError(f"{label}: null required values")
    if unique and frame.duplicated(unique).any():
        raise ValueError(f"{label}: duplicate keys")


def relative_crop_path(value):
    """Accept only unchanged relative JPEG paths beneath the warehouse's crop tree."""
    if not isinstance(value, str):
        raise ValueError("Crop path must be a string")
    path = PurePosixPath(value)
    if (path.is_absolute() or not path.parts or path.parts[0] != "crops"
            or any(part in ("..", ".") for part in value.split("/"))
            or "\\" in value or ":" in value or path.as_posix() != value
            or path.suffix.lower() != ".jpg"):
        raise ValueError(f"Invalid relative crop path: {value}")
    return path


def prepare_data(umap, index, *, plate, run_id):
    """Join exact nucleus positions, then retain the index's complete object identity.

    The analysis table does not retain object numbers. Plate/well/site plus the
    unrounded nucleus X/Y is its available key; require uniqueness on both sides.
    """
    require_table(umap, ["Metadata_Plate"], "UMAP")
    require_table(index, ["Metadata_Plate"], "Crop index")
    selected = umap.loc[umap.Metadata_Plate.eq(plate)].copy()
    crops = index.loc[index.Metadata_Plate.eq(plate)].copy()
    if selected.empty or crops.empty:
        raise ValueError(f"No UMAP cells or crop-index cells for plate {plate}")
    metadata = ["Metadata_cell_line", "Metadata_time_point"]
    umap_required = POSITION_KEY + metadata + ["cancer_group", "tissue_site", "umap_1", "umap_2", "cluster"]
    crop_required = list(dict.fromkeys(POSITION_KEY + OBJECT_KEY + metadata +
                                      list(PATH_COLUMNS.values()) + ["Metadata_Nucleocentric_Crop_Size"]))
    require_table(selected, umap_required, "UMAP", unique=POSITION_KEY)
    require_table(crops, crop_required, "Crop index", unique=OBJECT_KEY)
    require_table(crops, POSITION_KEY, "Crop positions", unique=POSITION_KEY)
    for frame, columns in [(selected, POSITION_KEY[3:] + ["umap_1", "umap_2"]),
                           (crops, POSITION_KEY[3:])]:
        if not all(math.isfinite(float(value)) for col in columns for value in frame[col]):
            raise ValueError("Non-finite coordinates")
    for frame, columns in [(selected, ["Metadata_Site", "Metadata_time_point", "cluster"]),
                           (crops, OBJECT_KEY[2:] + ["Metadata_time_point", "Metadata_Nucleocentric_Crop_Size"])]:
        for col in columns:
            if not all(float(v).is_integer() and float(v) > 0 for v in frame[col]):
                raise ValueError(f"Expected positive integral values: {col}")
    if crops.Metadata_Nucleocentric_Crop_Size.nunique() != 1:
        raise ValueError("Mixed crop dimensions in selected plate")
    for col in PATH_COLUMNS.values():
        for value in crops[col]:
            relative_crop_path(value)
        if crops[col].duplicated().any():
            raise ValueError(f"Reused crop paths: {col}")
    joined = selected.merge(crops[crop_required], on=POSITION_KEY, how="left",
                            validate="one_to_one", suffixes=("", "_index"), indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("UMAP cells without an exact crop-index match")
    for col in metadata:
        if not joined[col].eq(joined[col + "_index"]).all():
            raise ValueError(f"UMAP and crop metadata disagree: {col}")
    points = []
    for row in joined.to_dict("records"):
        identifier = (f"{plate}_{row['Metadata_Well']}_f{int(row['Metadata_Site']):02d}"
                      f"_nuc{int(row['Metadata_Nuclei_Number_Object_Number']):04d}"
                      f"_cell{int(row['Metadata_Cells_Number_Object_Number']):04d}")
        points.append({
            "id": identifier, "line": row["Metadata_cell_line"], "group": row["cancer_group"],
            "tissue": row["tissue_site"], "well": row["Metadata_Well"],
            "site": int(row["Metadata_Site"]), "cell": int(row["Metadata_Cells_Number_Object_Number"]),
            "nucleus": int(row["Metadata_Nuclei_Number_Object_Number"]),
            "time": int(row["Metadata_time_point"]), "cluster": int(row["cluster"]),
            "x": row["umap_1"], "y": row["umap_2"],
            "cx": row[POSITION_KEY[3]], "cy": row[POSITION_KEY[4]],
            "images": {ch: "assets/" + quote(row[col], safe="/") for ch, col in PATH_COLUMNS.items()},
        })
    return {
        "schema_version": 1, "plate": plate, "run_id": run_id,
        "summary": {"warehouse_cells": len(crops), "umap_cells": len(points),
                    "excluded_umap_rows": len(umap) - len(selected),
                    "cell_lines": selected.Metadata_cell_line.nunique(),
                    "cancer_groups": selected.cancer_group.nunique(),
                    "crop_size": int(crops.Metadata_Nucleocentric_Crop_Size.iloc[0])},
        "sources": {"qc": "Inherited published post-QC membership; QC not recalculated."},
        "images": {"status": "bundled", "message": "Original warehouse JPEGs.", "channels": CHANNELS},
        "points": points,
    }


def sha256(path):
    """Hash a source file without retaining its local/private absolute path."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_report(*, umap_path, index_path, warehouse, output, plate, run_id,
                 allow_missing_images=False, web_dir=None):
    """Build a new immutable static output with byte-identical, validated warehouse JPGs.

    A diagnostic coordinate-only build requires explicit opt-in and has no image
    URLs. A partial or corrupt image set never silently becomes a complete report.
    """
    output = Path(output)
    staging = output.with_name(f".{output.name}.partial")
    if output.exists() or output.is_symlink() or staging.exists():
        raise ValueError(f"Refusing to replace existing output/staging: {output}")
    umap_path, index_path = Path(umap_path), Path(index_path)
    payload = prepare_data(pd.read_parquet(umap_path), pd.read_parquet(index_path),
                           plate=plate, run_id=run_id)
    for label, path in [("umap", umap_path), ("crop_index", index_path)]:
        payload["sources"][label] = {"name": path.name, "sha256": sha256(path)}
    warehouse = Path(warehouse) if warehouse is not None else None
    available = warehouse is not None and warehouse.is_dir()
    if not available and not allow_missing_images:
        raise ValueError("Warehouse is unavailable; supply --warehouse with the extracted JPEG tree")
    if not available:
        payload["images"].update(status="unavailable", message=
            "Coordinate-only preview: warehouse JPEGs are not bundled. Cell crop previews are unavailable.")
        for point in payload["points"]:
            point["images"] = {}
    assets = []
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        seen = set()
        for point in payload["points"]:
            for channel, url in point["images"].items():
                relative = relative_crop_path(unquote(url.removeprefix("assets/")))
                if str(relative) in seen:
                    raise ValueError(f"Reused image path: {relative}")
                seen.add(str(relative))
                source = warehouse / relative
                if (source.is_symlink() or not source.resolve().is_relative_to(warehouse.resolve())
                        or not source.is_file()):
                    raise ValueError(f"Missing or unsafe warehouse image: {relative}")
                data = source.read_bytes()
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    expected_mode = "RGB" if channel == "composite" else "L"
                    size = payload["summary"]["crop_size"]
                    if image.format != "JPEG" or image.mode != expected_mode or image.size != (size, size):
                        raise ValueError(f"Wrong JPEG geometry/mode: {relative}")
                target = staging / "assets" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                if sha256(target) != digest:
                    raise ValueError(f"Copied image checksum mismatch: {relative}")
                assets.append({"path": f"assets/{relative}", "sha256": digest, "bytes": len(data)})
        payload["images"]["files"] = len(assets)
        payload["images"]["bytes"] = sum(asset["bytes"] for asset in assets)
        web_dir = Path(web_dir) if web_dir else Path(__file__).parent / "web"
        template = (web_dir / "template.html").read_text()
        # Escape data for an HTML script raw-text element, not merely for JSON.
        embedded = json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        embedded = embedded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        replacements = {"__REPORT_CSS__": (web_dir / "style.css").read_text(),
                        "__REPORT_JS__": (web_dir / "app.js").read_text(),
                        "__REPORT_DATA__": embedded}
        if any(template.count(token) != 1 for token in replacements):
            raise ValueError("Report template must contain each placeholder exactly once")
        html = re.sub("|".join(replacements), lambda match: replacements[match.group()], template)
        (staging / "index.html").write_text(html)
        (staging / ".nojekyll").touch()
        manifest = {"schema_version": 1, "plate": plate, "run_id": run_id,
                    "sources": payload["sources"], "summary": payload["summary"],
                    "images": payload["images"], "assets": assets,
                    "index_sha256": sha256(staging / "index.html"),
                    "code_sha256": {p.name: sha256(p) for p in [Path(__file__),
                        web_dir / "template.html", web_dir / "style.css", web_dir / "app.js"]}}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        total_bytes = payload["images"]["bytes"] + (staging / "index.html").stat().st_size + (staging / "manifest.json").stat().st_size
        if total_bytes > MAX_SITE_BYTES:
            raise ValueError(f"Static site exceeds size budget: {total_bytes} > {MAX_SITE_BYTES} bytes")
        if output.exists():
            raise ValueError(f"Output appeared during build: {output}")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return payload


def main(argv=None):
    """Build a static report, exiting nonzero rather than emitting an incomplete bundle."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--umap", type=Path, default=root / "6.profile_analysis/results/3.single_cell_embedding/single_cell_umap_coordinates_by_cancer_group.parquet")
    parser.add_argument("--crop-index", type=Path, required=True)
    parser.add_argument("--warehouse", type=Path, help="Extracted warehouse root containing crops/ (read only)")
    parser.add_argument("--plate", default="BR00150695")
    parser.add_argument("--run-id", required=True, help="Crop warehouse run identifier for provenance")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwritten")
    parser.add_argument("--allow-missing-images", action="store_true", help="Explicit coordinate-only diagnostic preview, NOT a complete crop dashboard")
    parser.add_argument("--web-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        payload = build_report(umap_path=args.umap, index_path=args.crop_index,
                               warehouse=args.warehouse, output=args.output, plate=args.plate,
                               run_id=args.run_id, allow_missing_images=args.allow_missing_images,
                               web_dir=args.web_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Report build failed: {error}\n")
    print(json.dumps({"output": str(args.output.resolve()), "summary": payload["summary"],
                      "images": payload["images"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
