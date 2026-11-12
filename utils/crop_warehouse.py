"""Write the nucleocentric crop of each single cell to the image-crop warehouse.

The warehouse (built by `5a.object_image_crops`) is a folder of JPG images of the single cells of
the atlas: five grayscale crops (one per Cell Painting channel) and one color composite per cell.
It is decoupled from the parquet files of the other modules: nothing here reads or writes the
profiles, and a manifest parquet shipped with the images carries the join keys, so any crop can be
joined back to the single-cell profiles and the UMAP tables (see the README of the module).

The crops are the same nucleocentric crops as the MorphEm features (`5.morphem_featurization`):
the same center, the same padding at the edge of the image, the same illumination correction, and
the same 8-bit rescaling before JPEG compression. JPEG is lossy: these are visualization
crops, not lossless model inputs. Geometry is shared through `nucleocentric_crops`.
"""

import io
import pathlib

import pandas as pd

import numpy as np
import tifffile
from PIL import Image, __version__ as pillow_version

from nucleocentric_crops import (
    MORPHEM_INPUT_SIZE,
    get_nucleocentric_crop_box,
    rescale_crop_to_8bit,
    crop_with_padding,
)

# channels of the Cell Painting assay, in the order of the profiles (Image_FileName_Orig<channel>)
CHANNELS = ["DNA", "ER", "RNA", "AGP", "Mito"]

# display color of each channel in the composite crop (RGB, 0-1), the same as the colors of the
# profile-analysis figures (utils/pccma_utils/cell_images.py CHANNELS)
CHANNEL_COLORS = {
    "DNA": (0.0, 0.35, 1.0),
    "ER": (0.0, 1.0, 0.0),
    "RNA": (1.0, 1.0, 0.0),
    "AGP": (1.0, 0.15, 0.0),
    "Mito": (1.0, 0.0, 1.0),
}

# JPEG display derivatives; do not infer model-input equivalence from the quality setting.
JPEG_QUALITY = 95

# side of the square crop centered on the nucleus, the same as the MorphEm features
CROP_SIZE = MORPHEM_INPUT_SIZE

# identifier of each cell within its plate (unique across the warehouse): plate, well, site
# (field of view), and the object numbers of the cell and its nucleus from CellProfiler.
# The profiles and the QC annotations join on the same columns (QC_MERGE_KEYS in
# preprocess_utils.py), and the UMAP tables of the profile analysis join on the plate,
# well, site, and nucleus center.
PROFILE_KEY_COLUMNS = [
    "Image_Metadata_Plate",
    "Image_Metadata_Well",
    "Image_Metadata_Site",
    "Metadata_ImageNumber",
    "Metadata_Nuclei_Number_Object_Number",
    "Metadata_Cells_Number_Object_Number",
]

# the position of the nucleus in the image, which the UMAP tables carry
# (6.profile_analysis/results/3.single_cell_embedding)
UMAP_KEY_COLUMNS = [
    "Image_Metadata_Plate",
    "Image_Metadata_Well",
    "Image_Metadata_Site",
    "Metadata_Nuclei_Location_Center_X",
    "Metadata_Nuclei_Location_Center_Y",
]


def correct_illumination(image: np.ndarray, illum_function: np.ndarray) -> np.ndarray:
    """Apply the illumination correction function of a channel to one field of view.

    The same as CorrectIlluminationApply in the CellProfiler pipelines, but in the 16-bit range
    instead of 0 to 1 (CellProfiler divides the images by 65535 when loading them); the constant
    factor cancels out in the 8-bit rescaling of each crop. Copied from the MorphEm extraction
    (`5.morphem_featurization/2.extract_morphem_features.ipynb`) so both use the same pixels.

    Parameters
    ----------
    image : np.ndarray
        Raw image of one channel of one field of view.
    illum_function : np.ndarray
        Illumination correction function of the channel (from 2.illumination_correction).

    Returns
    -------
    np.ndarray
        The corrected image, float32, clipped to the 16-bit range.
    """
    if image.ndim != 2 or image.dtype != np.uint16:
        raise ValueError("Expected a 2D uint16 source image")
    if image.shape != illum_function.shape:
        raise ValueError("Image and illumination function shapes differ")
    if not np.isfinite(illum_function).all() or (illum_function <= 0).any():
        raise ValueError("Illumination functions must be finite and positive")
    return np.clip(
        image.astype(np.float32) / illum_function, 0, np.iinfo(image.dtype).max
    )


def composite_crops(
    crops: dict[str, np.ndarray], colors: dict[str, tuple[float, float, float]] | None = None
) -> np.ndarray:
    """Combine the five channel crops into one RGB image with the Cell Painting colors.

    Each channel is stretched between the 1st and 99.8th percentile of its own crop (the same
    stretch as the example-cell images of the profile analysis, `pccma_utils.cell_images.crop_cell`)
    and the channels are blended with the screen blend (`1 - (1 - a)(1 - b)`), so a bright pixel in
    any channel makes the composite bright. The blend uses 0.85 of the full strength, like the
    example-cell images, so overlapping channels stay colorful.

    Parameters
    ----------
    crops : dict[str, np.ndarray]
        The 8-bit crop of each channel, keyed by channel name (DNA, ER, RNA, AGP, Mito).
    colors : dict[str, tuple[float, float, float]] | None
        RGB color (0-1) of each channel. If None, use CHANNEL_COLORS.

    Returns
    -------
    np.ndarray
        The composite image, height x width x 3, uint8 (0 is black, 255 is the full channel color).
    """
    if colors is None:
        colors = CHANNEL_COLORS
    screen = np.ones((*crops[CHANNELS[0]].shape, 3), dtype=np.float32)
    for channel in CHANNELS:
        crop = crops[channel].astype(np.float32)
        lower, upper = np.percentile(crop, [1, 99.8])
        # a crop without contrast (flat background, all padding) would divide by zero; show its
        # flat level directly instead, so a dark crop stays dark and a bright one keeps its color
        if upper <= lower:
            stretched = crop / 255.0
        else:
            stretched = np.clip((crop - lower) / (upper - lower), 0, 1)
        screen *= 1 - 0.85 * stretched[..., None] * np.asarray(colors[channel], dtype=np.float32)
    return np.round((1 - screen) * 255).astype(np.uint8)


def crop_id(record: dict) -> str:
    """File-stable identifier of one cell within the whole warehouse.

    The plate, well, site, and the object numbers of the cell and its nucleus from CellProfiler,
    joined with underscores, so the file name says which cell it is even without the manifest.

    Parameters
    ----------
    record : dict
        One row of the single-cell table (the columns of PROFILE_KEY_COLUMNS).

    Returns
    -------
    str
        The identifier, for example "BR00150698_C05_f02_nuc0005_cell0005".
    """
    return (
        f"{record['Image_Metadata_Plate']}_{record['Image_Metadata_Well']}"
        f"_f{int(record['Image_Metadata_Site']):02d}"
        f"_nuc{int(record['Metadata_Nuclei_Number_Object_Number']):04d}"
        f"_cell{int(record['Metadata_Cells_Number_Object_Number']):04d}"
    )


def save_cell_crops(
    crops: dict[str, np.ndarray],
    crop_dir: pathlib.Path,
    identifier: str,
    quality: int = JPEG_QUALITY,
) -> dict[str, str]:
    """Write the crops of one cell to its own folder: five grayscale JPGs and one composite JPG.

    Parameters
    ----------
    crops : dict[str, np.ndarray]
        The 8-bit crop of each channel, keyed by channel name.
    crop_dir : pathlib.Path
        Folder that holds the crops of the cell (created when missing).
    identifier : str
        Identifier of the cell (crop_id).
    quality : int
        JPEG quality, 0 to 100.

    Returns
    -------
    dict[str, str]
        The path of each written file, keyed by "grayscale_<channel>" and "composite".
    """
    crop_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for channel in CHANNELS:
        path = crop_dir / f"{identifier}_{channel}.jpg"
        path.write_bytes(encode_jpeg(crops[channel], quality))
        paths[f"grayscale_{channel}"] = str(path)
    composite_path = crop_dir / f"{identifier}_composite.jpg"
    composite_path.write_bytes(encode_jpeg(composite_crops(crops), quality))
    paths["composite"] = str(composite_path)
    return paths


def manifest_row(record: dict, crop_paths: dict[str, str], warehouse_dir: pathlib.Path) -> dict:
    """Build one row of the warehouse manifest from a cell record and its written crop paths.

    The row carries both key sets (PROFILE_KEY_COLUMNS and UMAP_KEY_COLUMNS), the cell line and
    time point, and the crop file paths relative to the warehouse, so the manifest alone can find
    every image and join every crop back to the profiles and the UMAP tables.

    Parameters
    ----------
    record : dict
        One row of the single-cell table.
    crop_paths : dict[str, str]
        The written file paths of the cell (from save_cell_crops).
    warehouse_dir : pathlib.Path
        Root folder of the warehouse (crop paths are stored relative to it).

    Returns
    -------
    dict
        One row of the manifest.
    """
    row = {column: record[column] for column in PROFILE_KEY_COLUMNS + UMAP_KEY_COLUMNS}
    row["Metadata_cell_line"] = record.get("Metadata_cell_line")
    row["Metadata_time_point"] = record.get("Metadata_time_point")
    row.update({f"Metadata_{name}": record[f"Image_Metadata_{name}"]
                for name in ("Plate", "Well", "Site")})
    row["Metadata_Nucleocentric_Crop_Size"] = CROP_SIZE
    row["Metadata_Crop_Generator"] = "nucleocentric_jpeg_v1"
    row["Metadata_Pillow_Version"] = pillow_version
    row["Metadata_NumPy_Version"] = np.__version__
    row["Metadata_JPEG_Quality"] = JPEG_QUALITY
    row["Metadata_JPEG_Subsampling"] = 0
    row["Metadata_Crop_Format"] = "JPEG"
    row["Metadata_Crop_Rescaling"] = "unsaturated_minmax_0_254_camera_saturated_255"
    box = get_nucleocentric_crop_box(record["Metadata_Nuclei_Location_Center_X"],
                                    record["Metadata_Nuclei_Location_Center_Y"])
    row.update(dict(zip(GEOMETRY_COLUMNS, box)))
    for channel in CHANNELS:
        for kind in ("FileName", "PathName"):
            column = f"Image_{kind}_Orig{channel}"
            if column in record:
                row[column] = record[column]
    row.update(
        {
            key: pathlib.Path(path).relative_to(warehouse_dir).as_posix()
            for key, path in crop_paths.items()
        }
    )
    return row


FINAL_PROFILE_KEY = [
    "Metadata_Plate", "Metadata_Well", "Metadata_Site", "Metadata_ImageNumber",
    "Metadata_Nuclei_Number_Object_Number", "Metadata_Cells_Number_Object_Number",
]
POSITION_KEY = FINAL_PROFILE_KEY[:3] + UMAP_KEY_COLUMNS[3:]
PATH_COLUMNS = [f"grayscale_{channel}" for channel in CHANNELS] + ["composite"]
GEOMETRY_COLUMNS = [
    f"Metadata_Nucleocentric_Crop_{bound}_{axis}"
    for axis in ("Y", "X") for bound in ("Minimum", "Maximum")
]


def encode_jpeg(array: np.ndarray, quality: int = JPEG_QUALITY) -> bytes:
    """Encode a grayscale or RGB uint8 image; keep composite colors without subsampling."""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="JPEG", quality=quality, subsampling=0)
    return buffer.getvalue()


def require_unique_keys(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    """Reject missing/null keys and duplicates rather than silently multiplying a join."""
    missing = set(keys) - set(frame.columns)
    if missing or frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{label}: missing, null or duplicate keys ({sorted(missing)})")


def select_passing_cells(cells: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    """Apply the same per-cell QC flags as the profile pipeline, failing closed on absent QC."""
    from preprocess_utils import QC_MERGE_KEYS

    require_unique_keys(cells, QC_MERGE_KEYS, "Converted cells")
    require_unique_keys(annotations, QC_MERGE_KEYS, "QC annotations")
    flags = [name for name in annotations if name.endswith("_is_outlier")]
    if not flags:
        raise ValueError("QC annotations have no _is_outlier flags")
    merged = cells.merge(annotations[QC_MERGE_KEYS + flags], on=QC_MERGE_KEYS,
                         how="left", validate="one_to_one")
    if merged[flags].isna().any().any() or not merged[flags].isin([True, False]).all().all():
        raise ValueError("Missing or invalid QC flags for converted cells")
    return merged.loc[~merged[flags].astype(bool).any(axis=1)].drop(columns=flags)


def resolve_image_path(record: dict, channel: str, images_dir: pathlib.Path) -> pathlib.Path:
    """Use the recorded TIFF, or relocate its plate-measurement suffix to an explicit root."""
    folder = pathlib.Path(record[f"Image_PathName_Orig{channel}"])
    name = record[f"Image_FileName_Orig{channel}"]
    if (folder / name).is_file():
        return folder / name
    # Match a measurement folder explicitly: do not silently choose a different measurement.
    parts = folder.parts
    plate = record["Image_Metadata_Plate"]
    measurement = next((i for i, part in enumerate(parts) if part.startswith(f"{plate}__")), None)
    if measurement is not None:
        candidate = images_dir.joinpath(*parts[measurement:], name)
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing source TIFF: {folder / name}; image root: {images_dir}")


def read_corrected_field(record: dict, illum_functions: dict,
                         images_dir: pathlib.Path) -> tuple[dict, dict]:
    """Read each TIFF once for a FOV, retaining the camera-saturation mask before correction."""
    corrected, saturated = {}, {}
    for channel in CHANNELS:
        image = tifffile.imread(resolve_image_path(record, channel, images_dir))
        corrected[channel] = correct_illumination(image, illum_functions[channel])
        saturated[channel] = image == np.iinfo(image.dtype).max
    if len({image.shape for image in corrected.values()}) != 1:
        raise ValueError("Source channels have different shapes")
    return corrected, saturated


def make_cell_crops(record: dict, corrected: dict, saturated: dict) -> tuple[dict, float]:
    """Make the five pre-JPEG crops with exactly the shared MorphEm geometry and rescaling."""
    x, y = record[UMAP_KEY_COLUMNS[3]], record[UMAP_KEY_COLUMNS[4]]
    height, width = corrected[CHANNELS[0]].shape
    if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < width and 0 <= y < height):
        raise ValueError(f"Nucleus center {(x, y)} is outside image {(height, width)}")
    box = get_nucleocentric_crop_box(x, y)
    crops = {}
    padding = 0.0
    for channel in CHANNELS:
        crop, padding = crop_with_padding(corrected[channel], box)
        mask = crop_with_padding(saturated[channel], box, pad_value=False)[0]
        crops[channel] = rescale_crop_to_8bit(crop, mask)
    return crops, padding


def validate_crop_files(index: pd.DataFrame, warehouse: pathlib.Path) -> None:
    """Check all six paths per row; no absolute paths, traversal, reused paths or absent files."""
    for column in PATH_COLUMNS:
        if column not in index or index[column].isna().any() or index[column].duplicated().any():
            raise ValueError(f"Missing or duplicate crop paths: {column}")
        for relative in index[column]:
            path = pathlib.Path(relative)
            if path.is_absolute() or ".." in path.parts or not (warehouse / path).is_file():
                raise ValueError(f"Missing or invalid crop file: {relative}")


def validate_profile_join(index: pd.DataFrame, profiles: pd.DataFrame, label: str) -> None:
    """Require identical post-QC cell sets (both directions), not just matching row counts."""
    require_unique_keys(index, FINAL_PROFILE_KEY, "Crop index")
    require_unique_keys(profiles, FINAL_PROFILE_KEY, label)
    merged = index[FINAL_PROFILE_KEY].merge(profiles[FINAL_PROFILE_KEY],
        on=FINAL_PROFILE_KEY, how="outer", validate="one_to_one", indicator=True)
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"{label}: crop/profile cell sets differ: {merged['_merge'].value_counts().to_dict()}")


def join_crop_paths(table: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Attach all six paths to a sampled single-cell table via its exact nucleus position."""
    require_unique_keys(index, POSITION_KEY, "Crop index positions")
    require_unique_keys(table, POSITION_KEY, "UMAP positions")
    joined = table.merge(index[POSITION_KEY + PATH_COLUMNS], on=POSITION_KEY,
                         how="left", validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("UMAP cells without a crop")
    return joined.drop(columns="_merge")


def validate_sample_pixels(index: pd.DataFrame, warehouse: pathlib.Path,
                           illum_dir: pathlib.Path, images_dir: pathlib.Path) -> int:
    """Recompute sampled crops and compare decoded JPEG pixels to a fresh local JPEG roundtrip.

    Comparing to the same codec roundtrip catches wrong images without pretending JPEG is lossless.
    Different codec/library versions can change pixels; rerun with the generation environment then.
    """
    checked = 0
    for plate, plate_df in index.groupby("Metadata_Plate"):
        functions = {ch: np.load(illum_dir / plate / f"{plate}_Illum{ch}.npy") for ch in CHANNELS}
        sample = plate_df.sample(min(3, len(plate_df)), random_state=0)
        for record in sample.to_dict("records"):
            corrected, saturated = read_corrected_field(record, functions, images_dir)
            crops, _ = make_cell_crops(record, corrected, saturated)
            arrays = {f"grayscale_{ch}": crops[ch] for ch in CHANNELS}
            arrays["composite"] = composite_crops(crops)
            for column, expected in arrays.items():
                with Image.open(warehouse / record[column]) as image:
                    actual = np.array(image)
                with Image.open(io.BytesIO(encode_jpeg(expected, int(record["Metadata_JPEG_Quality"])))) as image:
                    reference = np.array(image)
                if not np.array_equal(actual, reference):
                    raise ValueError(f"Wrong crop pixels: {record[column]}")
            checked += 1
    return checked
