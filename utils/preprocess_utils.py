"""Utility function to load the single-cell profiles of a plate that passed single-cell quality control (coSMicQC)."""

import pathlib

import pandas as pd

# columns that identify each single cell in a plate (each well only has one cell line)
QC_MERGE_KEYS = [
    "Image_Metadata_Plate",
    "Image_Metadata_Well",
    "Image_Metadata_Site",
    "Metadata_Cells_Number_Object_Number",
]


def load_plate_passing_qc(
    plate: str, converted_dir: pathlib.Path, qc_results_dir: pathlib.Path
) -> pd.DataFrame:
    """
    Load the converted profiles of every cell line on a plate, remove the single cells that failed QC
    for each cell line, and concatenate the cell lines into one plate data frame.

    Parameters
    ----------
    plate : str
        The name of the plate (e.g., BR00150695).
    converted_dir : pathlib.Path
        Directory with one folder per cell line holding the converted profiles (<plate>_<cell_line>_converted.parquet).
    qc_results_dir : pathlib.Path
        Directory with the QC annotations of each cell line (<cell_line>_qc_annotations.parquet).

    Returns
    -------
    pd.DataFrame
        The single-cell profiles of the plate that passed QC.
    """
    cell_line_dfs = []

    for profile_path in sorted(converted_dir.glob(f"*/{plate}_*_converted.parquet")):
        # name of the cell line is the name of the folder
        cell_line = profile_path.parent.name

        profile_df = pd.read_parquet(profile_path)

        # QC annotations for the cell line from this plate
        annotations_df = pd.read_parquet(
            qc_results_dir / f"{cell_line}_qc_annotations.parquet"
        )
        annotations_df = annotations_df[annotations_df["Image_Metadata_Plate"] == plate]
        outlier_columns = [
            col for col in annotations_df.columns if col.endswith("_is_outlier")
        ]

        # Merge the QC labels onto the profiles (one label per single cell)
        profile_df = profile_df.merge(
            annotations_df[QC_MERGE_KEYS + outlier_columns],
            on=QC_MERGE_KEYS,
            how="left",
            validate="one_to_one",
        )
        if profile_df[outlier_columns].isna().any().any():
            raise ValueError(
                f"Single cells without QC labels for {cell_line} on {plate} (run single-cell QC again)"
            )

        # Remove the single cells that failed any QC condition
        failed = profile_df[outlier_columns].astype(bool).any(axis=1)
        print(
            f"{cell_line}: removed {failed.sum()} of {len(failed)} single cells that failed QC"
        )
        cell_line_dfs.append(profile_df[~failed].drop(columns=outlier_columns))

    if not cell_line_dfs:
        raise FileNotFoundError(f"No converted profiles found for {plate}")

    return pd.concat(cell_line_dfs, ignore_index=True)
