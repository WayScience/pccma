"""Steps shared by the single-cell notebooks.

The notebooks `3.single_cell_embedding` and `5.neuroblastoma_single_cell` draw a sample of the single cells of the
feature-selected profiles, embed it with UMAP, cluster the embedding with k-means, and show the cells closest to the
center of each cluster. The functions here are the steps that both notebooks share:

- `sample_cells`: draw the same number of cells from each cell line on each plate.
- `closest_to_centers`: find the cells closest to the center of each cluster.
- `crop_cells_as_png`: cut the images of cells out of the raw images, as PNG bytes that fit in a table column.
- `png_to_array`: decode PNG bytes to an array that can be drawn.
"""

import io

import numpy as np
import pandas as pd
from matplotlib.image import imread, imsave

from pccma_utils.cell_images import crop_cell, find_image_dir

# The columns that identify a cell and tell where it is: plate, well, field of view, cell line, time point, and the
# position of its nucleus in the field of view (pixels).
CELL_COLUMNS = [
    "Metadata_Plate", "Metadata_Well", "Metadata_Site", "Metadata_cell_line", "Metadata_time_point",
    "Metadata_Nuclei_Location_Center_X", "Metadata_Nuclei_Location_Center_Y",
]


def sample_cells(plate_files, features, cells_per_line_per_plate, seed, cell_lines=None):
    """Draw up to `cells_per_line_per_plate` random cells (without replacement) of each cell line on each plate.

    `plate_files` are the single-cell profile files (parquet) to draw from, in the order that is used for the draws, and
    `features` are the feature columns to read besides `CELL_COLUMNS`. One random generator (from `seed`) is used for
    all plates and cell lines, so the same arguments always give the same cells.
    Only the cell lines in `cell_lines` are used when it is given, and every plate needs at least one of them.

    Returns the sampled cells of all plates (one row per cell, with `CELL_COLUMNS` and `features`) and a table with
    one row per plate: its file name, the number of cells sampled, and the number of cells it had to draw from.
    """
    rng = np.random.default_rng(seed)
    sampled, counts = [], []

    for plate_file in plate_files:
        plate_df = pd.read_parquet(plate_file, columns=CELL_COLUMNS + list(features))
        if cell_lines is not None:
            plate_df = plate_df[plate_df["Metadata_cell_line"].isin(cell_lines)]

        chosen = [
            rng.choice(np.asarray(rows), size=min(len(rows), cells_per_line_per_plate), replace=False)
            for rows in plate_df.groupby("Metadata_cell_line").indices.values()
        ]
        sampled.append(plate_df.iloc[np.concatenate(chosen)])
        counts.append((plate_file.name, len(sampled[-1]), len(plate_df)))

    return pd.concat(sampled, ignore_index=True), pd.DataFrame(counts, columns=["plate_file", "sampled", "available"])


def closest_to_centers(embedding, labels, centers, n=1):
    """The `n` cells closest to the center of each cluster, in the embedding.

    `embedding` has one row per cell, `labels` are the cluster labels of the cells counted from 0 (as k-means gives
    them), and `centers` are the cluster centers (row k is the center of label k).

    Returns a table with one row per selected cell: its `position` in `embedding`, its `cluster` (counted from 1),
    and its `rank` in the cluster (1 for the closest cell, 2 for the next, and so on). The clusters are in order, and
    a cluster with fewer than `n` cells gives all of its cells.
    """
    positions, clusters, ranks = [], [], []

    for cluster, center in enumerate(centers, start=1):
        in_cluster = np.flatnonzero(np.asarray(labels) == cluster - 1)
        distance = ((embedding[in_cluster] - center) ** 2).sum(axis=1)
        closest = in_cluster[np.argsort(distance)[:n]]
        positions.extend(closest)
        clusters.extend([cluster] * len(closest))
        ranks.extend(range(1, len(closest) + 1))

    return pd.DataFrame({"position": positions, "cluster": clusters, "rank": ranks})


def crop_cells_as_png(cells, screen_dir, crop_size):
    """Cut the image of each cell out of the raw images of its plate, as the bytes of a PNG file.

    `cells` has one row per cell, with the columns `Metadata_Plate`, `Metadata_Well`, `Metadata_Site`,
    `Metadata_Nuclei_Location_Center_X`, and `Metadata_Nuclei_Location_Center_Y`. `screen_dir` has the raw images
    of all plates, and `crop_size` is the width and height of the image in pixels (see `crop_cell`).

    Returns a list with the PNG bytes of each cell, in the order of the rows, so that it can be a column of the
    table of the cells (parquet stores it as binary). Use `png_to_array` to draw an image.
    """
    images = []

    for cell in cells.itertuples():
        image = crop_cell(
            find_image_dir(screen_dir, cell.Metadata_Plate), cell.Metadata_Well, cell.Metadata_Site,
            cell.Metadata_Nuclei_Location_Center_X, cell.Metadata_Nuclei_Location_Center_Y, size=crop_size,
        )
        buffer = io.BytesIO()
        imsave(buffer, np.clip(image, 0, 1), format="png")
        images.append(buffer.getvalue())

    return images


def png_to_array(png):
    """The image of PNG bytes (as `crop_cells_as_png` gives them), as an array (height x width x 4, values 0-1)."""
    return imread(io.BytesIO(png), format="png")
