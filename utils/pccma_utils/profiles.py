"""Helpers for the columns of the profile files."""

import pyarrow as pa


def numeric_features(schema):
    """Names of the feature columns of a profile file, from its pyarrow schema.

    A feature is a numeric (integer or floating point) column that is not metadata. Columns that start with
    `Metadata_` (plate, well, position of the cell, and so on) and `Image_` (the image file names and the
    image-level measurements) are not features. Returns a set, so that the features shared by several plates
    can be found with `set.intersection`.
    """
    return {
        field.name
        for field in schema
        if not field.name.startswith(("Metadata_", "Image_"))
        and (pa.types.is_floating(field.type) or pa.types.is_integer(field.type))
    }
