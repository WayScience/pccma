#!/usr/bin/env python
# coding: utf-8

# ## Generate platemap and barcode platemap from excel file

# In[1]:


import pandas as pd
import pathlib
import openpyxl


# In[2]:


# Path to the Excel file
file_path = pathlib.Path("./Rambutan_Atlasv2_April2026.xlsx")

# Output for platemap CSVs
output_dir = pathlib.Path("./platemaps")
output_dir.mkdir(exist_ok=True)


# In[3]:


# Plates and sheets we expect
plates = ["Plate-1", "Plate-2", "Plate-3"]
timepoints = ["24hr Metadata", "48hr Metadata", "72hr Metadata"]

# Cell lines that are spelled differently across sheets (name in Excel file -> standardized name)
cell_line_renames = {"U-2OS": "U2-OS"}

# Corrections for seeding densities that are wrong in the Excel file (plate barcode, well -> density).
# The densities in column 6 of the Plate-1 24hr sheet are shifted down by one cell line
# (C06 is empty and every well below has the density of the cell line above it), so
# these wells are set to the density of the other three wells from the same cell line,
# which matches the "Atlas conditions" sheet.
seeding_density_corrections = {
    ("BR00150695", "C06"): 12000,  # NB-1 (empty in the Excel file)
    ("BR00150695", "E06"): 8000,  # ONS-76 (12000 in the Excel file)
    ("BR00150695", "G06"): 4000,  # PA1 (8000 in the Excel file)
    ("BR00150695", "I06"): 8000,  # Saos-2 (4000 in the Excel file)
    ("BR00150695", "M06"): 12000,  # SK-N-AS (8000 in the Excel file)
}

# Read workbook once
xls = pd.ExcelFile(file_path)

# Map stripped sheet names to real names (handles trailing whitespace)
sheet_lookup = {s.strip(): s for s in xls.sheet_names}

for plate in plates:
    for tp in timepoints:

        expected_sheet = f"{plate} {tp}"

        if expected_sheet not in sheet_lookup:
            raise ValueError(
                f"Sheet '{expected_sheet}' not found (after stripping whitespace). "
                f"Available sheets: {xls.sheet_names}"
            )

        # Use the real sheet name (may contain trailing whitespace)
        sheet_name = sheet_lookup[expected_sheet]

        # Read sheet
        data = pd.read_excel(xls, sheet_name=sheet_name)

        # Remove rows where 'Cell Line' is 'media'
        data = data[data["Cell Line"] != "media"]

        # Rename columns to lowercase with underscores
        data.columns = data.columns.str.lower().str.replace(" ", "_")

        # Rename specific columns
        data = data.rename(
            columns={
                "time_point_(hours)": "time_point",
                "density_(cells/well)": "seeding_density",
            }
        )

        # Create well_position column (e.g., B + 3 -> B03)
        data["well_position"] = data["row"].astype(str) + data["column"].astype(
            str
        ).str.zfill(2)

        # Standardize cell line names so each cell line has one name across all plates
        data["cell_line"] = data["cell_line"].str.strip().replace(cell_line_renames)

        # Standardize capitalization of the plate coating (e.g., standard -> Standard)
        data["plate_coating"] = data["plate_coating"].str.strip().str.capitalize()

        # Get plate barcode for filename
        plate_barcode = data["plate_barcode"].dropna().unique()

        if len(plate_barcode) != 1:
            raise ValueError(f"Expected one plate_barcode but found: {plate_barcode}")

        # Apply the seeding density corrections for this plate
        for (barcode, well), density in seeding_density_corrections.items():
            if barcode != plate_barcode[0]:
                continue

            well_mask = data["well_position"] == well
            if well_mask.sum() != 1:
                raise ValueError(f"Expected one row for well {well} in {barcode}")

            print(
                f"Corrected seeding_density for {barcode} {well} "
                f"({data.loc[well_mask, 'cell_line'].item()}): "
                f"{data.loc[well_mask, 'seeding_density'].item()} -> {density}"
            )
            data.loc[well_mask, "seeding_density"] = density

        # All wells must have a seeding density after the corrections
        nan_rows = data[data["seeding_density"].isna()]

        if not nan_rows.empty:
            raise ValueError(
                f"Rows with NaNs in seeding_density ({plate}, {tp}):\n{nan_rows}"
            )

        # Save seeding density as a whole number (e.g., 12000 instead of 12000.0)
        data["seeding_density"] = data["seeding_density"].astype(int)

        # All wells from the same cell line and coating on a plate should have the same density
        densities_per_condition = data.groupby(["cell_line", "plate_coating"])[
            "seeding_density"
        ].nunique()

        if (densities_per_condition > 1).any():
            raise ValueError(
                f"More than one seeding_density per cell line ({plate}, {tp}):\n"
                f"{densities_per_condition[densities_per_condition > 1]}"
            )

        output_file = pathlib.Path(f"{output_dir}/{plate_barcode[0]}_platemap.csv")

        # Save processed platemap
        data.to_csv(output_file, index=False)

        print(f"Saved {output_file}")


# In[4]:


# Run a quick check across all files to confirm the platemaps are consistent
platemaps = pd.concat(
    [pd.read_csv(csv_file) for csv_file in sorted(output_dir.glob("*_platemap.csv"))]
)

if platemaps["seeding_density"].isna().any():
    raise ValueError("NaN values found in seeding_density")

print(f"Number of plates: {platemaps['plate_barcode'].nunique()}")
print(f"Number of unique cell lines: {platemaps['cell_line'].nunique()}")
print(f"Plate coatings: {sorted(platemaps['plate_coating'].unique())}")
print(f"Seeding densities: {sorted(platemaps['seeding_density'].unique().tolist())}")

