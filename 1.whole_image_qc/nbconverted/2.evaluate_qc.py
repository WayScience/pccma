#!/usr/bin/env python
# coding: utf-8

# # Whole image quality control metric evaluation
# 
# In this notebook, we will use the QC metrics developing threshold using coSMicQC for the blur metric.
# 
# **Blur metric to detect out of focus images** -> PowerLogLogSlope
# 
# We will use CytoDataFrame to visualize which images will fail QC if we set a threshold of 1% of pixels at the max intensity.
# 
# **Saturation metric to detect large smudges or oversaturation** -> PercentMaximal

# ## Import libraries

# In[ ]:


import pathlib
import pandas as pd
import numpy as np

from cytodataframe import CytoDataFrame
from cosmicqc import find_outliers, identify_outliers
import matplotlib.pyplot as plt
import seaborn as sns


# ## Set paths and load in data frame

# In[ ]:


import pathlib
import pandas as pd

# Directory for figures to be outputted
figure_dir = pathlib.Path("./qc_figures")
figure_dir.mkdir(exist_ok=True)

# Directory with QC CellProfiler outputs per plate
qc_results_dir = pathlib.Path("./whole_img_qc_output")

# Create an empty dictionary to store data frames for each plate
all_qc_data_frames = {}

# List all plate directories within the qc_results_dir
plates = [plate.name for plate in qc_results_dir.iterdir() if plate.is_dir()]

# Loop through each plate
for plate in plates:
    # Read in CSV with all image quality metrics per image for the current plate
    qc_df = pd.read_csv(qc_results_dir / plate / "Image.csv")

    # Store the data frame for the current plate in the dictionary
    all_qc_data_frames[plate] = qc_df

# Print the plate names to ensure they were loaded correctly
print(all_qc_data_frames.keys())

# Concatenate all plate dataframes into a single DataFrame, keeping track of the plate
orig_df = pd.concat(
    [df.assign(Metadata_Plate=plate) for plate, df in all_qc_data_frames.items()],
    ignore_index=True,
)

# Quick check
print(orig_df.shape)
orig_df.head()

# Optional: select the first plate for an example
first_plate = plates[0]
print(f"Showing example for the first plate: {first_plate}")
example_df = all_qc_data_frames[first_plate]
print(example_df.shape)
example_df.head()


# ## Create concat data frames combining blur and saturation metrics from all channels for all plates

# In[ ]:


# List of channels (excluding Brightfield since the metrics are not robust to this type of channel)
channels = ["OrigDNA", "OrigER", "OrigAGP", "OrigMito", "OrigRNA"]

# Create an empty dictionary to store data frames for each channel
all_combined_dfs = {}

# Iterate through each channel
for channel in channels:
    # Create an empty list to store data frames for each plate
    plate_dfs = []

    # Iterate through each plate and create the specified data frame for the channel
    for plate, qc_df in all_qc_data_frames.items():
        plate_df = qc_df.filter(like="Metadata_").copy()

        # Add PowerLogLogSlope column (blur metric)
        plate_df["ImageQuality_PowerLogLogSlope"] = qc_df[
            f"ImageQuality_PowerLogLogSlope_{channel}"
        ]

        # Add PercentMaximal column (saturation metric)
        plate_df["ImageQuality_PercentMaximal"] = qc_df[
            f"ImageQuality_PercentMaximal_{channel}"
        ]

        # Add "Channel" column
        plate_df["Channel"] = channel

        # Add "Metadata_Plate" column
        plate_df["Metadata_Plate"] = plate

        # Append the data frame to the list
        plate_dfs.append(plate_df)

    # Concatenate data frames for each plate for the current channel
    all_combined_dfs[channel] = pd.concat(
        plate_dfs, keys=list(all_qc_data_frames.keys()), names=["Metadata_Plate", None]
    )

# Concatenate the channel data frames together for plotting
df = pd.concat(list(all_combined_dfs.values()), ignore_index=True)

print(df.shape)
df.head()


# ## Determine blur thresholds per channel

# In[ ]:


# Plot the distribution of the PowerLogLogSlope metric across all channels and per plate
g = sns.FacetGrid(
    df,
    col="Channel",
    col_wrap=3,  # adjust to your channel count
    height=6,
    aspect=1.2,
    sharey=True,
)
g.map_dataframe(
    sns.boxplot,
    x="Metadata_Plate",
    y="ImageQuality_PowerLogLogSlope",
    hue="Metadata_Plate",  # Assign `x` variable to `hue`
    palette="Set2",
    legend=False,  # Disable the legend in the boxplot
)
g.set_axis_labels("Plate", "PowerLogLogSlope", fontsize=20)
g.set_titles(col_template="{col_name}", size=22)
g.tick_params(labelsize=16)
# Rotate plate names so they do not overlap
for ax in g.axes.flatten():
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
g.add_legend()
plt.tight_layout()
plt.savefig(figure_dir / "powerloglogslope_distribution.png", dpi=300, bbox_inches="tight")
plt.show()


# In[ ]:


from scipy.stats import zscore

# Compute z-scores per channel
df["PowerLogLogSlope_z"] = df.groupby("Channel")[
    "ImageQuality_PowerLogLogSlope"
].transform(zscore)

# Plot
g = sns.FacetGrid(
    df,
    col="Channel",
    col_wrap=3,
    height=6,
    aspect=1.2,
    sharey=True,
)
g.map_dataframe(
    sns.boxplot,
    x="Metadata_Plate",
    y="PowerLogLogSlope_z",
    hue="Metadata_Plate",
    palette="Set2",
    legend=False,
)
# Add red line at z = -2
for ax in g.axes.flatten():
    ax.axhline(-2, color="red", linestyle="--")
# Add red line at z = -1.5
for ax in g.axes.flatten():
    ax.axhline(-1.5, color="orange", linestyle="--")
g.set_axis_labels("Plate", "PowerLogLogSlope (z-score)", fontsize=20)
g.set_titles(col_template="{col_name}", size=22)
g.tick_params(labelsize=16)
# Rotate plate names so they do not overlap
for ax in g.axes.flatten():
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
g.add_legend()
plt.tight_layout()
plt.savefig(figure_dir / "powerloglogslope_zscore_distribution.png", dpi=300, bbox_inches="tight")
plt.show()


# There is no difference in distribution for blur for each channel across plates, so we will run QC thresholds per channel for all plates together.

# In[ ]:


# Find summary statistics for the PowerLogLogSlope metric across all channels and plates
powerloglog_summary = df.groupby("Channel")["ImageQuality_PowerLogLogSlope"].describe()
print(powerloglog_summary)


# In[ ]:


# metadata columns to include in output data frame
metadata_columns = [
    "Metadata_Plate",
    "Metadata_Well",
]


# In[ ]:


# Find DNA channel blur outliers
blurry_DNA_channel_outliers = find_outliers(
    df=orig_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "ImageQuality_PowerLogLogSlope_OrigDNA": -2,
    },
)
# Adjust pandas display options to show longer strings
pd.set_option("display.max_colwidth", None)

# Display the first few rows of the DataFrame
pd.DataFrame(blurry_DNA_channel_outliers).sort_values(
    by="ImageQuality_PowerLogLogSlope_OrigDNA", ascending=False
).head()


# In[ ]:


# Find ER channel blur outliers
blurry_ER_channel_outliers = find_outliers(
    df=orig_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "ImageQuality_PowerLogLogSlope_OrigER": -1.7,
    },
)
# Adjust pandas display options to show longer strings
pd.set_option("display.max_colwidth", None)

# Display the first few rows of the DataFrame
pd.DataFrame(blurry_ER_channel_outliers).sort_values(
    by="ImageQuality_PowerLogLogSlope_OrigER", ascending=False
).head()


# In[ ]:


# Find RNA channel blur outliers
blurry_RNA_channel_outliers = find_outliers(
    df=orig_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "ImageQuality_PowerLogLogSlope_OrigRNA": -1.8,
    },
)
# Adjust pandas display options to show longer strings
pd.set_option("display.max_colwidth", None)

# Display the first few rows of the DataFrame
pd.DataFrame(blurry_RNA_channel_outliers).sort_values(
    by="ImageQuality_PowerLogLogSlope_OrigRNA", ascending=False
).head()


# In[ ]:


# Find Mito channel blur outliers
blurry_Mito_channel_outliers = find_outliers(
    df=orig_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "ImageQuality_PowerLogLogSlope_OrigMito": -2,
    },
)
# Adjust pandas display options to show longer strings
pd.set_option("display.max_colwidth", None)

# Display the first few rows of the DataFrame
pd.DataFrame(blurry_Mito_channel_outliers).sort_values(
    by="ImageQuality_PowerLogLogSlope_OrigMito", ascending=False
).head()


# In[ ]:


# Find AGP channel blur outliers
blurry_AGP_channel_outliers = find_outliers(
    df=orig_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "ImageQuality_PowerLogLogSlope_OrigAGP": -1.7,
    },
)
# Adjust pandas display options to show longer strings
pd.set_option("display.max_colwidth", None)

# Display the first few rows of the DataFrame
pd.DataFrame(blurry_AGP_channel_outliers).sort_values(
    by="ImageQuality_PowerLogLogSlope_OrigAGP", ascending=False
).head()


# ### Identify threshold values to use to find outliers above and below the mean
# 
# **Note:** These values will be used in CellProfiler to flag and not process images.

# In[ ]:


# Initialize a dictionary to store thresholds
raw_thresholds = {}

# List of channels and their corresponding outlier dataframes
channel_outliers = {
    "OrigDNA": blurry_DNA_channel_outliers,
    "OrigER": blurry_ER_channel_outliers,
    "OrigRNA": blurry_RNA_channel_outliers,
    "OrigMito": blurry_Mito_channel_outliers,
    "OrigAGP": blurry_AGP_channel_outliers,
}

# Iterate through each channel and calculate the threshold
for channel, outliers_df in channel_outliers.items():
    column_name = f"ImageQuality_PowerLogLogSlope_{channel}"
    if column_name in outliers_df.columns and not outliers_df.empty:
        # Take the max as the threshold for that channel
        threshold_value = outliers_df[column_name].max()
        raw_thresholds[channel] = threshold_value
    else:
        raw_thresholds[channel] = None  # or np.nan if you prefer

# Print the thresholds in a vertical format
for channel, threshold in raw_thresholds.items():
    print(f"Channel: {channel}, Threshold: {threshold}")


# In[ ]:


# List of channels
channels = ["OrigDNA", "OrigER", "OrigRNA", "OrigMito", "OrigAGP"]

# Make list of PowerLogLogSlope columns corresponding to blur thresholds
powerlog_cols = [f"ImageQuality_PowerLogLogSlope_{ch}" for ch in channels]

# Compute percentage of FOVs failing QC per channel per plate using blur thresholds
fail_summary = []

for plate, plate_df in orig_df.groupby("Metadata_Plate"):
    total_fovs = len(plate_df)
    for ch, col in zip(channels, powerlog_cols):
        threshold = raw_thresholds.get(ch)
        if threshold is not None:
            # FOV fails if below the blur threshold
            num_fail = (plate_df[col] < threshold).sum()
            pct_fail = num_fail / total_fovs * 100
            fail_summary.append(
                {"Metadata_Plate": plate, "Channel": ch, "PctFail": pct_fail}
            )

fail_df = pd.DataFrame(fail_summary)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=fail_df, x="Metadata_Plate", y="PctFail", hue="Channel", palette="Set2"
)
plt.ylabel("Percentage of FOVs failing QC (%)")
plt.xlabel("Plate")
plt.ylim(0, 100)
plt.title("Percentage of FOVs failing QC per channel per plate")
plt.legend(title="Channel")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## Saturation metric
# 
# For saturation metrics, we are looking for:
# 
# - Smudged images or images containing large artifacts
# - Overly saturated channels which can occur in any channel
# 
# This means that we will be setting a threshold for outliers for all channels as the metric PercentMaximal is a range from 0 - 100. We expect if images have 1% of all pixels at the max value to be over-saturated.

# In[ ]:


summary_statistics = df["ImageQuality_PercentMaximal"].describe()
print(summary_statistics)


# In[ ]:


# List of channels to check
channels = ["OrigDNA", "OrigER", "OrigRNA", "OrigMito", "OrigAGP"]

# Make sure PercentMaximal columns are numeric
for channel in channels:
    col = f"ImageQuality_PercentMaximal_{channel}"
    if col in orig_df.columns:
        orig_df[col] = pd.to_numeric(orig_df[col], errors="coerce")

# Columns to keep
percentmax_cols = [f"ImageQuality_PercentMaximal_{ch}" for ch in channels]
keep_cols = ["Metadata_Plate", "Metadata_Well"] + percentmax_cols
file_path_cols = [
    c for c in orig_df.columns if c.startswith("FileName_") or c.startswith("PathName_")
]
keep_cols += file_path_cols

# Select only the rows that fail (>=1) in any PercentMaximal column
df_fails_trimmed = orig_df.loc[
    (orig_df[percentmax_cols] >= 1).any(axis=1), keep_cols
].copy()

# Print shape and percentage of FOVs failing
pct_failing = len(df_fails_trimmed) / len(orig_df) * 100
print(
    f"Total FOVs failing ≥1% in any channel: {len(df_fails_trimmed)} ({pct_failing:.2f}%)"
)

df_fails_trimmed.head()


# In[ ]:


# List of channels to check
channels = ["OrigDNA", "OrigER", "OrigRNA", "OrigMito", "OrigAGP"]
percentmax_cols = [f"ImageQuality_PercentMaximal_{ch}" for ch in channels]

# Make sure PercentMaximal columns are numeric
for col in percentmax_cols:
    if col in orig_df.columns:
        orig_df[col] = pd.to_numeric(orig_df[col], errors="coerce")

# Compute percentage failing per channel per plate
fail_summary = []
for plate, plate_df in orig_df.groupby("Metadata_Plate"):
    total_fovs = len(plate_df)
    for ch, col in zip(channels, percentmax_cols):
        num_fail = (plate_df[col] >= 1).sum()
        pct_fail = num_fail / total_fovs * 100
        fail_summary.append(
            {"Metadata_Plate": plate, "Channel": ch, "PctFail": pct_fail}
        )

fail_df = pd.DataFrame(fail_summary)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=fail_df, x="Metadata_Plate", y="PctFail", hue="Channel", palette="Set2"
)
plt.ylabel("Percentage of FOVs failing QC (%)")
plt.xlabel("Plate")
plt.title("Percentage of FOVs failing QC per channel per plate")
plt.legend(title="Channel")
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ## Generate overall plot with how many FOVs failed QC in general

# In[ ]:


# List of channels
channels = ["OrigDNA", "OrigER", "OrigRNA", "OrigMito", "OrigAGP"]

# Ensure numeric columns
for ch in channels:
    blur_col = f"ImageQuality_PowerLogLogSlope_{ch}"
    sat_col = f"ImageQuality_PercentMaximal_{ch}"
    if blur_col in orig_df.columns:
        orig_df[blur_col] = pd.to_numeric(orig_df[blur_col], errors="coerce")
    if sat_col in orig_df.columns:
        orig_df[sat_col] = pd.to_numeric(orig_df[sat_col], errors="coerce")

# Compute overall % FOVs failing per channel per plate
fail_summary = []

for plate, plate_df in orig_df.groupby("Metadata_Plate"):
    total_fovs = len(plate_df)
    for ch in channels:
        blur_col = f"ImageQuality_PowerLogLogSlope_{ch}"
        sat_col = f"ImageQuality_PercentMaximal_{ch}"
        blur_thresh = raw_thresholds.get(ch, 0)  # default 0 if missing
        sat_thresh = 1  # or whichever % threshold you want for saturation

        # FOV fails if below blur threshold OR above saturation threshold
        fail_mask = (plate_df[blur_col] < blur_thresh) | (
            plate_df[sat_col] >= sat_thresh
        )
        num_fail = fail_mask.sum()
        pct_fail = num_fail / total_fovs * 100

        fail_summary.append(
            {"Metadata_Plate": plate, "Channel": ch, "PctFail": pct_fail}
        )

fail_df = pd.DataFrame(fail_summary)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(
    data=fail_df, x="Metadata_Plate", y="PctFail", hue="Channel", palette="Set2"
)
plt.ylabel("Percentage of FOVs failing overall QC (%)")
plt.xlabel("Plate")
plt.title("Overall QC failure per channel per plate (blur OR saturation)")
plt.legend(title="Channel")
plt.ylim(0, 100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

