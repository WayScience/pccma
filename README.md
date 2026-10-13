<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./logo/pccma_logo_transparent_on_dark.svg">
    <img src="./logo/pccma_logo_light.svg" alt="PCCMA: Pediatric Cancer Cell Morphology Atlas" width="800">
  </picture>
</p>

## The atlas cell lines

The Pediatric Cancer Cell Morphology Atlas (PCCMA) profiles 56 cell lines: 52 pediatric cancer lines and 4 normal reference lines.
The metadata for every line is in [`cell_lines/pccma_cell_line_metadata.csv`](./cell_lines/pccma_cell_line_metadata.csv), and the figures below summarize it.

### Where the cancers arise

![Cell lines on a child-proportioned body, grouped by tissue site and cancer type](./cell_lines/notebooks/figures/cancer_type_anatogram_labels.png)

The atlas spans eight tissue sites, shown here at the organ where each tumor type typically arises.
The adrenal gland and bone are the largest groups with 12 lines each: the adrenal gland lines are all neuroblastoma, and bone has eight from the Ewing sarcoma family and four osteosarcoma.
The central nervous system follows with 11 lines (five medulloblastoma, four glioma, and one each of atypical teratoid rhabdoid tumor and ependymoma).
Soft tissue (nine lines, mostly rhabdomyosarcoma), the eye (four retinoblastoma), the kidney (two malignant rhabdoid tumor and one Wilms tumor), and a single ovarian germ cell tumor complete the cancer lines.
The four normal reference lines (colon, lung, microglia, and prostate) are drawn in gray and serve as non-cancer comparators.

### How the lines are described

![Sunburst of tissue site, origin, subtype, and plate condition](./cell_lines/notebooks/figures/cancer_type_sunburst.png)

Each ring of the sunburst subdivides the one inside it, from tissue site at the center out to the plate condition of each line.
Moving outward, a site is split by the origin of the line (whether it came from a primary tumor or a metastasis), then by molecular subtype where one is annotated, and finally by the plating condition used in the atlas (standard, Synthemax, Synthemax with double PFA fixation, laminin, or laminin with double PFA fixation).
Many lines are labeled unknown for origin or subtype because the source records do not report them; a subtype is annotated for 20 of the 56 lines.
Where a subtype is known it is informative, such as MYCN amplification in neuroblastoma, the EWS fusion partners in Ewing sarcoma, and PAX3-FOXO1 in rhabdomyosarcoma.
Of the 62 line-and-condition entries in the metadata table (a line can be plated under more than one condition), 43 are on Synthemax (25 of them with double PFA fixation), 16 are standard, and 3 are on laminin.

### Sex and age

![Sex and age of the cell lines by cancer group](./cell_lines/notebooks/figures/sex_age.png)

The left panels count the lines in each cancer group by the sex of the patient they came from, and the right panels place each line at the age of the patient, with a tick at the median for each group.
Where sex is recorded, the atlas is roughly balanced between female (23 lines) and male (20 lines) donors, while 13 lines have no recorded sex.
Age is known for 41 of the 56 lines and the histogram at the top shows that the donors are young, with a median age of 6 years.
Neuroblastoma and malignant rhabdoid tumor lines come from the youngest donors (medians of 2 years and about half a year), whereas the bone tumors are from adolescents, with medians of 13 to 14 years.
Retinoblastoma lines have no recorded sex or age.
The fetal normal reference lines are plotted at age 0, and the prostate line WPMY-1, from a 54-year-old donor, is beyond the axis and is drawn at its right edge.
