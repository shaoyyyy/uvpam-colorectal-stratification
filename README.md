# UV-PAM Case-Level Analysis Reproduction Package

## 1. Package purpose

This package accompanies the manuscript **“266-nm ultraviolet photoacoustic microscopy for label-free stratification of colorectal lesions.”**

It reproduces the reported case-level feature aggregation, task-specific
single-feature ranking, fixed-panel and baseline model evaluation,
leave-one-case-out predictions, ROC/AUC analysis, bootstrap confidence
intervals, and paired AUC comparisons from de-identified derived feature
tables.

Raw UV-PAM and H&E images and upstream acquisition or reconstruction software
are outside the present package scope. De-identified images may be made
available separately through the corresponding authors under applicable
institutional data-sharing procedures.

### Code provenance

The core feature-ranking, redundancy-handling, fixed-panel verification,
LOOCV, bootstrap, and paired-AUC implementation was retained from the archived
case-level `analysis_script.py` used for the final reanalysis. Packaging changes
were limited to de-identifying the shared input table, adding strict input
checks, reading the locked panel definitions for final evaluation, exporting
ROC coordinates, standardizing output names, and verifying results against
frozen reference values. These packaging changes do not alter the numerical
modeling procedure.

The script in `optional_nested_analysis/` is a separate sensitivity-analysis
implementation constructed to reproduce Supplementary Note S2. It does not
replace or modify the archived code path used for the primary fixed-panel
results.

## 2. Scope of reproducibility

The executable workflow starts from `data/ROI_level_features.csv` and covers:

1. validation of 148 unique ROI records and 31 nonredundant candidate features;
2. mean aggregation of multiple ROIs within each case;
3. verification of 29 unique case-level records;
4. construction of the three reported lesion-comparison tasks;
5. orientation-independent single-feature AUC ranking;
6. evaluation of four feature-family baselines, a 31-feature baseline, and a
   task-specific fixed compact panel;
7. training-fold-only standardization and logistic regression under LOOCV;
8. out-of-fold probability generation;
9. ROC/AUC calculation and 2,000-valid-resample case-level bootstrap
   confidence intervals;
10. paired AUC differences using identical resampled case indices for the two
    models in each comparison; and
11. automatic generation and verification of the numerical outputs.

The package intentionally excludes:

- laser-scanning and stage-control software;
- FPGA-control software;
- raw A-line signal reconstruction;
- H&E registration and manual ROI-selection software;
- raw patient UV-PAM and H&E images;
- manuscript or LaTeX source files;
- serialized model files and obsolete analysis versions.

## 3. Folder structure

```text
04_UVPAM_Reproducibility_Package/
├── README.md
├── requirements.txt
├── run_reproduction.py
├── data/
│   ├── ROI_level_features.csv
│   ├── Case_level_features.csv
│   └── case_metadata.csv
├── configuration/
│   ├── feature_family_definitions.json
│   ├── fixed_panel_definitions.json
│   ├── redundancy_group_definitions.json
│   ├── analysis_parameters.json
│   └── task_definitions.json
├── feature_selection/
│   ├── taskwise_single_feature_auc_rankings.csv
│   ├── panel_selection_log.csv
│   └── taskwise_spearman_correlations.csv
├── reference_results/
│   ├── oof_predictions.csv
│   ├── performance_summary.csv
│   ├── paired_auc_differences.csv
│   ├── case_aggregation_check.csv
│   └── expected_results.json
├── optional_nested_analysis/
│   ├── README_nested_analysis.md
│   ├── run_nested_loocv_sensitivity.py
│   └── reference_results/
│       ├── nested_loocv_summary.csv
│       ├── nested_loocv_fold_predictions.csv
│       └── nested_feature_selection_stability.csv
└── optional_outputs/
    ├── roc_curve_data.csv
    └── generated_tables/
```

Running the script creates a separate `reproduced_results/` folder so that the
frozen reference results are not overwritten.

## 4. Input data description

### `data/ROI_level_features.csv`

The direct input to the executable analysis. It contains:

- a package-specific unique ROI identifier;
- a de-identified case identifier;
- the pathology category; and
- 31 nonredundant UV-PAM-derived quantitative features.

Original image filenames, patient identifiers, pathology numbers, image
dimensions, notes, and other non-model metadata have been removed.

### `data/Case_level_features.csv`

The reference case-level table obtained by arithmetic-mean aggregation of all
ROIs belonging to the same case. It contains one row per case and is used only
to verify the aggregation reproduced from the ROI-level input.

### `data/case_metadata.csv`

Records pathology, ROI count, task inclusion, and binary label encoding for
each de-identified case. ROI count and metadata are not used as predictors.

`CavityRatio` and `SignalAreaRatio` are not model inputs. The former duplicates
`LowSignalRatio`, and the latter is its exact complement. Their exclusion is
documented in the configuration files.

## 5. Installation

Python 3.11 or later is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 6. How to run

From the package root:

```bash
python run_reproduction.py
```

The script requires no manual path editing. An alternative output location can
be supplied with:

```bash
python run_reproduction.py --output-dir path/to/output
```

To run the strictly nested case-level sensitivity analysis:

```bash
python optional_nested_analysis/run_nested_loocv_sensitivity.py
```

Its selection rules, scope, outputs, and expected results are documented in
`optional_nested_analysis/README_nested_analysis.md`.

## 7. Analysis implementation

The three binary tasks are defined in
`configuration/task_definitions.json`:

- adenocarcinoma versus all non-malignant categories;
- inflammatory polyp versus hyperplastic polyp; and
- tubular adenoma versus inflammatory plus hyperplastic polyps.

Single-feature ranking uses:

```text
orientation-independent AUC = max(raw AUC, 1 - raw AUC)
```

Final compact-panel evaluation reads the fixed panels from
`configuration/fixed_panel_definitions.json`. It does not automatically search
for or optimize a new best panel.

All six input strategies use the same classifier and the same cases:

- Intensity-distribution baseline;
- GLCM texture baseline;
- Signal-area baseline;
- Low-signal morphology baseline;
- Full-feature baseline using all 31 nonredundant features; and
- Task-specific compact panel.

Within every LOOCV fold, one case is held out. `StandardScaler` and
`LogisticRegression` are fitted using the remaining cases only, and the
training-fold transformation is then applied to the held-out case.

The locked classifier parameters are:

```python
LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="liblinear",
    class_weight="balanced",
    max_iter=1000,
    random_state=0,
)
```

## 8. Generated outputs

`reproduced_results/` contains:

- `Case_level_features.csv`;
- `case_metadata.csv`;
- `case_aggregation_check.csv`;
- `taskwise_single_feature_auc_rankings.csv`;
- `taskwise_spearman_correlations.csv`;
- `oof_predictions_reproduced.csv`;
- `performance_summary.csv`;
- `paired_auc_differences.csv`;
- `roc_curve_data.csv`; and
- `run_summary.json`.

The OOF prediction table is the primary reusable result. It permits independent
recalculation of ROC curves, AUCs, confidence intervals, and paired AUC
differences.

## 9. Expected numerical results

The expected task-specific compact-panel results are:

| Task | AUC | 95% bootstrap CI |
|---|---:|---:|
| Malignant vs non-malignant | 0.831 | 0.583–1.000 |
| Inflammatory vs hyperplastic | 0.900 | 0.667–1.000 |
| Tubular adenoma vs non-neoplastic | 0.865 | 0.643–1.000 |

The exact unrounded values are stored in
`reference_results/expected_results.json`. The script stops with an error if
the reproduced compact-panel values do not match the frozen expected values.

The optional strictly nested case-level sensitivity analysis is expected to
produce AUCs of 0.539, 0.633, and 0.677 for the three tasks, respectively.
Frozen nested summaries, outer-fold predictions, and selection frequencies are
stored in `optional_nested_analysis/reference_results/`.

## 10. Software versions

The tested environment is defined in `requirements.txt`:

- Python 3.12;
- NumPy 2.3.5;
- SciPy 1.17.0; and
- scikit-learn 1.8.0.

## 11. Data and interpretation limitations

- The sample size is 29 cases, and all reported estimates are exploratory
  same-cohort internal-validation results.
- Task-specific feature ranking and the documented fixed panels were derived
  from the analyzed cohort and were not nested within LOOCV.
- The fixed panels are therefore reproduced rather than reselected during
  final evaluation.
- Raw images are outside the present case-level package scope; de-identified
  images may be made available separately under applicable institutional
  data-sharing procedures.
- The shared feature tables are de-identified derived data and cannot be used
  to reconstruct patient images or identities.
- Image-level feature extraction is outside the present package scope; feature
  definitions and parameters are documented in the manuscript supplementary
  material.

## 12. Contact

Questions about the manuscript or package may be addressed to:

- Lei Liang: `lianglei@xupt.edu.cn`
- Chen Chang: `changchen9185@163.com`
