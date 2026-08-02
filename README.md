# uvpam-colorectal-stratification

This repository contains the analysis code and configuration files supporting the manuscript:

> **266-nm ultraviolet photoacoustic microscopy for label-free stratification of colorectal lesions**

This repository contains the Python source, locked method configuration, and aggregate reference results for the case-level UV-PAM feature-panel analysis. It is intentionally a **code-only public release**: the underlying feature tables, case/ROI metadata, raw UV-PAM images, H&E images, and individual prediction rows are not included.

## Analytical scope

The code documents and implements the manuscript workflow:

1. extraction of interpretable intensity-distribution, GLCM texture, signal-area, and low-signal morphology descriptors from UV-PAM ROIs;
2. arithmetic averaging of multiple ROIs within each case;
3. task-wise orientation-independent single-feature AUC ranking, defined as `max(AUC, 1 - AUC)`;
4. documented redundancy reduction and construction of fixed four-feature panels;
5. L2-regularized logistic regression with training-fold-only `StandardScaler` fitting;
6. case-level leave-one-case-out cross-validation (LOOCV);
7. AUC estimation, 2,000-valid-resample case-level bootstrap confidence intervals, and paired bootstrap comparisons.

The three manuscript tasks are:

- malignant versus non-malignant lesions;
- inflammatory versus hyperplastic polyps;
- tubular adenomas versus non-neoplastic polyps.

## Validation-scope note

The supplied analysis is **not strict nested feature selection**. The task-specific single-feature ranking, panel construction, and pooled low/high-signal threshold estimation used the complete analyzed cohort. The panels were then fixed before LOOCV. Within each LOOCV fold, only standardization and logistic-regression fitting used the training cases.

Accordingly, the reported AUCs are exploratory same-cohort internal-validation estimates and may be optimistic. No audited source package contained an implementation that repeated panel construction and threshold estimation within every outer training fold, so this release does not claim a strictly nested sensitivity analysis.

## Repository contents

```text
.
|-- README.md
|-- LICENSE
|-- CITATION.cff
|-- MANIFEST.md
|-- SHA256SUMS.txt
|-- requirements.txt
|-- scripts/
|   |-- analysis_script.py
|   |-- extract_roi_features_from_images.py
|   |-- run_feature_based_reanalysis.py
|   |-- make_selected_feature_boxplots.py
|   `-- audit_roi_and_bootstrap_spec.py
|-- configuration/
|   |-- analysis_parameters.json
|   |-- feature_extraction_parameters.json
|   |-- fixed_panel_definitions.json
|   `-- redundancy_group_definitions.json
|-- documentation/
|   |-- INPUT_DATA_POLICY.md
|   |-- feature_definitions.csv
|   |-- software_environment.txt
|   `-- archived_reproducibility_note.txt
`-- reference_results/
    |-- expected_results.json
    |-- generated_panel_definitions.json
    |-- performance_summary.csv
    |-- paired_auc_differences.csv
    `-- feature_selection/
        |-- taskwise_single_feature_auc_rankings.csv
        |-- panel_selection_log.csv
        `-- taskwise_spearman_correlations.csv
```

`MANIFEST.md` records the source archive and SHA-256 digest of each retained manuscript-analysis script and documents the excluded material.

## Script roles

- `scripts/analysis_script.py` is the final case-level panel-construction and evaluation implementation. It regenerates rankings, panel-selection records, fixed-panel checks, LOOCV predictions, AUC summaries, bootstrap intervals, paired comparisons, and aggregation checks when approved input tables are available.
- `scripts/extract_roi_features_from_images.py` is the archived feature-extraction implementation used with restricted ROI images and metadata.
- `scripts/run_feature_based_reanalysis.py` is the manuscript reanalysis implementation retained from the final JBO submission package. It evaluates fixed panels and baselines and generates supporting tables and figures.
- `scripts/make_selected_feature_boxplots.py` generates the selected-feature box-plot data and figures from restricted case-level inputs.
- `scripts/audit_roi_and_bootstrap_spec.py` checks ROI dimensions and prints the stored bootstrap specification from restricted-input runs.

These files retain their archived relative-path conventions. They do not contain embedded feature rows, patient identifiers, or absolute user-machine paths.

## Environment

The recorded primary environment was Python 3.13.5 with NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0, pandas 2.2.3, and Matplotlib 3.10.8. Install the public package requirements with:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduction with approved data access

The public repository cannot execute the complete analysis without the restricted inputs. Researchers who have received institutional approval and the required de-identified tables can use the archived layouts below.

For the final case-level script, place the approved files as:

```text
data/
|-- ROI_level_features.csv
`-- Case_level_features.csv
```

Then run from the repository root:

```bash
python scripts/analysis_script.py --package-root .
```

Outputs are written to `results/`, which is excluded from version control because it contains case-level records.

The supporting reanalysis expects approved files under `inputs/` with the names documented in `documentation/INPUT_DATA_POLICY.md`. The feature-extraction script expects restricted ROI images under `roi_images/` and a restricted `metadata.csv` file. These locations are excluded by `.gitignore`.

## Manuscript-matched reference results

The fixed compact panels yielded:

| Task | AUC | 95% case-level bootstrap CI |
|---|---:|---:|
| Malignant vs non-malignant | 0.831 | 0.583-1.000 |
| Inflammatory vs hyperplastic | 0.900 | 0.667-1.000 |
| Tubular adenoma vs non-neoplastic | 0.865 | 0.643-1.000 |

Machine-readable full-precision values are provided in `reference_results/expected_results.json`. The other public CSV files contain task-, model-, or feature-level aggregate summaries only; they do not contain case IDs, ROI IDs, filenames, individual feature vectors, or individual out-of-fold scores.

## Data availability

The underlying de-identified feature tables and raw UV-PAM and H&E images are not publicly available because of ethical and institutional data-sharing restrictions. De-identified data may be made available to qualified researchers upon reasonable request to the corresponding authors and subject to applicable institutional approval.

## Citation and license

Citation metadata are provided in `CITATION.cff`. The code is released under the MIT License. The license applies to the repository code and documentation only; it does not grant access to or rights in the restricted clinical, imaging, or derived datasets.

