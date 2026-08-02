# Public release manifest

Release: `JBO GitHub public code repository v1.0`  
Repository directory: `uvpam-colorectal-stratification`  
Audit date: 2026-08-02  
Associated manuscript: *266-nm ultraviolet photoacoustic microscopy for label-free stratification of colorectal lesions*

## Retained manuscript-analysis source

The following files are byte-for-byte copies of archived Python sources. SHA-256 digests are calculated from the files in this public release.

| Public path | Source archive and entry | Role | SHA-256 |
|---|---|---|---|
| `scripts/analysis_script.py` | `UVPAM_caselevel_reproducibility_package_final.zip` -> `UVPAM_caselevel_reproducibility_package/scripts/analysis_script.py` | Final case-level ranking, panel construction, LOOCV, bootstrap, and aggregate result generation | `a43fc47677514859d5f62429f7bff15d7074b8fe46c329bf41c8b8e8ed16f347` |
| `scripts/run_feature_based_reanalysis.py` | `JBO_Final_Submission_Package_20260711_v1.zip` -> `03_reproducibility/feature_based_reanalysis/scripts/run_feature_based_reanalysis.py` | Fixed-panel and baseline reanalysis used in the JBO submission package | `b3c2f020cf6a9951c818111fa6390f743f4c47fa746c061a6f6f519354e86a88` |
| `scripts/extract_roi_features_from_images.py` | `JBO_Final_Submission_Package_20260711_v1.zip` -> `03_reproducibility/feature_based_reanalysis/scripts/extract_roi_features_from_images.py` | ROI image feature extraction | `634cf660c50e910db71d7f0e0becc8f3fb78fd7416c4f5d9cc211e859e9eba56` |
| `scripts/make_selected_feature_boxplots.py` | `JBO_Final_Submission_Package_20260711_v1.zip` -> `03_reproducibility/feature_based_reanalysis/scripts/make_selected_feature_boxplots.py` | Selected-feature plot generation | `ccb13ee011d1448f58477e6b9f5bc58c7a4ad579bd41a8ee0518af256b6aec55` |
| `scripts/audit_roi_and_bootstrap_spec.py` | `UVPAM_verified_bootstrap_ROI_audit.zip` -> `UVPAM_verified_bootstrap_ROI_audit/scripts/audit_roi_and_bootstrap_spec.py` | ROI-dimension and bootstrap-specification audit | `62010345000b38895b1faa28155cbac562babb7af7331145281c525200667437` |

The feature-extraction source in the final JBO submission package has the same SHA-256 digest as the corresponding source in `UVPAM_verified_bootstrap_ROI_audit.zip` and `PG_IQPA_verified_reanalysis_package.zip`.

## Retained configuration

- `configuration/analysis_parameters.json`: classifier, standardization, LOOCV, bootstrap, aggregation, and ranking specifications.
- `configuration/feature_extraction_parameters.json`: archived image-feature extraction thresholds and parameters.
- `configuration/fixed_panel_definitions.json`: the three locked four-feature panels used by `analysis_script.py`.
- `configuration/redundancy_group_definitions.json`: algebraic exclusions, semantic redundancy groups, tie handling, and the explicit non-nested validation-scope note.

These files contain method parameters and task-level definitions only. They contain no case IDs, ROI IDs, filenames, individual rows, or absolute paths.

## Retained public-safe reference artifacts

- `reference_results/expected_results.json`: full-precision compact-panel AUC and CI checks plus manuscript-rounded values.
- `reference_results/performance_summary.csv`: task/model aggregate AUCs and confidence intervals.
- `reference_results/paired_auc_differences.csv`: task/model aggregate paired-bootstrap differences.
- `reference_results/generated_panel_definitions.json`: regenerated panel definitions and locked-panel match flag.
- `reference_results/feature_selection/taskwise_single_feature_auc_rankings.csv`: task/feature aggregate rankings and descriptive statistics.
- `reference_results/feature_selection/panel_selection_log.csv`: task/feature selection decisions and reasons.
- `reference_results/feature_selection/taskwise_spearman_correlations.csv`: task/feature-pair aggregate correlations.
- `documentation/feature_definitions.csv`: method-level feature definitions used in the supplementary material.
- `documentation/software_environment.txt`: archived software versions.

Every retained CSV was parsed and checked for a consistent column count. None contains `case_id`, `roi_id`, filename, patient/subject ID, or individual prediction-score columns.

## Audited source packages

The source inventory covered the available manuscript and analysis packages relevant to this release:

- `UVPAM_caselevel_reproducibility_package_final.zip`;
- `JBO_Final_Submission_Package_20260711_v1.zip`;
- `JBO_complete_manuscript_and_reanalysis.zip`;
- `UVPAM_verified_bootstrap_ROI_audit.zip`;
- `PG_IQPA_verified_reanalysis_package.zip`;
- `pg_iqpa_reanalysis_package.zip`;
- `UVPAM_feature_family_method_redesign.zip`;
- `UVPAM_reconstructed_original_selection.zip`;
- `UVPAM_GitHub_Repository_ready.zip`.

The three synced `submission_ready_figures_*.zip` packages were also inventoried. They contained only TIFF/PNG figure assets and, in one package, an aggregate figure-summary CSV; they contained no Python source and were excluded from this code repository.

## Deliberately excluded data and metadata

All copies and naming variants of the following were excluded, including duplicates inside nested ZIP files:

- ROI-level and case-level feature tables;
- `metadata.csv`, `metadata_used.csv`, and case metadata;
- raw, reconstructed, cropped, or rendered UV-PAM/H&E images;
- ROI maps and image filenames;
- individual out-of-fold predictions and scores;
- case-level aggregation checks;
- case-level box-plot values;
- ROI-dimension audit rows;
- serialized estimators, caches, compiled Python files, and nested archives.

The excluded feature tables remain derived human-specimen data even when their case labels are pseudonymous. They were not copied, anonymized again, or summarized at the individual level for this release.

## Deliberately excluded historical or non-final code

- `run_pg_iqpa_reanalysis.py`: earlier naming/method version superseded by the manuscript-matched feature-based reanalysis.
- `run_feature_family_analysis.py`: method-redesign experiment, not the submitted primary workflow.
- `run_reconstructed_auc_search.py`: historical reconstruction/AUC search, not the submitted primary workflow.
- `make_metadata_template.py`: metadata utility rather than an analysis source; one archived version also contained a machine-specific absolute path.
- the earlier packaged `run_reproduction.py`: a post hoc repository-packaging derivative bundled with restricted data; the underlying archived final sources are retained instead.

## Validation terminology finding

The audited scripts consistently state that task-specific feature panels and pooled thresholds were derived from the complete cohort and were not re-estimated within each LOOCV training fold. Standardization and classifier fitting were performed within the training fold. No archived source implemented strict nested feature/panel selection. The README and public manifest therefore use the manuscript-matched term **fixed-panel case-level LOOCV** and do not claim a strictly nested sensitivity analysis.

## Release validation performed

Before packaging:

1. all five Python sources were parsed successfully with Python's abstract-syntax-tree parser;
2. all six JSON files were parsed successfully;
3. all seven CSV files were parsed and checked for consistent row widths and sensitive header names;
4. filenames and extensions were screened for datasets, metadata, images, office documents, serialized arrays/models, compiled code, PDFs, and nested archives;
5. text was screened for concrete case tokens, patient/subject identifiers, email addresses, and machine-specific absolute paths;
6. the final ZIP central directory was separately checked for prohibited names, file types, duplicate paths, and nested archives;
7. the final ZIP was extracted to a clean temporary directory and compared byte-for-byte with the staged repository.

`SHA256SUMS.txt` supplies file-level checksums for the public release contents. The final ZIP checksum is reported outside the archive at delivery.

