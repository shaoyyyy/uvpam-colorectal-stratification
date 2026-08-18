# Strictly Nested Case-Level LOOCV Sensitivity Analysis

## Purpose

This optional analysis evaluates the compact-panel construction procedure when
case-level feature ranking and selection are restricted to the training cases
in each outer leave-one-case-out cross-validation (LOOCV) fold. It complements,
but does not replace, the fixed-panel primary analysis reproduced by
`run_reproduction.py`.

## Scope

The nesting starts from the frozen case-level feature table. In every outer
fold, the script:

1. holds out one case;
2. reranks all 31 descriptors in the outer-training cases using
   orientation-independent single-feature AUC;
3. removes redundant candidates sequentially at absolute Spearman
   correlation greater than or equal to 0.90 and retains six candidates;
4. performs greedy forward selection by inner LOOCV AUC, retaining at most four
   descriptors;
5. fits `StandardScaler` and L2 logistic regression only on the applicable
   training cases; and
6. generates one outer-fold held-out probability.

Equal single-feature AUCs and equal inner-LOOCV AUCs are resolved
alphabetically by internal descriptor name. Forward selection stops when the
best available addition decreases inner-LOOCV AUC; an equal AUC is retained.

Image-level feature extraction and the pooled intensity thresholds of 42 and
150 are fixed upstream and are not re-estimated in each outer fold. This is
therefore a strictly nested case-level feature-selection and modeling
sensitivity analysis, not a fully end-to-end nested image-processing pipeline.

## Run

From the package root:

```bash
python optional_nested_analysis/run_nested_loocv_sensitivity.py
```

The script requires no path editing and writes:

```text
optional_nested_analysis/generated_results/
├── nested_loocv_summary.csv
├── nested_loocv_fold_predictions.csv
└── nested_feature_selection_stability.csv
```

Frozen comparison files are stored in
`optional_nested_analysis/reference_results/`.

## Expected Results

| Task | Outer-LOOCV AUC | 95% bootstrap CI |
|---|---:|---:|
| Malignant vs non-malignant | 0.539 | 0.233-0.834 |
| Inflammatory vs hyperplastic | 0.633 | 0.300-0.927 |
| Tubular adenoma vs non-neoplastic | 0.677 | 0.417-0.915 |

Intervals use 2,000 valid case-level bootstrap resamples with random seed 0.
Single-class draws are rejected and redrawn.
