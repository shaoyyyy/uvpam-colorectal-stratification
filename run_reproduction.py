#!/usr/bin/env python3
"""Reproduce the case-level UV-PAM feature-panel analysis.

Outputs:
- taskwise_single_feature_auc_rankings.csv
- taskwise_spearman_correlations.csv
- panel_selection_log.csv
- generated_panel_definitions.json
- oof_predictions_reproduced.csv
- performance_summary.csv
- paired_auc_differences.csv
- roc_curve_data.csv
- case_aggregation_check.csv

Important: feature ranking and panel construction use the complete cohort and are
not nested within LOOCV, matching the exploratory method reported in the manuscript.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import mannwhitneyu, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

INTENSITY = ['Mean','StdDev','Median','Min','Max','P10','P25','P75','P90','IQR','Skewness','Kurtosis','GrayEntropy']
GLCM = ['GLCM_contrast','GLCM_dissimilarity','GLCM_homogeneity','GLCM_ASM','GLCM_energy','GLCM_correlation','GLCM_entropy']
SIGNAL_AREA = ['LowSignalRatio','HighSignalRatio']
LOW_MORPH = ['LowObj_Count','LowObj_Density_per_10kpx','LowObj_MeanArea_px','LowObj_MedianArea_px','LowObj_MeanCircularity','LowObj_MeanSolidity','LowObj_MeanEccentricity','LowObj_MeanAspectRatio','LowObj_TotalAreaRatio_filtered']
FEATURES = INTENSITY + GLCM + SIGNAL_AREA + LOW_MORPH
FEATURE_FAMILY = {f:'intensity-distribution' for f in INTENSITY}
FEATURE_FAMILY.update({f:'GLCM texture' for f in GLCM})
FEATURE_FAMILY.update({f:'signal-area' for f in SIGNAL_AREA})
FEATURE_FAMILY.update({f:'low-signal morphology' for f in LOW_MORPH})

TASKS = {
    'malignant_vs_non_malignant': {
        'display': 'Malignant vs non-malignant',
        'include_pathologies': ['adenocarcinoma','tubular_adenoma','inflammatory_polyp','hyperplastic_polyp'],
        'positive_pathologies': ['adenocarcinoma'],
        'positive_label': 'Malignant',
        'negative_label': 'Non-malignant',
        'expected_positive_cases': 7,
        'expected_negative_cases': 22,
    },
    'inflammatory_vs_hyperplastic': {
        'display': 'Inflammatory vs hyperplastic',
        'include_pathologies': ['inflammatory_polyp','hyperplastic_polyp'],
        'positive_pathologies': ['inflammatory_polyp'],
        'positive_label': 'Inflammatory',
        'negative_label': 'Hyperplastic',
        'expected_positive_cases': 10,
        'expected_negative_cases': 6,
    },
    'tubular_adenoma_vs_non_neoplastic': {
        'display': 'Tubular adenoma vs non-neoplastic',
        'include_pathologies': ['tubular_adenoma','inflammatory_polyp','hyperplastic_polyp'],
        'positive_pathologies': ['tubular_adenoma'],
        'positive_label': 'Tubular adenoma',
        'negative_label': 'Non-neoplastic',
        'expected_positive_cases': 6,
        'expected_negative_cases': 16,
    },
}

LR_PARAMS = dict(
    penalty='l2',
    solver='liblinear',
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    random_state=0,
)
N_BOOTSTRAP = 2000
RANDOM_SEED = 0


def read_csv_numeric(path: Path) -> Tuple[List[dict], List[str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def write_csv(path: Path, rows: List[dict], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def subset_task(rows: List[dict], task_key: str) -> Tuple[List[dict], np.ndarray]:
    cfg = TASKS[task_key]
    selected = [r for r in rows if r['pathology'] in cfg['include_pathologies']]
    y = np.asarray([1 if r['pathology'] in cfg['positive_pathologies'] else 0 for r in selected], dtype=int)
    if int(np.sum(y == 1)) != cfg['expected_positive_cases']:
        raise RuntimeError(f'Unexpected positive case count for {task_key}.')
    if int(np.sum(y == 0)) != cfg['expected_negative_cases']:
        raise RuntimeError(f'Unexpected negative case count for {task_key}.')
    return selected, y


def feature_values(rows: List[dict], feature: str) -> np.ndarray:
    return np.asarray([float(r[feature]) for r in rows], dtype=float)


def fast_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(np.sum(y_true == 1)); n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    ranks = rankdata(scores, method='average')
    return float((np.sum(ranks[y_true == 1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg))


def ranking_rows(case_rows: List[dict]) -> List[dict]:
    output = []
    for task_key, cfg in TASKS.items():
        rows, y = subset_task(case_rows, task_key)
        task_records = []
        for feature in FEATURES:
            x = feature_values(rows, feature)
            raw_auc = float(roc_auc_score(y, x))
            oriented_auc = max(raw_auc, 1.0 - raw_auc)
            pos = x[y == 1]; neg = x[y == 0]
            p_value = float(mannwhitneyu(pos, neg, alternative='two-sided').pvalue)
            direction = 'higher in positive group' if raw_auc > 0.5 else ('lower in positive group' if raw_auc < 0.5 else 'no direction')
            task_records.append({
                'task': task_key,
                'task_display': cfg['display'],
                'feature': feature,
                'feature_family': FEATURE_FAMILY[feature],
                'raw_auc': raw_auc,
                'orientation_independent_auc': oriented_auc,
                'direction': direction,
                'positive_mean': float(np.mean(pos)),
                'negative_mean': float(np.mean(neg)),
                'positive_median': float(np.median(pos)),
                'negative_median': float(np.median(neg)),
                'mann_whitney_p_descriptive': p_value,
                'n_positive': int(np.sum(y == 1)),
                'n_negative': int(np.sum(y == 0)),
            })
        # Stable primary ordering; tie priorities are applied during panel construction.
        task_records.sort(key=lambda r: (-r['orientation_independent_auc'], FEATURES.index(r['feature'])))
        for rank, rec in enumerate(task_records, start=1):
            rec['rank'] = rank
            output.append(rec)
    return output


def correlation_rows(case_rows: List[dict]) -> List[dict]:
    out = []
    for task_key, cfg in TASKS.items():
        rows, _ = subset_task(case_rows, task_key)
        for i, f1 in enumerate(FEATURES):
            x1 = feature_values(rows, f1)
            for f2 in FEATURES[i+1:]:
                x2 = feature_values(rows, f2)
                rho, p = spearmanr(x1, x2)
                out.append({
                    'task': task_key, 'task_display': cfg['display'],
                    'feature_1': f1, 'feature_2': f2,
                    'spearman_rho': float(rho), 'spearman_p_descriptive': float(p),
                    'abs_spearman_rho': float(abs(rho)),
                })
    return out


def semantic_group_map(redundancy: dict) -> Tuple[Dict[str, str], Dict[str, dict]]:
    member_to_group = {}
    groups = {}
    for g in redundancy['semantic_redundancy_groups']:
        groups[g['group_id']] = g
        for m in g['members']:
            member_to_group[m] = g['group_id']
    return member_to_group, groups


def task_priority(task_key: str, feature: str, redundancy: dict) -> Tuple[int, int]:
    rules = redundancy.get('task_specific_equal_auc_priorities', {}).get(task_key, [])
    for rule_index, rule in enumerate(rules):
        if feature in rule['features']:
            return (rule_index, rule['preferred_order'].index(feature))
    return (9999, FEATURES.index(feature))


def order_tie_group(task_key: str, records: List[dict], redundancy: dict) -> List[dict]:
    member_to_group, groups = semantic_group_map(redundancy)
    def key(rec: dict):
        f = rec['feature']
        # First honor task-specific boundary priorities.
        task_key_value = task_priority(task_key, f, redundancy)
        if task_key_value[0] < 9999:
            return (0, task_key_value[0], task_key_value[1], FEATURES.index(f))
        # Then honor preferred descriptor inside an equal-AUC redundancy group.
        gid = member_to_group.get(f)
        if gid:
            pref = groups[gid].get('preferred_if_auc_equal')
            if f == pref:
                return (1, 0, 0, FEATURES.index(f))
            return (1, 0, 1, FEATURES.index(f))
        return (2, 0, 0, FEATURES.index(f))
    return sorted(records, key=key)


def group_auc_ties(records: List[dict], tol: float) -> List[List[dict]]:
    groups: List[List[dict]] = []
    for rec in records:
        if not groups or abs(rec['orientation_independent_auc'] - groups[-1][0]['orientation_independent_auc']) > tol:
            groups.append([rec])
        else:
            groups[-1].append(rec)
    return groups


def select_panels(rankings: List[dict], redundancy: dict) -> Tuple[Dict[str, List[str]], List[dict]]:
    member_to_group, groups = semantic_group_map(redundancy)
    tol = float(redundancy.get('auc_tie_tolerance', 1e-12))
    panels = {}
    logs = []
    for task_key in TASKS:
        task_records = [dict(r) for r in rankings if r['task'] == task_key]
        task_records.sort(key=lambda r: (-r['orientation_independent_auc'], FEATURES.index(r['feature'])))
        selected: List[str] = []
        selected_group_member: Dict[str, str] = {}
        processed = set()
        for tie_group in group_auc_ties(task_records, tol):
            ordered = order_tie_group(task_key, tie_group, redundancy)
            for rec in ordered:
                f = rec['feature']; processed.add(f)
                gid = member_to_group.get(f, '')
                if len(selected) >= 4:
                    reason = 'Panel already contained four descriptors.'
                    # Make task-specific tie-boundary decisions explicit.
                    for rule in redundancy.get('task_specific_equal_auc_priorities', {}).get(task_key, []):
                        if f in rule['features']:
                            reason = 'Not retained at an equal-AUC boundary: ' + rule['reason']
                            break
                    logs.append({**rec, 'decision':'not retained', 'redundancy_group':gid, 'decision_reason':reason})
                    continue
                if gid and gid in selected_group_member:
                    chosen = selected_group_member[gid]
                    logs.append({**rec, 'decision':'excluded as redundant', 'redundancy_group':gid,
                                 'decision_reason':f'Overlaps with retained descriptor {chosen}; only one member of redundancy group {gid} was retained.'})
                    continue
                selected.append(f)
                if gid:
                    selected_group_member[gid] = f
                reason = 'Retained as the next highest-ranking nonredundant descriptor.'
                for rule in redundancy.get('task_specific_equal_auc_priorities', {}).get(task_key, []):
                    if f in rule['features'] and len(rule['features']) > 1:
                        reason = 'Retained at an equal-AUC boundary: ' + rule['reason']
                        break
                if gid:
                    reason += ' ' + groups[gid]['reason']
                logs.append({**rec, 'decision':'retained', 'redundancy_group':gid, 'decision_reason':reason})
        panels[task_key] = selected[:4]
    return panels, logs


def fit_predict_loocv(rows: List[dict], y: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    X = np.asarray([[float(r[f]) for f in feature_names] for r in rows], dtype=float)
    scores = np.zeros(len(y), dtype=float)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(X[train_idx])
        x_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(**LR_PARAMS)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message="'penalty' was deprecated",
                category=FutureWarning,
            )
            clf.fit(x_train, y[train_idx])
        scores[test_idx[0]] = clf.predict_proba(x_test)[0,1]
    return scores


def bootstrap_auc_ci(y: np.ndarray, scores: np.ndarray, n_boot: int=N_BOOTSTRAP, seed: int=RANDOM_SEED) -> Tuple[float,float,int]:
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y))
    vals = []; rejected = 0
    while len(vals) < n_boot:
        idx = rng.choice(idx_all, size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            rejected += 1; continue
        vals.append(fast_auc(y[idx], scores[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), rejected


def paired_bootstrap_diff(y: np.ndarray, a: np.ndarray, b: np.ndarray, n_boot: int=N_BOOTSTRAP, seed: int=RANDOM_SEED) -> Tuple[float,float,int]:
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y))
    vals = []; rejected = 0
    while len(vals) < n_boot:
        idx = rng.choice(idx_all, size=len(y), replace=True)
        if len(np.unique(y[idx])) < 2:
            rejected += 1; continue
        vals.append(fast_auc(y[idx], a[idx]) - fast_auc(y[idx], b[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), rejected


def aggregate_roi_rows(roi_rows: List[dict]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for r in roi_rows:
        grouped[r['case_id']].append(r)
    result = {}
    for case_id, rows in grouped.items():
        rec = {'case_id':case_id, 'pathology':rows[0]['pathology'], 'ROI_Count':len(rows)}
        for f in FEATURES:
            rec[f] = float(np.mean([float(r[f]) for r in rows]))
        result[case_id] = rec
    return result


def validate_roi_input(roi_rows: List[dict], roi_fields: Sequence[str]) -> None:
    """Validate the de-identified ROI table before any case-level modeling."""
    required_identifiers = ['roi_id', 'case_id', 'pathology']
    missing_identifiers = [field for field in required_identifiers if field not in roi_fields]
    if missing_identifiers:
        raise RuntimeError(f'Missing required ROI identifiers: {missing_identifiers}')
    missing_features = [feature for feature in FEATURES if feature not in roi_fields]
    if missing_features:
        raise RuntimeError(f'Missing required ROI-level features: {missing_features}')
    if len(roi_rows) != 148:
        raise RuntimeError(f'Expected 148 ROI rows, found {len(roi_rows)}.')

    roi_ids = [row['roi_id'].strip() for row in roi_rows]
    if any(not roi_id for roi_id in roi_ids) or len(set(roi_ids)) != len(roi_ids):
        raise RuntimeError('Every ROI must have a non-empty unique identifier.')

    forbidden_model_columns = {
        'roi_id', 'case_id', 'pathology', 'ROI_Count',
        'Image_Height_px', 'Image_Width_px', 'Image_Area_px',
        'CavityRatio', 'SignalAreaRatio',
    }
    overlap = forbidden_model_columns.intersection(FEATURES)
    if overlap:
        raise RuntimeError(f'Forbidden columns were included as model features: {sorted(overlap)}')

    pathology_by_case: Dict[str, set] = defaultdict(set)
    for row_number, row in enumerate(roi_rows, start=2):
        if not row['case_id'].strip() or not row['pathology'].strip():
            raise RuntimeError(f'Missing case_id or pathology at ROI CSV row {row_number}.')
        pathology_by_case[row['case_id']].add(row['pathology'])
        for feature in FEATURES:
            value = float(row[feature])
            if not math.isfinite(value):
                raise RuntimeError(f'Non-finite value for {feature} at ROI CSV row {row_number}.')
    inconsistent = {
        case_id: sorted(pathologies)
        for case_id, pathologies in pathology_by_case.items()
        if len(pathologies) != 1
    }
    if inconsistent:
        raise RuntimeError(f'Inconsistent pathology labels within cases: {inconsistent}')


def case_metadata_rows(case_rows: List[dict]) -> List[dict]:
    output = []
    for row in case_rows:
        record = {
            'case_id': row['case_id'],
            'pathology': row['pathology'],
            'ROI_Count': int(row['ROI_Count']),
        }
        for task_key, cfg in TASKS.items():
            included = row['pathology'] in cfg['include_pathologies']
            record[f'{task_key}_included'] = int(included)
            record[f'{task_key}_label'] = (
                int(row['pathology'] in cfg['positive_pathologies']) if included else ''
            )
        output.append(record)
    return output


def verify_expected_compact_results(performance: List[dict], expected_path: Path) -> None:
    if not expected_path.exists():
        return
    with expected_path.open('r', encoding='utf-8') as handle:
        expected = json.load(handle)
    tolerance = float(expected.get('absolute_tolerance', 1e-12))
    compact = {
        row['task']: row
        for row in performance
        if row['model_name'] == 'Task-specific compact panel'
    }
    for task_key, expected_values in expected['compact_panel_results'].items():
        for field in ['auc', 'ci_lower', 'ci_upper']:
            observed = float(compact[task_key][field])
            target = float(expected_values[field])
            if abs(observed - target) > tolerance:
                raise RuntimeError(
                    f'Expected-result check failed for {task_key} {field}: '
                    f'observed={observed}, expected={target}.'
                )


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduce the reported case-level UV-PAM analysis.')
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    root = args.package_root.resolve()
    data_dir = root/'data'
    config_dir = root/'configuration'
    results_dir = args.output_dir.resolve() if args.output_dir else root/'reproduced_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    reference_case_rows, case_fields = read_csv_numeric(data_dir/'Case_level_features.csv')
    roi_rows, roi_fields = read_csv_numeric(data_dir/'ROI_level_features.csv')
    with (config_dir/'redundancy_group_definitions.json').open('r', encoding='utf-8') as f:
        redundancy = json.load(f)
    with (config_dir/'fixed_panel_definitions.json').open('r', encoding='utf-8') as f:
        fixed = json.load(f)
    with (config_dir/'feature_family_definitions.json').open('r', encoding='utf-8') as f:
        feature_config = json.load(f)

    missing = [f for f in FEATURES if f not in case_fields]
    if missing:
        raise RuntimeError(f'Missing required case-level features: {missing}')
    configured_features = [
        feature
        for family in feature_config['feature_families'].values()
        for feature in family
    ]
    if configured_features != FEATURES:
        raise RuntimeError('Feature-family configuration does not match the archived analysis feature order.')
    validate_roi_input(roi_rows, roi_fields)

    # Recompute case-level features from the ROI-level table and validate the reference.
    aggregated = aggregate_roi_rows(roi_rows)
    if len(aggregated) != 29:
        raise RuntimeError(f'Expected 29 cases after ROI aggregation, found {len(aggregated)}.')
    case_rows = [aggregated[case_id] for case_id in sorted(aggregated)]
    if len({row['case_id'] for row in case_rows}) != 29:
        raise RuntimeError('Case aggregation did not produce one row per case.')

    reference_map = {row['case_id']: row for row in reference_case_rows}
    if len(reference_map) != 29:
        raise RuntimeError('Reference Case_level_features.csv must contain 29 unique cases.')
    aggregation_checks = []
    max_abs_diff = 0.0
    for a in case_rows:
        r = reference_map[a['case_id']]
        if r['pathology'] != a['pathology']:
            raise RuntimeError(f'Pathology mismatch for case {a["case_id"]}.')
        for f in FEATURES:
            diff = abs(float(r[f]) - float(a[f]))
            max_abs_diff = max(max_abs_diff, diff)
        case_max_diff = max(abs(float(r[f])-float(a[f])) for f in FEATURES)
        roi_count_match = int(float(r['ROI_Count'])) == int(a['ROI_Count'])
        aggregation_checks.append({
            'case_id':r['case_id'], 'pathology':r['pathology'],
            'roi_count_reference':int(float(r['ROI_Count'])),
            'roi_count_reproduced':int(a['ROI_Count']),
            'max_absolute_feature_difference':case_max_diff,
            'aggregation_match_within_1e-10':case_max_diff <= 1e-10 and roi_count_match,
        })
    if max_abs_diff > 1e-10 or not all(row['aggregation_match_within_1e-10'] for row in aggregation_checks):
        raise RuntimeError(f'ROI-to-case aggregation validation failed; maximum difference={max_abs_diff}.')
    write_csv(
        results_dir/'Case_level_features.csv',
        case_rows,
        ['case_id', 'pathology', 'ROI_Count', *FEATURES],
    )
    write_csv(results_dir/'case_metadata.csv', case_metadata_rows(case_rows))
    write_csv(results_dir/'case_aggregation_check.csv', aggregation_checks)

    rankings = ranking_rows(case_rows)
    write_csv(results_dir/'taskwise_single_feature_auc_rankings.csv', rankings)
    correlations = correlation_rows(case_rows)
    write_csv(results_dir/'taskwise_spearman_correlations.csv', correlations)

    generated_panels, selection_log = select_panels(rankings, redundancy)
    write_csv(results_dir/'panel_selection_log.csv', selection_log)
    generated_doc = {
        'method_name': fixed['method_name'],
        'panel_size': 4,
        'panels': generated_panels,
        'matches_locked_definitions': generated_panels == fixed['panels'],
    }
    (results_dir/'generated_panel_definitions.json').write_text(
        json.dumps(generated_doc, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    if generated_panels != fixed['panels']:
        raise RuntimeError(f'Generated panels do not match fixed definitions. Generated={generated_panels}, expected={fixed["panels"]}')

    models = {
        'Intensity-distribution baseline': INTENSITY,
        'GLCM texture baseline': GLCM,
        'Signal-area baseline': SIGNAL_AREA,
        'Low-signal morphology baseline': LOW_MORPH,
        'Full-feature baseline': FEATURES,
        'Task-specific compact panel': None,
    }
    oof = []
    performance = []
    roc_data = []
    score_map: Dict[Tuple[str,str], np.ndarray] = {}
    y_map: Dict[str, np.ndarray] = {}
    for task_key, cfg in TASKS.items():
        rows, y = subset_task(case_rows, task_key)
        y_map[task_key] = y
        for model_name, feats in models.items():
            # Final evaluation uses the locked panel definition, not a new search.
            use_feats = fixed['panels'][task_key] if feats is None else feats
            scores = fit_predict_loocv(rows, y, use_feats)
            score_map[(task_key, model_name)] = scores
            auc = float(roc_auc_score(y, scores))
            ci_l, ci_u, rejected = bootstrap_auc_ci(y, scores)
            performance.append({
                'task':task_key, 'task_display':cfg['display'], 'model_name':model_name,
                'n_positive':int(np.sum(y==1)), 'n_negative':int(np.sum(y==0)),
                'auc':auc, 'ci_lower':ci_l, 'ci_upper':ci_u,
                'bootstrap_valid':N_BOOTSTRAP, 'bootstrap_rejected_single_class':rejected,
                'n_features':len(use_feats), 'feature_names':';'.join(use_feats),
            })
            for r, yy, ss in zip(rows, y, scores):
                oof.append({
                    'task':task_key, 'case_id':r['case_id'], 'pathology':r['pathology'],
                    'true_label':int(yy), 'model_name':model_name,
                    'oof_probability':float(ss),
                    'positive_label':cfg['positive_label'], 'negative_label':cfg['negative_label'],
                })
            fpr, tpr, thresholds = roc_curve(y, scores)
            for point_index, (fpr_value, tpr_value, threshold) in enumerate(zip(fpr, tpr, thresholds)):
                roc_data.append({
                    'task':task_key, 'model_name':model_name, 'point_index':point_index,
                    'false_positive_rate':float(fpr_value),
                    'true_positive_rate':float(tpr_value),
                    'threshold':float(threshold),
                })
    write_csv(results_dir/'oof_predictions_reproduced.csv', oof)
    write_csv(results_dir/'performance_summary.csv', performance)
    write_csv(results_dir/'roc_curve_data.csv', roc_data)

    paired = []
    single_family_names = ['Intensity-distribution baseline','GLCM texture baseline','Signal-area baseline','Low-signal morphology baseline']
    for task_key, cfg in TASKS.items():
        task_perf = [r for r in performance if r['task']==task_key]
        compact = next(r for r in task_perf if r['model_name']=='Task-specific compact panel')
        best_single = max((r for r in task_perf if r['model_name'] in single_family_names), key=lambda r:r['auc'])
        full = next(r for r in task_perf if r['model_name']=='Full-feature baseline')
        for comparison_name, ref in [('best single-feature-family baseline', best_single), ('full-feature baseline', full)]:
            y = y_map[task_key]
            a = score_map[(task_key,'Task-specific compact panel')]
            b = score_map[(task_key,ref['model_name'])]
            ci_l, ci_u, rejected = paired_bootstrap_diff(y,a,b)
            paired.append({
                'task':task_key, 'task_display':cfg['display'],
                'comparison':f'compact panel - {comparison_name}',
                'model_1':'Task-specific compact panel', 'model_2':ref['model_name'],
                'auc_model_1':compact['auc'], 'auc_model_2':ref['auc'],
                'auc_difference':compact['auc']-ref['auc'],
                'ci_lower':ci_l, 'ci_upper':ci_u,
                'bootstrap_valid':N_BOOTSTRAP, 'bootstrap_rejected_single_class':rejected,
            })
    write_csv(results_dir/'paired_auc_differences.csv', paired)

    verify_expected_compact_results(
        performance,
        root/'reference_results'/'expected_results.json',
    )
    compact_results = {
        row['task']: {
            'auc':row['auc'], 'ci_lower':row['ci_lower'], 'ci_upper':row['ci_upper']
        }
        for row in performance
        if row['model_name'] == 'Task-specific compact panel'
    }
    run_summary = {
        'validation_passed':True,
        'source_code_basis':'Archived analysis_script.py from UVPAM_caselevel_reproducibility_package_final.zip',
        'roi_count':len(roi_rows),
        'case_count':len(case_rows),
        'candidate_feature_count':len(FEATURES),
        'maximum_roi_to_case_aggregation_difference':max_abs_diff,
        'generated_panels_match_fixed_definitions':generated_panels == fixed['panels'],
        'fixed_panels_used_for_final_evaluation':fixed['panels'],
        'compact_panel_results':compact_results,
    }
    (results_dir/'run_summary.json').write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )

    print('Reproduction completed successfully.')
    print(f'ROI rows validated: {len(roi_rows)}')
    print(f'Cases after aggregation: {len(case_rows)}')
    print(f'Maximum ROI-to-case aggregation difference: {max_abs_diff:.3g}')
    print()
    print('Malignant vs non-malignant:')
    print(f"Compact panel AUC = {compact_results['malignant_vs_non_malignant']['auc']:.3f}")
    print()
    print('Inflammatory vs hyperplastic:')
    print(f"Compact panel AUC = {compact_results['inflammatory_vs_hyperplastic']['auc']:.3f}")
    print()
    print('Tubular adenoma vs non-neoplastic:')
    print(f"Compact panel AUC = {compact_results['tubular_adenoma_vs_non_neoplastic']['auc']:.3f}")
    print()
    print(f'Outputs written to: {results_dir}')

if __name__ == '__main__':
    main()
