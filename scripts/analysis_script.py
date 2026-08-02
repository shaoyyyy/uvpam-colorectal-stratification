#!/usr/bin/env python3
"""Reproduce the case-level UV-PAM feature-panel analysis.

Outputs:
- taskwise_single_feature_auc_rankings.csv
- taskwise_spearman_correlations.csv
- panel_selection_log.csv
- generated_panel_definitions.json
- oof_predictions.csv
- performance_summary.csv
- paired_auc_differences.csv
- case_aggregation_check.csv

Important: feature ranking and panel construction use the complete cohort and are
not nested within LOOCV, matching the exploratory method reported in the manuscript.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import mannwhitneyu, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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
    },
    'inflammatory_vs_hyperplastic': {
        'display': 'Inflammatory vs hyperplastic',
        'include_pathologies': ['inflammatory_polyp','hyperplastic_polyp'],
        'positive_pathologies': ['inflammatory_polyp'],
        'positive_label': 'Inflammatory',
        'negative_label': 'Hyperplastic',
    },
    'adenoma_vs_non_neoplastic': {
        'display': 'Tubular adenoma vs non-neoplastic',
        'include_pathologies': ['tubular_adenoma','inflammatory_polyp','hyperplastic_polyp'],
        'positive_pathologies': ['tubular_adenoma'],
        'positive_label': 'Tubular adenoma',
        'negative_label': 'Non-neoplastic',
    },
}

LR_PARAMS = dict(solver='liblinear', C=1.0, class_weight='balanced', max_iter=1000, random_state=0)
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
                'raw_single_feature_auc': raw_auc,
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
            rec['auc_rank'] = rank
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.package_root.resolve()
    data_dir = root/'data'; config_dir = root/'config'; results_dir = root/'results'
    results_dir.mkdir(parents=True, exist_ok=True)

    case_rows, case_fields = read_csv_numeric(data_dir/'Case_level_features.csv')
    roi_rows, _ = read_csv_numeric(data_dir/'ROI_level_features.csv')
    with (config_dir/'redundancy_group_definitions.json').open('r', encoding='utf-8') as f:
        redundancy = json.load(f)
    with (config_dir/'fixed_panel_definitions.json').open('r', encoding='utf-8') as f:
        fixed = json.load(f)

    missing = [f for f in FEATURES if f not in case_fields]
    if missing:
        raise RuntimeError(f'Missing required case-level features: {missing}')

    # Verify ROI-to-case aggregation.
    aggregated = aggregate_roi_rows(roi_rows)
    aggregation_checks = []
    max_abs_diff = 0.0
    for r in case_rows:
        a = aggregated[r['case_id']]
        for f in FEATURES:
            diff = abs(float(r[f]) - float(a[f]))
            max_abs_diff = max(max_abs_diff, diff)
        aggregation_checks.append({
            'case_id':r['case_id'], 'pathology':r['pathology'],
            'roi_count_file':int(float(r['ROI_Count'])), 'roi_count_recomputed':int(a['ROI_Count']),
            'max_absolute_feature_difference':max(abs(float(r[f])-float(a[f])) for f in FEATURES),
            'aggregation_match_within_1e-10':max(abs(float(r[f])-float(a[f])) for f in FEATURES) <= 1e-10,
        })
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
    (results_dir/'generated_panel_definitions.json').write_text(json.dumps(generated_doc, ensure_ascii=False, indent=2), encoding='utf-8')
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
    score_map: Dict[Tuple[str,str], np.ndarray] = {}
    y_map: Dict[str, np.ndarray] = {}
    for task_key, cfg in TASKS.items():
        rows, y = subset_task(case_rows, task_key)
        y_map[task_key] = y
        for model_name, feats in models.items():
            use_feats = generated_panels[task_key] if feats is None else feats
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
                    'task':task_key, 'task_display':cfg['display'], 'model_name':model_name,
                    'case_id':r['case_id'], 'pathology':r['pathology'], 'y_true':int(yy),
                    'positive_label':cfg['positive_label'], 'negative_label':cfg['negative_label'],
                    'y_score':float(ss), 'n_features':len(use_feats), 'feature_names':';'.join(use_feats),
                })
    write_csv(results_dir/'oof_predictions.csv', oof)
    write_csv(results_dir/'performance_summary.csv', performance)

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
                'comparison':comparison_name, 'reference_model':ref['model_name'],
                'auc_compact_panel':compact['auc'], 'auc_reference':ref['auc'],
                'auc_difference_unrounded':compact['auc']-ref['auc'],
                'difference_ci_lower':ci_l, 'difference_ci_upper':ci_u,
                'bootstrap_valid':N_BOOTSTRAP, 'bootstrap_rejected_single_class':rejected,
                'statistical_superiority_established':bool(ci_l>0),
            })
    write_csv(results_dir/'paired_auc_differences.csv', paired)

    print('Reanalysis complete.')
    print(f'Package root: {root}')
    print(f'Max ROI-to-case aggregation difference: {max_abs_diff:.3g}')
    print('Generated panels match fixed definitions:', generated_panels == fixed['panels'])
    for row in performance:
        if row['model_name']=='Task-specific compact panel':
            print(f"{row['task_display']}: AUC={row['auc']:.6f}, 95% CI={row['ci_lower']:.6f}-{row['ci_upper']:.6f}")

if __name__ == '__main__':
    main()
