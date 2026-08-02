
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducible same-dataset exploratory case-level feature-based analysis.

This script:
1) loads case-level UV-PAM features;
2) runs leave-one-case-out CV with fold-wise StandardScaler;
3) compares literature-inspired baseline models and task-specific proposed feature panels;
4) exports out-of-fold prediction scores, AUC summaries, bootstrap 95% CI,
   paired bootstrap AUC-difference summaries, feature-definition tables, and figures.

Important limitation:
The proposed selected-feature panels are fixed here according to the manuscript-defined,
data-informed task-specific panels. This is NOT nested feature selection. The reported AUCs are
same-dataset exploratory point estimates and may be optimistic.
"""
from __future__ import annotations

import json
import platform
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / 'inputs'
OUT_DIR = ROOT / 'output'
FIG_DIR = ROOT / 'figures'
OUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)

CASE_FILE = INPUT_DIR / 'Case_level_features_used.csv'
ROI_FILE = INPUT_DIR / 'ROI_level_features_used.csv'
PARAM_FILE = INPUT_DIR / 'feature_extraction_parameters.json'

RANDOM_SEED = 0
N_BOOTSTRAP = 2000

GRAY_FEATURES = [
    'Mean','StdDev','Median','Min','Max','P10','P25','P75','P90','IQR',
    'Skewness','Kurtosis','GrayEntropy'
]
GLCM_FEATURES = [
    'GLCM_contrast','GLCM_dissimilarity','GLCM_homogeneity','GLCM_ASM',
    'GLCM_energy','GLCM_correlation','GLCM_entropy'
]
EXTRACTED_SIGNAL_RATIO_FEATURES = [
    'LowSignalRatio','HighSignalRatio','SignalAreaRatio','CavityRatio'
]
# CavityRatio duplicates LowSignalRatio exactly, and SignalAreaRatio is its
# complement (1 - LowSignalRatio). Retain all four in the extraction record,
# but use only the two nonredundant signal-ratio variables for modeling.
MODELING_SIGNAL_RATIO_FEATURES = ['LowSignalRatio','HighSignalRatio']
LOW_SIGNAL_MORPHOLOGY_FEATURES = [
    'LowObj_Count','LowObj_Density_per_10kpx','LowObj_MeanArea_px',
    'LowObj_MedianArea_px','LowObj_MeanCircularity','LowObj_MeanSolidity',
    'LowObj_MeanEccentricity','LowObj_MeanAspectRatio','LowObj_TotalAreaRatio_filtered'
]
ALL_EXTRACTED_NON_QC_FEATURES = GRAY_FEATURES + GLCM_FEATURES + EXTRACTED_SIGNAL_RATIO_FEATURES + LOW_SIGNAL_MORPHOLOGY_FEATURES
ALL_NONREDUNDANT_MODELING_FEATURES = GRAY_FEATURES + GLCM_FEATURES + MODELING_SIGNAL_RATIO_FEATURES + LOW_SIGNAL_MORPHOLOGY_FEATURES

TASKS = {
    'malignant_vs_non_malignant': {
        'display': 'Malignant vs non-malignant',
        'include_pathologies': ['adenocarcinoma','tubular_adenoma','inflammatory_polyp','hyperplastic_polyp'],
        'positive_pathologies': ['adenocarcinoma'],
        'positive_label': 'Malignant',
        'negative_label': 'Non-malignant',
        'proposed_features': ['GLCM_correlation','Kurtosis','LowObj_MeanEccentricity','LowObj_MeanSolidity'],
    },
    'inflammatory_vs_hyperplastic': {
        'display': 'Inflammatory vs hyperplastic',
        'include_pathologies': ['inflammatory_polyp','hyperplastic_polyp'],
        'positive_pathologies': ['inflammatory_polyp'],
        'positive_label': 'Inflammatory',
        'negative_label': 'Hyperplastic',
        'proposed_features': ['Skewness','LowObj_MeanCircularity','LowObj_MeanSolidity','LowSignalRatio'],
    },
    'adenoma_vs_non_neoplastic': {
        'display': 'Tubular adenoma vs non-neoplastic',
        'include_pathologies': ['tubular_adenoma','inflammatory_polyp','hyperplastic_polyp'],
        'positive_pathologies': ['tubular_adenoma'],
        'positive_label': 'Adenoma',
        'negative_label': 'Non-neoplastic',
        'proposed_features': ['GLCM_dissimilarity','GLCM_homogeneity','LowObj_Density_per_10kpx','GLCM_contrast'],
    },
}

MODELS = {
    'Intensity-statistics baseline': GRAY_FEATURES,
    'GLCM texture baseline': GLCM_FEATURES,
    'Low-signal morphology baseline': LOW_SIGNAL_MORPHOLOGY_FEATURES,
    'Full-feature baseline': ALL_NONREDUNDANT_MODELING_FEATURES,
    'Proposed method': None,  # per-task selected feature panel
}
MODEL_PLOT_LABELS = {
    'Intensity-statistics baseline': 'Intensity-statistics',
    'GLCM texture baseline': 'GLCM texture',
    'Low-signal morphology baseline': 'Low-signal morphology',
    'Full-feature baseline': 'Full-feature',
    'Proposed method': 'Proposed method',
}

LR_PARAMS = {
    'solver': 'liblinear',
    'C': 1.0,
    'class_weight': 'balanced',
    'max_iter': 1000,
    'random_state': RANDOM_SEED,
}


def load_case_features() -> pd.DataFrame:
    df = pd.read_csv(CASE_FILE, encoding='utf-8-sig')
    if df.empty:
        raise ValueError(f'Case-level feature table is empty: {CASE_FILE}')
    required = ['case_id','pathology'] + ALL_EXTRACTED_NON_QC_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns in case-level table: {missing}')
    return df


def subset_task(df: pd.DataFrame, task_key: str) -> Tuple[pd.DataFrame, np.ndarray]:
    cfg = TASKS[task_key]
    d = df[df['pathology'].isin(cfg['include_pathologies'])].copy().reset_index(drop=True)
    y = d['pathology'].isin(cfg['positive_pathologies']).astype(int).to_numpy()
    return d, y


def fit_predict_loocv(d: pd.DataFrame, y: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, int]:
    X = d[feature_names].astype(float).to_numpy()
    loo = LeaveOneOut()
    y_score = np.zeros_like(y, dtype=float)
    convergence_warning_count = 0

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        scaler = StandardScaler()
        X_train_z = scaler.fit_transform(X_train)
        X_test_z = scaler.transform(X_test)
        clf = LogisticRegression(**LR_PARAMS)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            clf.fit(X_train_z, y_train)
            convergence_warning_count += sum(isinstance(w.message, ConvergenceWarning) for w in caught)
        y_score[test_idx[0]] = clf.predict_proba(X_test_z)[:, 1][0]
    return y_score, convergence_warning_count




def fast_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Fast AUC for binary labels using rank-sum formula."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(y_score, method='average')
    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int=N_BOOTSTRAP, seed: int=RANDOM_SEED) -> Tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    rejected = 0
    idx_all = np.arange(n)
    while len(vals) < n_boot:
        idx = rng.choice(idx_all, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            rejected += 1
            continue
        vals.append(fast_auc_binary(y_true[idx], y_score[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), rejected


def paired_bootstrap_diff(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, n_boot: int=N_BOOTSTRAP, seed: int=RANDOM_SEED) -> Tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []
    rejected = 0
    idx_all = np.arange(n)
    while len(vals) < n_boot:
        idx = rng.choice(idx_all, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            rejected += 1
            continue
        vals.append(fast_auc_binary(y_true[idx], score_a[idx]) - fast_auc_binary(y_true[idx], score_b[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), rejected


def latex_escape(value: object) -> str:
    replacements = {
        '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
        '#': r'\#', '_': r'\_\allowbreak{}', '{': r'\{', '}': r'\}',
        'π': r'$\pi$', 'Σ': r'$\sum$', '²': r'$^2$', '°': r'$^\circ$',
    }
    return ''.join(replacements.get(char, char) for char in str(value))


def dataframe_to_latex(df: pd.DataFrame, path: Path, longtable: bool=False) -> None:
    """Write a dependency-free LaTeX table (avoids pandas/Jinja2 dependency)."""
    env = 'longtable' if longtable else 'tabular'
    if longtable and len(df.columns) == 7:
        widths = [0.09, 0.14, 0.18, 0.12, 0.09, 0.16, 0.14]
        spec = ''.join(r'>{\raggedright\arraybackslash}p{' + f'{w:.2f}' + r'\linewidth}' for w in widths)
    else:
        spec = 'l' * len(df.columns)
    lines = [f'\\begin{{{env}}}{{{spec}}}', r'\hline']
    lines.append(' & '.join(latex_escape(c) for c in df.columns) + r' \\')
    lines.append(r'\hline')
    for row in df.itertuples(index=False, name=None):
        lines.append(' & '.join(latex_escape(v) for v in row) + r' \\')
    lines.extend([r'\hline', f'\\end{{{env}}}', ''])
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_latex_table(summary: pd.DataFrame, path: Path) -> None:
    pivot_rows = []
    model_order = list(MODELS.keys())
    task_order = list(TASKS.keys())
    for model in model_order:
        row = {'Model configuration': model}
        for task in task_order:
            r = summary[(summary['task']==task)&(summary['model_name']==model)].iloc[0]
            row[TASKS[task]['display'] + f"\n(n={int(r['n_positive'])}/{int(r['n_negative'])})"] = f"{r['auc']:.3f} ({r['ci_lower']:.3f}--{r['ci_upper']:.3f})"
        pivot_rows.append(row)
    table = pd.DataFrame(pivot_rows)
    dataframe_to_latex(table, path)


def make_feature_definition_table(low_thr: float, high_thr: float) -> pd.DataFrame:
    rows = []
    def add(group, name, definition, inp, denom='', params='', notes=''):
        rows.append({
            'Feature group': group,
            'Feature name': name,
            'Definition / formula': definition,
            'Input image or mask': inp,
            'Normalization denominator': denom,
            'Key parameters': params,
            'Notes': notes,
        })
    # Gray-level features
    gray_defs = {
        'Mean':'Arithmetic mean of ROI intensity values.',
        'StdDev':'Standard deviation of ROI intensity values.',
        'Median':'Median of ROI intensity values.',
        'Min':'Minimum ROI intensity value.',
        'Max':'Maximum ROI intensity value.',
        'P10':'10th percentile of ROI intensity values.',
        'P25':'25th percentile of ROI intensity values.',
        'P75':'75th percentile of ROI intensity values.',
        'P90':'90th percentile of ROI intensity values.',
        'IQR':'Interquartile range, P75 - P25.',
        'Skewness':'Third standardized moment of ROI intensity distribution.',
        'Kurtosis':'Fourth standardized moment excess/shape descriptor of ROI intensity distribution.',
        'GrayEntropy':'Entropy of the ROI gray-level histogram.',
    }
    for name, desc in gray_defs.items():
        add('Gray-level statistics', name, desc, 'Gaussian-smoothed UV-PAM ROI intensities', '', 'Gaussian sigma = 0.8', '')
    # GLCM features
    glcm_params = '32 gray levels; distances = 1, 2 px; angles = 0°, 45°, 90°, 135°; symmetric=True; normed=True; properties averaged across distance-angle matrices.'
    glcm_defs = {
        'GLCM_contrast':'Haralick/GLCM contrast from normalized gray-level co-occurrence matrices.',
        'GLCM_dissimilarity':'GLCM dissimilarity from normalized gray-level co-occurrence matrices.',
        'GLCM_homogeneity':'GLCM homogeneity from normalized gray-level co-occurrence matrices.',
        'GLCM_ASM':'GLCM angular second moment from normalized gray-level co-occurrence matrices.',
        'GLCM_energy':'Square root of angular second moment as reported by graycoprops.',
        'GLCM_correlation':'GLCM correlation from normalized gray-level co-occurrence matrices.',
        'GLCM_entropy':'Mean entropy, -Σ p log2(p), computed from nonzero normalized GLCM probabilities for each distance-angle matrix and averaged.',
    }
    for name, desc in glcm_defs.items():
        add('GLCM texture', name, desc, 'Quantized Gaussian-smoothed UV-PAM ROI intensities', '', glcm_params, '')
    # Signal ratios
    ratio_params = f'Low/high thresholds from pooled ROI pixel distribution: 15th percentile = {low_thr:g}, 85th percentile = {high_thr:g} in the current dataset.'
    add('Signal-ratio features', 'LowSignalRatio', 'Fraction of ROI pixels with intensity <= low-signal threshold.', 'Low-signal mask', 'ROI pixel count', ratio_params, '')
    add('Signal-ratio features', 'HighSignalRatio', 'Fraction of ROI pixels with intensity >= high-signal threshold.', 'High-signal mask', 'ROI pixel count', ratio_params, '')
    add('Signal-ratio features', 'SignalAreaRatio', 'Fraction of ROI pixels above the low-signal threshold.', 'Signal-bearing mask', 'ROI pixel count', ratio_params, '')
    add('Signal-ratio features', 'CavityRatio', 'Operationally defined as fraction of ROI pixels with intensity <= low-signal threshold.', 'Low-signal mask', 'ROI pixel count', ratio_params, 'Exact duplicate of LowSignalRatio in the extraction code; retained only to document the original output and excluded from the nonredundant full-feature baseline.')
    # Morphology features
    morph_params = 'Low mask = intensity <= low threshold; remove objects smaller than 16 px; connected components labeled with skimage.measure.label; region properties from regionprops.'
    add('Low-signal morphology', 'LowObj_Count', 'Number of filtered low-signal connected components.', 'Filtered low-signal object mask', '', morph_params, '')
    add('Low-signal morphology', 'LowObj_Density_per_10kpx', 'LowObj_Count × 10000 / ROI area in pixels.', 'Filtered low-signal object mask', 'ROI pixel count', morph_params, '')
    add('Low-signal morphology', 'LowObj_MeanArea_px', 'Mean area of filtered low-signal connected components in pixels.', 'Filtered low-signal object mask', '', morph_params, '')
    add('Low-signal morphology', 'LowObj_MedianArea_px', 'Median area of filtered low-signal connected components in pixels.', 'Filtered low-signal object mask', '', morph_params, '')
    add('Low-signal morphology', 'LowObj_MeanCircularity', 'Mean circularity = 4π × area / perimeter² across filtered low-signal objects.', 'Filtered low-signal object mask', '', morph_params, 'Objects with zero perimeter are excluded or assigned zero according to implementation.')
    add('Low-signal morphology', 'LowObj_MeanSolidity', 'Mean solidity of filtered low-signal objects from regionprops.', 'Filtered low-signal object mask', '', morph_params, 'Solidity = object area / convex hull area.')
    add('Low-signal morphology', 'LowObj_MeanEccentricity', 'Mean eccentricity of filtered low-signal objects from regionprops.', 'Filtered low-signal object mask', '', morph_params, '')
    add('Low-signal morphology', 'LowObj_MeanAspectRatio', 'Mean major_axis_length / minor_axis_length across filtered low-signal objects.', 'Filtered low-signal object mask', '', morph_params, 'Objects with zero minor axis are excluded from aspect-ratio calculation.')
    add('Low-signal morphology', 'LowObj_TotalAreaRatio_filtered', 'Total area of filtered low-signal objects / ROI pixel count.', 'Filtered low-signal object mask', 'ROI pixel count', morph_params, '')
    return pd.DataFrame(rows)


def plot_roc(oof: pd.DataFrame) -> None:
    task_order = list(TASKS.keys())
    model_order = list(MODELS.keys())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    roc_rows = []
    for ax, task in zip(axes, task_order):
        for model in model_order:
            s = oof[(oof['task']==task)&(oof['model_name']==model)].copy()
            fpr, tpr, thr = roc_curve(s['y_true'], s['y_score'])
            model_auc = roc_auc_score(s['y_true'], s['y_score'])
            ax.plot(fpr, tpr, lw=1.6, label=f"{MODEL_PLOT_LABELS[model]} ({model_auc:.3f})")
            for f,t,th in zip(fpr,tpr,thr):
                roc_rows.append({'task':task,'task_display':TASKS[task]['display'],'model_name':model,'fpr':f,'tpr':t,'threshold':th,'auc':model_auc})
        ax.plot([0,1],[0,1], ls='--', lw=1.0)
        ax.set_title(TASKS[task]['display'], fontsize=10)
        ax.set_xlabel('1 - Specificity')
        ax.set_ylabel('Sensitivity')
        ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
        ax.legend(fontsize=6, loc='lower right', frameon=False)
    fig.tight_layout()
    for ext in ['png']:
        fig.savefig(FIG_DIR/f'Figure6_ROC_comparison_recomputed.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(roc_rows).to_csv(OUT_DIR/'roc_curve_data.csv', index=False, encoding='utf-8-sig')


def plot_auc_bar(summary: pd.DataFrame) -> None:
    task_order = list(TASKS.keys())
    model_order = list(MODELS.keys())
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = np.arange(len(task_order))
    width = 0.15
    for i, model in enumerate(model_order):
        vals=[]; lo=[]; hi=[]
        for task in task_order:
            r=summary[(summary['task']==task)&(summary['model_name']==model)].iloc[0]
            vals.append(r['auc']); lo.append(r['auc']-r['ci_lower']); hi.append(r['ci_upper']-r['auc'])
        ax.bar(x + (i-2)*width, vals, width=width, label=MODEL_PLOT_LABELS[model])
        ax.errorbar(x + (i-2)*width, vals, yerr=[lo,hi], fmt='none', capsize=2, lw=0.8)
    labels=[TASKS[t]['display']+f"\n(n={int(summary[summary['task']==t]['n_positive'].iloc[0])}/{int(summary[summary['task']==t]['n_negative'].iloc[0])})" for t in task_order]
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Case-level AUC')
    ax.set_ylim(0,1.05)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    ax.set_title('Exploratory same-dataset case-level AUC with bootstrap 95% CI', fontsize=11)
    fig.tight_layout()
    for ext in ['png']:
        fig.savefig(FIG_DIR/f'AUC_barplot_with_bootstrap_CI.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_selected_feature_boxplots(df: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for task, cfg in TASKS.items():
        d,y=subset_task(df, task)
        d=d.copy(); d['y_true']=y
        features=cfg['proposed_features']
        n=len(features)
        fig, axes = plt.subplots(1, n, figsize=(3.0*n, 3.5))
        if n == 1:
            axes=[axes]
        for ax, feat in zip(axes, features):
            neg=d[d['y_true']==0][feat].astype(float).values
            pos=d[d['y_true']==1][feat].astype(float).values
            try:
                p=mannwhitneyu(neg,pos,alternative='two-sided').pvalue
            except Exception:
                p=np.nan
            ax.boxplot([neg,pos], tick_labels=[f"{cfg['negative_label']}\n(n={len(neg)})", f"{cfg['positive_label']}\n(n={len(pos)})"], widths=0.45, showfliers=True)
            # jittered raw points
            rng=np.random.default_rng(0)
            for xi, vals in enumerate([neg,pos], start=1):
                jitter=rng.normal(0,0.04,size=len(vals))
                ax.scatter(np.full(len(vals), xi)+jitter, vals, s=14, alpha=0.65)
            ax.set_title(feat, fontsize=9)
            ax.set_ylabel(feat)
            ax.text(0.5, 0.96, f'p = {p:.3g}' if np.isfinite(p) else 'p = NA', ha='center', va='top', transform=ax.transAxes, fontsize=8)
            for group_name, vals in [(cfg['negative_label'], neg), (cfg['positive_label'], pos)]:
                for v in vals:
                    rows.append({'task':task,'task_display':cfg['display'],'feature':feat,'group':group_name,'value':v})
        fig.suptitle(cfg['display'] + ' selected feature distributions', fontsize=11, y=1.04)
        fig.tight_layout()
        safe=task
        for ext in ['png']:
            fig.savefig(FIG_DIR/f'{safe}_selected_feature_boxplots.{ext}', dpi=220, bbox_inches='tight')
        plt.close(fig)
    return pd.DataFrame(rows)


def main():
    df=load_case_features()
    oof_rows=[]
    conv_warnings=[]
    for task_key, cfg in TASKS.items():
        d,y=subset_task(df, task_key)
        for model_name, feats in MODELS.items():
            feat_names = cfg['proposed_features'] if model_name == 'Proposed method' else feats
            missing=[f for f in feat_names if f not in d.columns]
            if missing:
                raise ValueError(f'Missing features for {task_key} / {model_name}: {missing}')
            y_score, n_warn=fit_predict_loocv(d,y,feat_names)
            conv_warnings.append({'task':task_key,'model_name':model_name,'convergence_warning_count':n_warn})
            for case_id, pathology, true, score in zip(d['case_id'], d['pathology'], y, y_score):
                oof_rows.append({
                    'task':task_key,'task_display':cfg['display'],'model_name':model_name,
                    'case_id':case_id,'pathology':pathology,'positive_label':cfg['positive_label'],
                    'negative_label':cfg['negative_label'],'y_true':int(true),'y_score':float(score),
                    'feature_names': ';'.join(feat_names),
                    'n_features': len(feat_names),
                })
    oof=pd.DataFrame(oof_rows)
    oof.to_csv(OUT_DIR/'oof_prediction_scores.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(conv_warnings).to_csv(OUT_DIR/'convergence_warnings.csv', index=False, encoding='utf-8-sig')

    # AUC summaries with CI
    summary_rows=[]
    for (task, model), s in oof.groupby(['task','model_name'], sort=False):
        y=s['y_true'].to_numpy()
        score=s['y_score'].to_numpy()
        auc_val=float(roc_auc_score(y,score))
        ci_l, ci_u, rejected=bootstrap_auc_ci(y,score)
        summary_rows.append({
            'task':task,'task_display':TASKS[task]['display'],'model_name':model,
            'n_positive':int(np.sum(y==1)),'n_negative':int(np.sum(y==0)),
            'auc':auc_val,'ci_lower':ci_l,'ci_upper':ci_u,
            'bootstrap_n_valid':N_BOOTSTRAP,'bootstrap_rejected_single_class':rejected,
        })
    summary=pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR/'caselevel_auc_summary_with_ci.csv', index=False, encoding='utf-8-sig')
    summary[['task','task_display','model_name','n_positive','n_negative','auc']].to_csv(OUT_DIR/'caselevel_auc_summary.csv', index=False, encoding='utf-8-sig')
    write_latex_table(summary, OUT_DIR/'Table2_auc_with_ci.tex')

    # Paired comparison vs strongest baseline per task
    paired_rows=[]
    for task in TASKS.keys():
        proposed=oof[(oof['task']==task)&(oof['model_name']=='Proposed method')].sort_values('case_id')
        baseline_summary=summary[(summary['task']==task)&(summary['model_name']!='Proposed method')].sort_values('auc', ascending=False)
        best_model=baseline_summary.iloc[0]['model_name']
        bs=oof[(oof['task']==task)&(oof['model_name']==best_model)].sort_values('case_id')
        # sanity same case order
        assert list(proposed['case_id']) == list(bs['case_id'])
        y=proposed['y_true'].to_numpy(); score_proposed=proposed['y_score'].to_numpy(); score_bs=bs['y_score'].to_numpy()
        auc_proposed=float(roc_auc_score(y,score_proposed)); auc_bs=float(roc_auc_score(y,score_bs))
        ci_l, ci_u, rejected=paired_bootstrap_diff(y,score_proposed,score_bs)
        paired_rows.append({
            'task':task,'task_display':TASKS[task]['display'],'best_baseline_name':best_model,
            'auc_proposed':auc_proposed,'auc_best_baseline':auc_bs,'auc_diff':auc_proposed-auc_bs,
            'diff_ci_lower':ci_l,'diff_ci_upper':ci_u,
            'bootstrap_n_valid':N_BOOTSTRAP,'bootstrap_rejected_single_class':rejected,
            'statistical_superiority_established': bool(ci_l>0) if np.isfinite(ci_l) else False,
        })
    paired=pd.DataFrame(paired_rows)
    paired.to_csv(OUT_DIR/'paired_auc_difference_proposed_vs_best_baseline.csv', index=False, encoding='utf-8-sig')

    # feature definitions
    feature_params={}
    if PARAM_FILE.exists():
        with open(PARAM_FILE,'r',encoding='utf-8') as f:
            feature_params=json.load(f)
    low_thr=float(feature_params['low_signal_threshold_value'])
    high_thr=float(feature_params['high_signal_threshold_value'])
    fdef=make_feature_definition_table(low_thr, high_thr)
    fdef.to_csv(OUT_DIR/'Supplementary_Table_feature_definitions.csv', index=False, encoding='utf-8-sig')
    dataframe_to_latex(fdef, OUT_DIR/'Supplementary_Table_feature_definitions.tex', longtable=True)

    # plot data and figures
    plot_roc(oof)
    plot_auc_bar(summary)
    # Selected-feature boxplot data and figures are generated by scripts/make_selected_feature_boxplots.py.
    box_rows = []
    for task, cfg in TASKS.items():
        d, y = subset_task(df, task)
        d = d.copy(); d['y_true'] = y
        for feat in cfg['proposed_features']:
            for _, row in d.iterrows():
                box_rows.append({'task': task, 'task_display': cfg['display'], 'feature': feat, 'case_id': row['case_id'], 'pathology': row['pathology'], 'group': cfg['positive_label'] if row['y_true'] == 1 else cfg['negative_label'], 'value': float(row[feat])})
    pd.DataFrame(box_rows).to_csv(OUT_DIR/'selected_feature_boxplot_data.csv', index=False, encoding='utf-8-sig')

    # model and data parameters
    meta={
        'analysis_date_note':'Generated by the supplied reproducible reanalysis script.',
        'input_case_file': str(CASE_FILE.name),
        'input_roi_file': str(ROI_FILE.name),
        'n_cases': int(df.shape[0]),
        'case_counts_by_pathology': df['pathology'].value_counts().to_dict(),
        'feature_groups': {
            'gray_level_statistics': GRAY_FEATURES,
            'glcm_texture': GLCM_FEATURES,
            'signal_ratio_extracted': EXTRACTED_SIGNAL_RATIO_FEATURES,
            'signal_ratio_used_for_modeling': MODELING_SIGNAL_RATIO_FEATURES,
            'low_signal_morphology': LOW_SIGNAL_MORPHOLOGY_FEATURES,
            'all_extracted_non_qc_features': ALL_EXTRACTED_NON_QC_FEATURES,
            'all_nonredundant_modeling_features': ALL_NONREDUNDANT_MODELING_FEATURES,
        },
        'tasks': TASKS,
        'models': {k: (v if v is not None else 'task-specific proposed feature panel') for k,v in MODELS.items()},
        'logistic_regression': LR_PARAMS,
        'standardization':'StandardScaler fit on LOOCV training cases only and applied to held-out case.',
        'validation':'Leave-one-case-out cross-validation; case is the unit of analysis.',
        'bootstrap_auc_ci': {'unit':'case','n_valid_resamples':N_BOOTSTRAP,'random_seed':RANDOM_SEED,'ci_type':'percentile (2.5th and 97.5th percentiles)','single_class_resamples':'rejected and redrawn until 2,000 valid resamples are obtained'},
        'paired_bootstrap': {'unit':'case','n_valid_resamples':N_BOOTSTRAP,'random_seed':RANDOM_SEED,'same_case_indices_for_both_models':True,'single_class_resamples':'rejected and redrawn until 2,000 valid resamples are obtained'},
        'software_versions': {
            'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
            'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,'matplotlib':matplotlib.__version__,
        },
        'feature_extraction_parameters_uploaded': feature_params,
        'important_limitations': [
            'The task-specific proposed feature panels are fixed according to manuscript-defined data-informed panels and are not selected within each LOOCV fold.',
            'Low/high thresholds were previously derived from pooled analyzed ROI pixels and are not re-estimated within each LOOCV fold in this reanalysis.',
            'CavityRatio is operationally equivalent to LowSignalRatio and SignalAreaRatio is its exact complement; these redundant columns are documented but excluded from the 31-feature full-feature baseline.',
            'AUCs are exploratory same-dataset estimates, not externally validated diagnostic performance.'
        ],
    }
    with open(OUT_DIR/'modeling_parameters.json','w',encoding='utf-8') as f:
        json.dump(meta,f,ensure_ascii=False,indent=2)

    # short report markdown
    report_lines=[]
    report_lines.append('# Reproducible case-level feature-based reanalysis report\n')
    report_lines.append('## Inputs\n')
    report_lines.append(f'- Case-level file: `{CASE_FILE.name}` ({df.shape[0]} cases)\n')
    report_lines.append(f'- ROI-level file: `{ROI_FILE.name}`\n')
    report_lines.append('## Key modeling choices\n')
    report_lines.append('- Logistic regression: scikit-learn `LogisticRegression(solver="liblinear", C=1.0, class_weight="balanced", max_iter=1000, random_state=0)`.\n')
    report_lines.append('- Standardization: `StandardScaler` fit only on each LOOCV training fold.\n')
    report_lines.append('- AUC CI: 2,000 valid case-level bootstrap resamples; single-class draws were rejected and redrawn; 95% percentile intervals used the 2.5th and 97.5th percentiles.\n')
    report_lines.append('- Paired AUC comparisons used the same resampled case indices for both models.\n')
    report_lines.append('- Proposed panels: fixed manuscript-defined task-specific selected features; not nested feature selection.\n')
    report_lines.append('- Full-feature baseline: 31 nonredundant descriptors; the exact duplicate CavityRatio and complementary SignalAreaRatio columns were documented but excluded.\n')
    report_lines.append('## Recomputed AUC with bootstrap 95% CI\n\n')
    report_lines.append('```csv\n' + summary.to_csv(index=False) + '```')
    report_lines.append('\n\n## Paired bootstrap difference versus strongest baseline\n\n')
    report_lines.append('```csv\n' + paired.to_csv(index=False) + '```')
    report_lines.append('\n\n## Important note\n')
    report_lines.append('These are exploratory same-dataset estimates. The task-specific panels and pooled intensity thresholds were derived from the analyzed dataset rather than within each LOOCV training fold; the results may therefore be optimistic and are not external-validation estimates.\n')
    with open(ROOT/'README_reanalysis.md','w',encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print('Completed reanalysis package.')
    print(summary)
    print(paired)

if __name__ == '__main__':
    main()
