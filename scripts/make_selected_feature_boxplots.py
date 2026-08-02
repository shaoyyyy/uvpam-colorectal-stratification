#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'inputs'/'Case_level_features_used.csv'
OUT=ROOT/'output'
FIG=ROOT/'figures'
OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)

df=pd.read_csv(INPUT, encoding='utf-8-sig')
TASKS={
 'malignant_vs_non_malignant':{
  'display':'Malignant vs non-malignant','include':['adenocarcinoma','tubular_adenoma','inflammatory_polyp','hyperplastic_polyp'],'positive':['adenocarcinoma'],'positive_label':'Malignant','negative_label':'Non-malignant','features':['GLCM_correlation','Kurtosis','LowObj_MeanEccentricity','LowObj_MeanSolidity']},
 'inflammatory_vs_hyperplastic':{
  'display':'Inflammatory vs hyperplastic','include':['inflammatory_polyp','hyperplastic_polyp'],'positive':['inflammatory_polyp'],'positive_label':'Inflammatory','negative_label':'Hyperplastic','features':['Skewness','LowObj_MeanCircularity','LowObj_MeanSolidity','LowSignalRatio']},
 'adenoma_vs_non_neoplastic':{
  'display':'Tubular adenoma vs non-neoplastic','include':['tubular_adenoma','inflammatory_polyp','hyperplastic_polyp'],'positive':['tubular_adenoma'],'positive_label':'Adenoma','negative_label':'Non-neoplastic','features':['GLCM_dissimilarity','GLCM_homogeneity','LowObj_Density_per_10kpx','GLCM_contrast']},
}
rows=[]
for task,cfg in TASKS.items():
    d=df[df['pathology'].isin(cfg['include'])].copy()
    d['y_true']=d['pathology'].isin(cfg['positive']).astype(int)
    fig, axes=plt.subplots(1,len(cfg['features']),figsize=(2.7*len(cfg['features']),3.2))
    if len(cfg['features'])==1: axes=[axes]
    rng=np.random.default_rng(0)
    for ax,feat in zip(axes,cfg['features']):
        neg=d[d['y_true']==0][feat].astype(float).to_numpy()
        pos=d[d['y_true']==1][feat].astype(float).to_numpy()
        p=mannwhitneyu(neg,pos,alternative='two-sided').pvalue
        ax.boxplot([neg,pos], tick_labels=[f"{cfg['negative_label']}\n(n={len(neg)})", f"{cfg['positive_label']}\n(n={len(pos)})"], widths=0.45, showfliers=False)
        for xi, vals in enumerate([neg,pos],1):
            ax.scatter(np.full(len(vals),xi)+rng.normal(0,0.04,len(vals)), vals, s=13, alpha=0.75)
        ax.set_title(feat,fontsize=8)
        ax.tick_params(axis='x',labelsize=7)
        ax.tick_params(axis='y',labelsize=7)
        ax.text(0.5,0.97,f'p = {p:.3g}',transform=ax.transAxes,ha='center',va='top',fontsize=7)
        for _,r in d.iterrows():
            rows.append({'task':task,'task_display':cfg['display'],'feature':feat,'case_id':r['case_id'],'pathology':r['pathology'],'group':cfg['positive_label'] if r['y_true']==1 else cfg['negative_label'],'value':float(r[feat])})
    fig.suptitle(cfg['display'], fontsize=10, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG/f'{task}_selected_feature_boxplots.png', dpi=220, bbox_inches='tight')
    plt.close(fig)
pd.DataFrame(rows).to_csv(OUT/'selected_feature_boxplot_data.csv', index=False, encoding='utf-8-sig')
print('Saved selected feature boxplots and data.')
