#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
roi = pd.read_csv(BASE / "inputs" / "ROI_level_features_used.csv")
params = json.loads((BASE / "outputs" / "modeling_parameters.json").read_text(encoding="utf-8"))

print("=== ROI DIMENSION AUDIT ===")
print(f"ROI count: {len(roi)}")
print(
    f"Height: {roi['Image_Height_px'].min()}-{roi['Image_Height_px'].max()} px; "
    f"median {roi['Image_Height_px'].median():g} px"
)
print(
    f"Width: {roi['Image_Width_px'].min()}-{roi['Image_Width_px'].max()} px; "
    f"median {roi['Image_Width_px'].median():g} px"
)
print(f"All square: {(roi['Image_Height_px'] == roi['Image_Width_px']).all()}")
print(f"Non-square ROI count: {(roi['Image_Height_px'] != roi['Image_Width_px']).sum()}")
print(
    "100x100 ROI count: "
    f"{((roi['Image_Height_px'] == 100) & (roi['Image_Width_px'] == 100)).sum()}"
)
print(
    "Physical range IF native 5 um/pixel was preserved without resizing: "
    f"{roi['Image_Width_px'].min()*5:g}-{roi['Image_Width_px'].max()*5:g} um; "
    f"median {roi['Image_Width_px'].median()*5:g} um"
)
print(
    "WARNING: pixel dimensions do not prove physical field of view if images were resized, "
    "resampled, or exported with a changed scale."
)

print("\n=== BOOTSTRAP SPECIFICATION ===")
print(json.dumps(params["bootstrap_auc_ci"], indent=2))
print(json.dumps(params["paired_bootstrap"], indent=2))
