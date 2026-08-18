from __future__ import annotations

import argparse
import csv
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PACKAGE = PACKAGE_ROOT

LR_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "solver": "liblinear",
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": 0,
}

warnings.filterwarnings(
    "ignore",
    message="'penalty' was deprecated in version 1.8",
    category=FutureWarning,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_configuration(package: Path):
    with (package / "configuration/feature_family_definitions.json").open(
        "r", encoding="utf-8"
    ) as fh:
        family_cfg = json.load(fh)
    with (package / "configuration/task_definitions.json").open(
        "r", encoding="utf-8"
    ) as fh:
        task_cfg = json.load(fh)["tasks"]
    features = []
    for values in family_cfg["feature_families"].values():
        features.extend(values)
    if len(features) != 31 or len(set(features)) != 31:
        raise ValueError("Expected 31 unique candidate features.")
    return features, task_cfg


def load_case_data(package: Path, features: list[str]):
    rows = read_csv(package / "data/Case_level_features.csv")
    case_ids = np.asarray([r["case_id"] for r in rows], dtype=object)
    pathologies = np.asarray([r["pathology"] for r in rows], dtype=object)
    x = np.asarray(
        [[float(r[f]) for f in features] for r in rows], dtype=float
    )
    if not np.isfinite(x).all():
        raise ValueError("Non-finite feature value.")
    return case_ids, pathologies, x


def task_subset(pathologies, x, task):
    included = set(task["included_pathologies"])
    positive = set(task["positive_pathologies"])
    mask = np.asarray([p in included for p in pathologies])
    y = np.asarray([int(p in positive) for p in pathologies[mask]], dtype=int)
    return x[mask], y, mask


def orientation_auc(y: np.ndarray, values: np.ndarray) -> float:
    auc = roc_auc_score(y, values)
    return float(max(auc, 1.0 - auc))


def rank_features(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
) -> list[tuple[int, str, float]]:
    ranked = []
    for idx, name in enumerate(feature_names):
        ranked.append((idx, name, orientation_auc(y_train, x_train[:, idx])))
    # Stable deterministic order: descending AUC, then feature name.
    ranked.sort(key=lambda item: (-item[2], item[1]))
    return ranked


def spearman_prune(
    x_train: np.ndarray,
    ranked: list[tuple[int, str, float]],
    max_candidates: int = 6,
    rho_threshold: float = 0.90,
) -> list[tuple[int, str, float]]:
    kept: list[tuple[int, str, float]] = []
    for item in ranked:
        idx = item[0]
        redundant = False
        for kept_item in kept:
            rho = spearmanr(x_train[:, idx], x_train[:, kept_item[0]]).statistic
            if np.isfinite(rho) and abs(float(rho)) >= rho_threshold:
                redundant = True
                break
        if not redundant:
            kept.append(item)
        if len(kept) == max_candidates:
            break
    if not kept:
        raise RuntimeError("Spearman pruning removed every feature.")
    return kept


def loocv_scores(x: np.ndarray, y: np.ndarray, feature_indices: list[int]):
    scores = np.empty(len(y), dtype=float)
    for train_idx, test_idx in LeaveOneOut().split(x):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx][:, feature_indices])
        x_test = scaler.transform(x[test_idx][:, feature_indices])
        clf = LogisticRegression(**LR_PARAMS)
        clf.fit(x_train, y[train_idx])
        scores[test_idx[0]] = clf.predict_proba(x_test)[0, 1]
    return scores


def forward_select(
    x_train: np.ndarray,
    y_train: np.ndarray,
    candidates: list[tuple[int, str, float]],
    max_features: int = 4,
    tolerance: float = 1e-12,
):
    remaining = list(candidates)
    selected: list[tuple[int, str, float]] = []
    best_auc = -np.inf
    steps = []
    while remaining and len(selected) < max_features:
        evaluations = []
        for candidate in remaining:
            indices = [item[0] for item in selected] + [candidate[0]]
            scores = loocv_scores(x_train, y_train, indices)
            auc = roc_auc_score(y_train, scores)
            evaluations.append((float(auc), candidate))
        # Highest inner-LOOCV AUC; ties are resolved by feature name.
        step_auc = max(item[0] for item in evaluations)
        tied = [
            item[1]
            for item in evaluations
            if abs(item[0] - step_auc) <= tolerance
        ]
        chosen = sorted(tied, key=lambda item: item[1])[0]
        if selected and step_auc < best_auc - tolerance:
            steps.append(
                {
                    "step": len(selected) + 1,
                    "candidate": chosen[1],
                    "inner_auc": step_auc,
                    "decision": "stopped: best available addition decreased inner AUC",
                }
            )
            break
        selected.append(chosen)
        remaining = [item for item in remaining if item[0] != chosen[0]]
        best_auc = step_auc
        steps.append(
            {
                "step": len(selected),
                "candidate": chosen[1],
                "inner_auc": step_auc,
                "decision": "retained",
            }
        )
    return selected, steps


def nested_outer_predictions(
    x: np.ndarray,
    y: np.ndarray,
    case_ids: np.ndarray,
    feature_names: list[str],
):
    scores = np.empty(len(y), dtype=float)
    fold_rows = []
    for outer_fold, (train_idx, test_idx) in enumerate(
        LeaveOneOut().split(x), start=1
    ):
        x_train, y_train = x[train_idx], y[train_idx]
        ranked = rank_features(x_train, y_train, feature_names)
        candidates = spearman_prune(x_train, ranked)
        selected, steps = forward_select(x_train, y_train, candidates)
        selected_indices = [item[0] for item in selected]
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train[:, selected_indices])
        x_test_scaled = scaler.transform(x[test_idx][:, selected_indices])
        clf = LogisticRegression(**LR_PARAMS)
        clf.fit(x_train_scaled, y_train)
        probability = float(clf.predict_proba(x_test_scaled)[0, 1])
        scores[test_idx[0]] = probability
        fold_rows.append(
            {
                "outer_fold": outer_fold,
                "held_out_case": case_ids[test_idx[0]],
                "true_label": int(y[test_idx[0]]),
                "oof_probability": probability,
                "candidate_features_after_spearman_pruning": ";".join(
                    item[1] for item in candidates
                ),
                "selected_feature_count": len(selected),
                "selected_features": ";".join(item[1] for item in selected),
                "final_inner_loocv_auc": steps[-1]["inner_auc"]
                if steps and steps[-1]["decision"] == "retained"
                else steps[-2]["inner_auc"],
                "selection_steps": json.dumps(steps, separators=(",", ":")),
            }
        )
    return scores, fold_rows


def bootstrap_auc_ci(
    y: np.ndarray,
    scores: np.ndarray,
    n_valid: int = 2000,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    aucs = []
    rejected = 0
    while len(aucs) < n_valid:
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2:
            rejected += 1
            continue
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    lower, upper = np.percentile(np.asarray(aucs), [2.5, 97.5])
    return float(lower), float(upper), rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated_results",
    )
    args = parser.parse_args()
    features, tasks = load_configuration(args.package)
    case_ids, pathologies, x_all = load_case_data(args.package, features)
    summary = []
    all_folds = []
    stability = []
    for task_name, task in tasks.items():
        x, y, mask = task_subset(pathologies, x_all, task)
        ids = case_ids[mask]
        scores, folds = nested_outer_predictions(x, y, ids, features)
        auc = float(roc_auc_score(y, scores))
        lower, upper, rejected = bootstrap_auc_ci(y, scores)
        summary.append(
            {
                "task": task_name,
                "task_display": task["display_name"],
                "n_positive": int(y.sum()),
                "n_negative": int((1 - y).sum()),
                "nested_loocv_auc": auc,
                "ci_lower": lower,
                "ci_upper": upper,
                "bootstrap_valid": 2000,
                "bootstrap_rejected_single_class": rejected,
            }
        )
        for row in folds:
            row = {"task": task_name, "task_display": task["display_name"], **row}
            all_folds.append(row)
        counts = Counter(
            feature
            for row in folds
            for feature in row["selected_features"].split(";")
            if feature
        )
        for feature, count in sorted(
            counts.items(), key=lambda item: (-item[1], features.index(item[0]))
        ):
            stability.append(
                {
                    "task": task_name,
                    "task_display": task["display_name"],
                    "feature": feature,
                    "outer_folds_selected": count,
                    "outer_fold_count": len(folds),
                    "selection_frequency": count / len(folds),
                }
            )
        print(
            f"{task['display_name']}: AUC={auc:.6f}, "
            f"95% CI={lower:.6f}-{upper:.6f}"
        )
    write_csv(args.output_dir / "nested_loocv_summary.csv", summary)
    write_csv(args.output_dir / "nested_loocv_fold_predictions.csv", all_folds)
    write_csv(args.output_dir / "nested_feature_selection_stability.csv", stability)


if __name__ == "__main__":
    main()
