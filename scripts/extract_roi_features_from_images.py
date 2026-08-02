from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import skew, kurtosis
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix
    from skimage.feature import greycoprops as graycoprops


# =========================
# 基础路径
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
IMG_DIR = BASE_DIR / "roi_images"
META_PATH = BASE_DIR / "metadata.csv"
OUT_DIR = BASE_DIR / "output"
OUT_DIR.mkdir(exist_ok=True)


# =========================
# 参数冻结区
# 后面论文 Methods 也按这里写
# =========================
PARAMS = {
    "gaussian_sigma": 0.8,
    "glcm_levels": 32,
    "glcm_distances": [1, 2],
    "glcm_angles_degree": [0, 45, 90, 135],
    "low_signal_percentile": 15,
    "high_signal_percentile": 85,
    "min_object_size_px": 16,
}


# =========================
# 读取图像：稳定处理 RGB / 灰度 / 透明 PNG
# =========================
def read_image_as_gray_uint8(path: Path) -> np.ndarray:
    """
    稳定读取图像：
    - 灰度图：直接读取
    - RGB图：转灰度
    - RGBA透明图：透明区域强制变黑
    """
    img = Image.open(path)

    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        rgba = img.convert("RGBA")
        arr = np.array(rgba)

        rgb = arr[:, :, :3].astype(float)
        alpha = arr[:, :, 3]

        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        gray[alpha == 0] = 0

        gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray

    gray = img.convert("L")
    arr = np.array(gray)

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


# =========================
# 统一预处理
# =========================
def preprocess_image(arr: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """
    统一轻微高斯滤波。
    不做每张图单独归一化，不做自动增强。
    """
    if sigma is None or sigma <= 0:
        return arr.astype(np.uint8)

    arr_f = gaussian_filter(arr.astype(float), sigma=sigma)
    arr_f = np.clip(arr_f, 0, 255).astype(np.uint8)
    return arr_f


# =========================
# 灰度统计特征
# =========================
def extract_gray_features(arr: np.ndarray) -> dict:
    pixels = arr.astype(float).ravel()

    hist = np.bincount(arr.ravel(), minlength=256).astype(float)
    prob = hist / hist.sum()
    prob_nonzero = prob[prob > 0]

    return {
        "Mean": float(np.mean(pixels)),
        "StdDev": float(np.std(pixels, ddof=1)),
        "Median": float(np.median(pixels)),
        "Min": float(np.min(pixels)),
        "Max": float(np.max(pixels)),
        "P10": float(np.percentile(pixels, 10)),
        "P25": float(np.percentile(pixels, 25)),
        "P75": float(np.percentile(pixels, 75)),
        "P90": float(np.percentile(pixels, 90)),
        "IQR": float(np.percentile(pixels, 75) - np.percentile(pixels, 25)),
        "Skewness": float(skew(pixels)),
        "Kurtosis": float(kurtosis(pixels)),
        "GrayEntropy": float(-np.sum(prob_nonzero * np.log2(prob_nonzero))),
    }


# =========================
# GLCM 纹理特征
# =========================
def quantize_for_glcm(arr: np.ndarray, levels: int = 32) -> np.ndarray:
    """
    将 0-255 灰度压缩到 0-(levels-1)。
    这样比 256 levels 更稳定、抗噪声更好。
    """
    arr_q = np.floor(arr.astype(float) / 256 * levels)
    arr_q = np.clip(arr_q, 0, levels - 1).astype(np.uint8)
    return arr_q


def extract_glcm_features(arr: np.ndarray, levels: int, distances: list, angles_degree: list) -> dict:
    arr_q = quantize_for_glcm(arr, levels=levels)
    angles = [np.deg2rad(a) for a in angles_degree]

    glcm = graycomatrix(
        arr_q,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    features = {}
    props = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]

    for prop in props:
        vals = graycoprops(glcm, prop)
        features[f"GLCM_{prop}"] = float(np.nanmean(vals))

    # 每个 distance-angle 单独算 entropy，再取平均
    entropies = []
    for i in range(len(distances)):
        for j in range(len(angles)):
            p = glcm[:, :, i, j].astype(float)
            p_nonzero = p[p > 0]
            ent = -np.sum(p_nonzero * np.log2(p_nonzero))
            entropies.append(ent)

    features["GLCM_entropy"] = float(np.mean(entropies))

    return features


# =========================
# 低信号/高信号比例特征
# =========================
def extract_ratio_features(arr: np.ndarray, low_thr: float, high_thr: float) -> dict:
    total = arr.size

    low_mask = arr <= low_thr
    high_mask = arr >= high_thr
    signal_mask = arr > low_thr

    return {
        "LowSignalRatio": float(np.sum(low_mask) / total),
        "HighSignalRatio": float(np.sum(high_mask) / total),
        "SignalAreaRatio": float(np.sum(signal_mask) / total),
        "CavityRatio": float(np.sum(low_mask) / total),
    }


# =========================
# 低信号区域形态特征
# 用于描述腔隙样区域/低信号结构
# =========================
def extract_low_signal_morphology(arr: np.ndarray, low_thr: float, min_object_size_px: int = 16) -> dict:
    low_mask = arr <= low_thr
    low_mask = remove_small_objects(low_mask.astype(bool), min_size=min_object_size_px)

    lbl = label(low_mask)
    regions = regionprops(lbl)

    if len(regions) == 0:
        return {
            "LowObj_Count": 0,
            "LowObj_Density_per_10kpx": 0.0,
            "LowObj_MeanArea_px": 0.0,
            "LowObj_MedianArea_px": 0.0,
            "LowObj_MeanCircularity": np.nan,
            "LowObj_MeanSolidity": np.nan,
            "LowObj_MeanEccentricity": np.nan,
            "LowObj_MeanAspectRatio": np.nan,
            "LowObj_TotalAreaRatio_filtered": 0.0,
        }

    areas = np.array([r.area for r in regions], dtype=float)
    solidities = np.array([r.solidity for r in regions], dtype=float)
    eccentricities = np.array([r.eccentricity for r in regions], dtype=float)

    circularities = []
    aspect_ratios = []

    for r in regions:
        if r.perimeter > 0:
            circ = 4 * np.pi * r.area / (r.perimeter ** 2)
        else:
            circ = np.nan
        circularities.append(circ)

        minor = r.minor_axis_length
        major = r.major_axis_length
        if minor > 0:
            aspect_ratios.append(major / minor)
        else:
            aspect_ratios.append(np.nan)

    return {
        "LowObj_Count": int(len(regions)),
        "LowObj_Density_per_10kpx": float(len(regions) * 10000 / arr.size),
        "LowObj_MeanArea_px": float(np.mean(areas)),
        "LowObj_MedianArea_px": float(np.median(areas)),
        "LowObj_MeanCircularity": float(np.nanmean(circularities)),
        "LowObj_MeanSolidity": float(np.nanmean(solidities)),
        "LowObj_MeanEccentricity": float(np.nanmean(eccentricities)),
        "LowObj_MeanAspectRatio": float(np.nanmean(aspect_ratios)),
        "LowObj_TotalAreaRatio_filtered": float(np.sum(low_mask) / arr.size),
    }


# =========================
# metadata 检查
# =========================
def validate_metadata(meta: pd.DataFrame):
    required_cols = [
        "file_name",
        "case_id",
        "pathology",
        "main_group",
        "secondary_group",
        "roi_id",
        "use_flag",
        "note",
    ]

    missing = [c for c in required_cols if c not in meta.columns]
    if missing:
        raise ValueError(f"metadata.csv 缺少列: {missing}")

    allowed_main = {"malignant", "non_malignant"}
    bad_groups = set(meta["main_group"].dropna().unique()) - allowed_main
    if bad_groups:
        raise ValueError(f"main_group 只能写 malignant 或 non_malignant。发现错误: {bad_groups}")

    for fn in meta["file_name"]:
        if not (IMG_DIR / fn).exists():
            raise FileNotFoundError(f"metadata 中的文件不存在: {IMG_DIR / fn}")


# =========================
# 主程序
# =========================
def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    if not META_PATH.exists():
        raise FileNotFoundError(f"找不到 metadata.csv: {META_PATH}")

    meta = pd.read_csv(META_PATH)
    validate_metadata(meta)

    # 只分析 use_flag = yes 的 ROI
    meta["use_flag"] = meta["use_flag"].astype(str).str.lower()
    meta_use = meta[meta["use_flag"].isin(["yes", "y", "1", "true"])].copy()

    if meta_use.empty:
        raise ValueError("没有 use_flag=yes 的 ROI。")

    image_cache = {}
    all_pixels = []

    # 第一次遍历：读取、预处理、汇总全局像素
    for _, row in meta_use.iterrows():
        img_path = IMG_DIR / row["file_name"]

        arr = read_image_as_gray_uint8(img_path)
        arr = preprocess_image(arr, sigma=PARAMS["gaussian_sigma"])

        image_cache[row["file_name"]] = arr
        all_pixels.append(arr.ravel())

    all_pixels = np.concatenate(all_pixels).astype(np.uint8)

    # 全局阈值：保证所有图阈值一致，不单张自适应
    low_thr_percentile = float(np.percentile(all_pixels, PARAMS["low_signal_percentile"]))
    high_thr_percentile = float(np.percentile(all_pixels, PARAMS["high_signal_percentile"]))

    try:
        otsu_thr = float(threshold_otsu(all_pixels))
    except Exception:
        otsu_thr = low_thr_percentile

    # 最终低信号阈值：用 percentile，更稳定；Otsu 记录但不强制用
    low_thr = low_thr_percentile
    high_thr = high_thr_percentile

    threshold_info = {
        "low_signal_threshold_percentile": PARAMS["low_signal_percentile"],
        "high_signal_threshold_percentile": PARAMS["high_signal_percentile"],
        "low_signal_threshold_value": low_thr,
        "high_signal_threshold_value": high_thr,
        "otsu_threshold_reference": otsu_thr,
        "parameters": PARAMS,
    }

    with open(OUT_DIR / "feature_extraction_parameters.json", "w", encoding="utf-8") as f:
        json.dump(threshold_info, f, indent=4, ensure_ascii=False)

    rows = []

    for _, row in meta_use.iterrows():
        arr = image_cache[row["file_name"]]

        result = {
            "file_name": row["file_name"],
            "case_id": row["case_id"],
            "pathology": row["pathology"],
            "main_group": row["main_group"],
            "secondary_group": row["secondary_group"],
            "roi_id": row["roi_id"],
            "note": row["note"],
            "Image_Height_px": arr.shape[0],
            "Image_Width_px": arr.shape[1],
            "Image_Area_px": arr.size,
        }

        result.update(extract_gray_features(arr))
        result.update(
            extract_glcm_features(
                arr,
                levels=PARAMS["glcm_levels"],
                distances=PARAMS["glcm_distances"],
                angles_degree=PARAMS["glcm_angles_degree"],
            )
        )
        result.update(extract_ratio_features(arr, low_thr=low_thr, high_thr=high_thr))
        result.update(
            extract_low_signal_morphology(
                arr,
                low_thr=low_thr,
                min_object_size_px=PARAMS["min_object_size_px"],
            )
        )

        rows.append(result)

    roi_df = pd.DataFrame(rows)

    # 保存 ROI 级结果
    roi_out_csv = OUT_DIR / "ROI_level_features.csv"
    roi_out_xlsx = OUT_DIR / "ROI_level_features.xlsx"

    roi_df.to_csv(roi_out_csv, index=False, encoding="utf-8-sig")
    roi_df.to_excel(roi_out_xlsx, index=False)

    # 病例级平均：只对数值列求平均
    id_cols = ["case_id", "pathology", "main_group", "secondary_group"]
    numeric_cols = roi_df.select_dtypes(include=[np.number]).columns.tolist()

    # 不建议把图像尺寸列用于建模，但保留在 case 表里方便 QC
    case_df = roi_df.groupby(id_cols, as_index=False)[numeric_cols].mean()

    # 每个病例 ROI 数量
    roi_count = roi_df.groupby("case_id").size().reset_index(name="ROI_Count")
    case_df = case_df.merge(roi_count, on="case_id", how="left")

    case_out_csv = OUT_DIR / "Case_level_features.csv"
    case_out_xlsx = OUT_DIR / "Case_level_features.xlsx"

    case_df.to_csv(case_out_csv, index=False, encoding="utf-8-sig")
    case_df.to_excel(case_out_xlsx, index=False)

    # QC 表：检查每个病例有几个 ROI
    qc = (
        roi_df.groupby(["case_id", "pathology", "main_group", "secondary_group"])
        .agg(
            ROI_Count=("roi_id", "count"),
            Mean_Image_Area_px=("Image_Area_px", "mean"),
            Mean_Mean=("Mean", "mean"),
            SD_Mean=("Mean", "std"),
            Mean_Entropy=("GLCM_entropy", "mean"),
            SD_Entropy=("GLCM_entropy", "std"),
        )
        .reset_index()
    )

    qc_out = OUT_DIR / "QC_case_roi_summary.xlsx"
    qc.to_excel(qc_out, index=False)

    print("\n========== 特征提取完成 ==========")
    print(f"使用 ROI 数量: {len(roi_df)}")
    print(f"病例数量: {case_df['case_id'].nunique()}")
    print(f"低信号阈值 low_thr = {low_thr:.3f}")
    print(f"高信号阈值 high_thr = {high_thr:.3f}")
    print(f"ROI级特征表: {roi_out_csv}")
    print(f"病例级特征表: {case_out_csv}")
    print(f"QC检查表: {qc_out}")
    print("参数记录: feature_extraction_parameters.json")
    print("=================================\n")


if __name__ == "__main__":
    main()