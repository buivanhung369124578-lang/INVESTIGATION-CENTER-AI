from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_pipeline import engineer_features


NUMERIC_FEATURES = [
    "amount_log",
    "hour_sin",
    "hour_cos",
    "customer_prev_count",
    "customer_prev_avg_amount",
    "amount_vs_customer_prev_avg",
    "amount_customer_z",
    "customer_hour_deviation",
    "minutes_since_previous",
    "same_day_customer_count_before",
    "amount_percentile",
    "global_amount_z",
]

CATEGORICAL_FEATURES = [
    "category",
    "merchant",
    "location",
    "payment_method",
    "device",
    "weekday",
]

SIGNAL_WEIGHTS = {
    "amount_top_1pct": (30, "Số tiền nằm trong 1% cao nhất của toàn bộ dữ liệu"),
    "amount_vs_customer": (28, "Số tiền cao hơn đáng kể so với mức chi tiêu lịch sử của khách hàng"),
    "new_device": (20, "Thiết bị chưa từng được quan sát với khách hàng này"),
    "new_location": (15, "Địa điểm chưa từng xuất hiện trong lịch sử của khách hàng"),
    "night": (12, "Giao dịch trong khung 00:00–05:00"),
    "rapid": (10, "Có giao dịch khác của khách hàng trong vòng 120 phút"),
    "new_merchant": (7, "Người bán chưa từng xuất hiện với khách hàng này"),
    "weekend": (4, "Giao dịch phát sinh vào cuối tuần"),
}


def _make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # sklearn < 1.2 compatibility
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _build_pipeline(contamination: float, n_estimators: int, random_state: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", _make_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("pre", preprocessor),
            (
                "iforest",
                IsolationForest(
                    n_estimators=n_estimators,
                    contamination=contamination,
                    random_state=random_state,
                    n_jobs=-1,
                    bootstrap=False,
                ),
            ),
        ]
    )


def _build_explanations(work: pd.DataFrame) -> pd.DataFrame:
    q99 = work["amount"].quantile(0.99)
    signal_frames = {
        "amount_top_1pct": work["amount"].ge(q99),
        "amount_vs_customer": work["amount_vs_customer_prev_avg"].ge(3.0),
        "new_device": work["new_device_for_customer"].astype(bool),
        "new_location": work["new_location_for_customer"].astype(bool),
        "night": work["is_night"].astype(bool),
        "rapid": work["rapid_transaction"].astype(bool),
        "new_merchant": work["new_merchant_for_customer"].astype(bool),
        "weekend": work["is_weekend"].astype(bool),
    }
    context_score = np.zeros(len(work), dtype=float)
    reasons: list[str] = []
    counts = np.zeros(len(work), dtype=int)
    for i in range(len(work)):
        row_reasons = []
        score = 0.0
        for key, (weight, label) in SIGNAL_WEIGHTS.items():
            if bool(signal_frames[key].iloc[i]):
                score += weight
                row_reasons.append(label)
        context_score[i] = min(score, 100.0)
        counts[i] = len(row_reasons)
        reasons.append("; ".join(row_reasons) if row_reasons else "Chưa có tín hiệu quy tắc nổi bật")

    work["context_signal_score"] = np.round(context_score, 1)
    work["signal_count"] = counts
    work["reason"] = reasons
    return work


def run_detection(
    df: pd.DataFrame,
    contamination: float = 0.035,
    n_estimators: int = 300,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = engineer_features(df)
    model = _build_pipeline(contamination, n_estimators, random_state)
    X = work[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    # Median imputation at this stage keeps the feature space robust to first-transaction nulls.
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        X[col] = X[col].fillna(float(work[col].median()) if work[col].notna().any() else 0.0)
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("string").fillna("Không xác định")

    prediction = model.fit_predict(X)
    decision = model.decision_function(X)
    anomaly_strength = -decision

    # Relative rank inside the analyzed dataset: 100 means among the highest anomaly scores.
    risk_score = pd.Series(anomaly_strength).rank(pct=True, method="average").to_numpy() * 100.0

    work["anomaly_score_raw"] = anomaly_strength.astype(float)
    work["risk_score"] = np.round(np.clip(risk_score, 0, 100), 1)
    work["is_anomaly"] = prediction == -1
    work["status"] = np.where(work["is_anomaly"], "Bất thường", "Bình thường")
    work["risk_level"] = pd.cut(
        work["risk_score"],
        bins=[-0.01, 49.9, 79.9, 94.9, 100.01],
        labels=["Thấp", "Trung bình", "Cao", "Nghiêm trọng"],
        include_lowest=True,
        ordered=True,
    )
    work = _build_explanations(work)
    work["priority"] = pd.Categorical(
        np.select(
            [work["risk_level"].eq("Nghiêm trọng"), work["risk_level"].eq("Cao"), work["is_anomaly"]],
            ["Điều tra ngay", "Ưu tiên xem", "Theo dõi"],
            default="Thấp",
        ),
        categories=["Điều tra ngay", "Ưu tiên xem", "Theo dõi", "Thấp"],
        ordered=True,
    )

    work = work.sort_values(["risk_score", "timestamp"], ascending=[False, True]).reset_index(drop=True)

    transformed_dim = None
    try:
        transformed_dim = int(model.named_steps["pre"].get_feature_names_out().shape[0])
    except Exception:
        transformed_dim = None

    quantiles = np.quantile(anomaly_strength, [0.5, 0.9, 0.95, 0.99]).tolist() if len(anomaly_strength) else [0, 0, 0, 0]
    info = {
        "algorithm": "Isolation Forest",
        "n_estimators": int(n_estimators),
        "contamination": float(contamination),
        "random_state": int(random_state),
        "training_rows": int(len(work)),
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "raw_feature_count": int(len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)),
        "transformed_feature_count": transformed_dim,
        "flagged_anomalies": int(work["is_anomaly"].sum()),
        "flagged_rate": float(work["is_anomaly"].mean() * 100) if len(work) else 0.0,
        "decision_threshold": float(model.named_steps["iforest"].offset_),
        "anomaly_strength_quantiles": quantiles,
    }
    return work, info
