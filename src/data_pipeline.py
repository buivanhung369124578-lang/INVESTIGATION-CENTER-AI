from __future__ import annotations

import io
import re
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "transaction_id",
    "customer_id",
    "date",
    "hour",
    "amount",
    "category",
    "merchant",
    "location",
    "payment_method",
    "device",
]

COLUMN_ALIASES = {
    "transaction_id": ["transaction_id", "transactionid", "transaction id", "ma giao dich", "mã giao dịch", "id giao dich", "id", "MaGiaoDich"],
    "customer_id": ["customer_id", "customerid", "customer id", "ma khach hang", "mã khách hàng", "khach hang", "customer", "nguoi thuc hien", "người thực hiện", "NguoiThucHien"],
    "date": ["date", "datetime", "timestamp", "time", "ngay", "ngày", "ngay giao dich", "ngày giao dịch"],
    "hour": ["hour", "gio", "giờ", "transaction_hour", "transaction hour"],
    "amount": ["amount", "transaction_amount", "transaction amount", "value", "gia tri", "giá trị", "so tien", "số tiền", "tien", "tiền", "SoTien"],
    "category": ["category", "danh muc", "danh mục", "loai giao dich", "loại giao dịch", "nhom chi phi", "nhóm chi phí", "NhomChiPhi"],
    "merchant": ["merchant", "nguoi ban", "người bán", "merchant_name", "merchant name", "nha cung cap", "nhà cung cấp", "NhaCungCap"],
    "location": ["location", "dia diem", "địa điểm", "city", "thanh pho", "thành phố", "bo phan", "bộ phận", "BoPhan"],
    "payment_method": ["payment_method", "payment method", "phuong thuc thanh toan", "phương thức thanh toán", "method", "PhuongThucThanhToan"],
    "device": ["device", "thiet bi", "thiết bị", "device_id", "device id"],
}

TEXT_DEFAULT = "Không xác định"


# Chuyển các giá trị hiển thị phổ biến sang tiếng Việt; khóa/schema/model vẫn giữ nguyên.
VALUE_TRANSLATIONS = {
    "category": {
        "Food": "Ăn uống", "Shopping": "Mua sắm", "Transport": "Di chuyển",
        "Bills": "Hóa đơn", "Entertainment": "Giải trí", "Healthcare": "Y tế",
        "Electronics": "Điện tử",
        "Ăn uống": "Ăn uống", "Mua sắm": "Mua sắm", "Di chuyển": "Di chuyển",
        "Hóa đơn": "Hóa đơn", "Giải trí": "Giải trí", "Y tế": "Y tế", "Điện tử": "Điện tử",
    },
    "merchant": {
        "Supermarket": "Siêu thị", "Restaurant": "Nhà hàng", "Online Shop": "Cửa hàng trực tuyến",
        "Taxi": "Taxi", "Utility": "Dịch vụ tiện ích", "Cinema": "Rạp chiếu phim",
        "Pharmacy": "Nhà thuốc", "Electronics Store": "Cửa hàng điện tử",
    },
    "location": {
        "Ha Noi": "Hà Nội", "Hanoi": "Hà Nội", "Ho Chi Minh City": "TP. Hồ Chí Minh",
        "Da Nang": "Đà Nẵng", "Hai Phong": "Hải Phòng", "Can Tho": "Cần Thơ",
    },
    "payment_method": {
        "Card": "Thẻ", "E-wallet": "Ví điện tử", "Bank Transfer": "Chuyển khoản", "Cash": "Tiền mặt",
    },
    "device": {
        "Known Device": "Thiết bị quen thuộc", "New Device": "Thiết bị mới",
    },
}


def localize_transaction_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in VALUE_TRANSLATIONS.items():
        if col in out.columns:
            out[col] = out[col].astype("string").map(mapping).fillna(out[col].astype("string"))
    return out


def _normalize_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def standardize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Map common Vietnamese/English column names into the app's canonical schema."""
    out = df.copy()
    existing = {_normalize_name(c): c for c in out.columns}
    rename_map: dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            key = _normalize_name(alias)
            if key in existing:
                source = existing[key]
                rename_map[source] = canonical
                break

    if rename_map:
        out = out.rename(columns=rename_map)
    return out, {str(k): str(v) for k, v in rename_map.items()}


def parse_amount_value(value: Any) -> float:
    """Parse common numeric/currency formats without turning the model feature into text."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan

    text = re.sub(r"[₫đĐ]|VND|VNĐ|USD|EUR|GBP", "", text, flags=re.IGNORECASE)
    text = text.replace(" ", "")
    if not text:
        return np.nan

    # Both separators: infer decimal separator from the last occurrence.
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        if len(parts[-1]) == 3 and all(p.isdigit() for p in parts):
            text = "".join(parts)
        else:
            text = text.replace(",", ".")
    elif "." in text:
        parts = text.split(".")
        if len(parts[-1]) == 3 and all(p.isdigit() for p in parts):
            text = "".join(parts)

    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_amount_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)
    return series.map(parse_amount_value).astype(float)


def read_csv_flexible(uploaded_bytes: bytes) -> pd.DataFrame:
    """Read UTF-8/UTF-8-SIG/CP1258 CSVs commonly produced in Vietnam."""
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(uploaded_bytes), encoding=encoding)
        except Exception as exc:  # pragma: no cover - only used for unusual file encodings
            errors.append(f"{encoding}: {exc}")
    raise ValueError("Không thể đọc file CSV. Hãy kiểm tra encoding hoặc định dạng file.")


def read_uploaded_file(uploaded_file: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """Read a Streamlit UploadedFile or file-like object as CSV/XLSX/XLS.

    The public app calls this helper directly; keeping it here preserves the
    original app API while allowing arbitrary CSV/Excel uploads.
    """
    if uploaded_file is None:
        raise ValueError("Chưa chọn file dữ liệu.")

    # The current app may pass either a Streamlit UploadedFile/file-like object
    # OR raw bytes plus an optional filename as the second positional argument.
    # Support both forms so uploads remain compatible with older app code.
    name = str(getattr(uploaded_file, "name", "")).lower()
    if not name and args and isinstance(args[0], str):
        name = args[0].lower()

    if isinstance(uploaded_file, (bytes, bytearray, memoryview)):
        payload = bytes(uploaded_file)
    elif hasattr(uploaded_file, "getvalue"):
        payload = uploaded_file.getvalue()
    else:
        payload = uploaded_file.read()

    if isinstance(payload, str):
        payload = payload.encode("utf-8")

    if name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(io.BytesIO(payload))
        except Exception as exc:
            raise ValueError(f"Không thể đọc file Excel: {exc}") from exc

    # CSV is the default for unknown extensions so existing uploads continue
    # to work even when a browser supplies a generic MIME type.
    return read_csv_flexible(payload)



def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Validate and normalize raw transactions. No model-specific features are created here."""
    out, renamed = standardize_columns(df)
    # Optional fields can be derived/defaulted so business CSVs do not need
    # to be manually reshaped before analysis.
    if "hour" not in out.columns:
        out["hour"] = np.nan
        renamed["<derived>"] = "hour từ date; nếu date chỉ có ngày thì dùng 12:00"
    if "device" not in out.columns:
        out["device"] = TEXT_DEFAULT
        renamed["<derived>"] = renamed.get("<derived>", "") + "; device=Không xác định"

    missing = [c for c in CANONICAL_COLUMNS if c not in out.columns]
    if missing:
        return None, {
            "input_rows": int(len(df)),
            "valid_rows": 0,
            "dropped_rows": int(len(df)),
            "missing_columns": missing,
            "renamed_columns": renamed,
            "duplicate_transaction_ids": 0,
            "invalid_amounts": 0,
            "negative_amounts": 0,
            "invalid_dates": 0,
        }

    out = out[CANONICAL_COLUMNS].copy()
    input_rows = len(out)

    out["transaction_id"] = out["transaction_id"].astype("string").str.strip()
    out["customer_id"] = out["customer_id"].astype("string").str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    invalid_dates = int(out["date"].isna().sum())

    # Derive hour from a datetime column when the source does not provide one.
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce")
    missing_hours = out["hour"].isna() & out["date"].notna()
    out.loc[missing_hours, "hour"] = out.loc[missing_hours, "date"].dt.hour
    invalid_hours = int(out["hour"].isna().sum())
    out["hour"] = out["hour"].clip(0, 23)

    out["amount"] = parse_amount_series(out["amount"])
    invalid_amounts = int(out["amount"].isna().sum())
    negative_amounts = int((out["amount"] < 0).sum())

    text_cols = ["category", "merchant", "location", "payment_method", "device"]
    for col in text_cols:
        out[col] = out[col].astype("string").fillna(TEXT_DEFAULT).str.strip()
        out.loc[out[col].eq(""), col] = TEXT_DEFAULT

    duplicate_ids = int(out["transaction_id"].duplicated(keep=False).sum())
    invalid_id = out["transaction_id"].isna() | out["transaction_id"].eq("")
    invalid_customer = out["customer_id"].isna() | out["customer_id"].eq("")
    invalid_amount = out["amount"].isna() | (out["amount"] < 0)
    invalid = out["date"].isna() | out["hour"].isna() | invalid_amount | invalid_id | invalid_customer

    out = out.loc[~invalid].copy()
    out["hour"] = out["hour"].round().astype(int)
    out["amount"] = out["amount"].astype(float)

    # A single transaction timestamp is useful for chronological customer baselines.
    out = localize_transaction_values(out)
    out["timestamp"] = out["date"].dt.normalize() + pd.to_timedelta(out["hour"], unit="h")
    out["is_weekend"] = out["timestamp"].dt.dayofweek >= 5
    out["is_night"] = out["hour"].between(0, 5)

    quality = {
        "input_rows": int(input_rows),
        "valid_rows": int(len(out)),
        "dropped_rows": int(input_rows - len(out)),
        "missing_columns": [],
        "renamed_columns": renamed,
        "duplicate_transaction_ids": duplicate_ids,
        "invalid_amounts": invalid_amounts,
        "negative_amounts": negative_amounts,
        "invalid_dates": invalid_dates + invalid_hours,
    }
    return out.reset_index(drop=True), quality


def generate_demo_data(n: int = 12000, seed: int = 42) -> pd.DataFrame:
    """Generate a repeatable dataset with several distinct anomaly patterns."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=180, freq="D")
    date = pd.Series(rng.choice(dates, n))
    hour = np.clip(rng.normal(14, 4.8, n).round().astype(int), 0, 23)
    amount = np.maximum(rng.lognormal(np.log(230_000), 0.68, n).round(-2), 20_000)

    categories = ["Ăn uống", "Mua sắm", "Di chuyển", "Hóa đơn", "Giải trí", "Y tế", "Điện tử"]
    methods = ["Thẻ", "Ví điện tử", "Chuyển khoản", "Tiền mặt"]
    locations = ["Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ"]
    merchants = ["Siêu thị", "Nhà hàng", "Cửa hàng trực tuyến", "Taxi", "Dịch vụ tiện ích", "Rạp chiếu phim", "Nhà thuốc", "Cửa hàng điện tử"]

    df = pd.DataFrame({
        "transaction_id": [f"TX{100000+i}" for i in range(n)],
        "customer_id": rng.integers(1001, 1901, n).astype(str),
        "date": date,
        "hour": hour,
        "amount": amount.astype(float),
        "category": rng.choice(categories, n, p=[.27,.19,.16,.13,.09,.07,.09]),
        "merchant": rng.choice(merchants, n),
        "location": rng.choice(locations, n, p=[.27,.25,.18,.12,.18]),
        "payment_method": rng.choice(methods, n, p=[.42,.28,.20,.10]),
        "device": rng.choice(["Thiết bị quen thuộc", "Thiết bị quen thuộc", "Thiết bị quen thuộc", "Thiết bị mới"]),
    })

    # Pattern A: unusually large amounts
    anomaly_idx = rng.choice(n, size=max(1, int(n * 0.035)), replace=False)
    chunks = np.array_split(anomaly_idx, 5)
    for g in chunks:
        if len(g):
            df.loc[g, "amount"] *= rng.uniform(6, 35, len(g))
    # Pattern B: unusual hours
    for g in chunks[:4]:
        if len(g):
            df.loc[g, "hour"] = rng.choice([0,1,2,3,4,5], len(g))
    # Pattern C: new device
    for g in chunks[:3]:
        if len(g):
            df.loc[g, "device"] = "Thiết bị mới"
    # Pattern D: location shift
    for g in chunks[1:4]:
        if len(g):
            df.loc[g, "location"] = rng.choice(locations, len(g))
    # Pattern E: rapid activity – create same-hour bursts within a customer.
    burst_idx = chunks[4] if len(chunks) > 4 else np.array([], dtype=int)
    if len(burst_idx) >= 2:
        for i in range(0, len(burst_idx) - 1, 2):
            a, b = int(burst_idx[i]), int(burst_idx[i + 1])
            df.loc[b, "customer_id"] = df.loc[a, "customer_id"]
            df.loc[b, "date"] = df.loc[a, "date"]
            df.loc[b, "hour"] = df.loc[a, "hour"]

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create historical/customer-context features while avoiding look-ahead leakage."""
    out = df.copy()
    out = out.sort_values(["customer_id", "timestamp", "transaction_id"]).reset_index(drop=True)

    global_median = float(out["amount"].median()) if len(out) else 0.0
    global_mean = float(out["amount"].mean()) if len(out) else 0.0
    global_std = float(out["amount"].std(ddof=1)) if len(out) > 1 else 1.0
    global_std = max(global_std, 1.0)

    g_customer = out.groupby("customer_id", sort=False)
    out["customer_prev_count"] = g_customer.cumcount().astype(float)
    cumulative_sum = g_customer["amount"].cumsum()
    out["customer_prev_sum"] = cumulative_sum - out["amount"]
    out["customer_prev_avg_amount"] = (
        out["customer_prev_sum"] / out["customer_prev_count"].replace(0, np.nan)
    ).fillna(global_median)
    out["amount_vs_customer_prev_avg"] = (
        out["amount"] / out["customer_prev_avg_amount"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    cumulative_sq = g_customer["amount"].transform(lambda s: s.pow(2).cumsum())
    prev_sq_sum = cumulative_sq - out["amount"].pow(2)
    n = out["customer_prev_count"]
    prev_var = (prev_sq_sum - (out["customer_prev_sum"].pow(2) / n.replace(0, np.nan))) / (n - 1).replace(0, np.nan)
    out["customer_prev_std_amount"] = np.sqrt(prev_var.clip(lower=0)).replace([np.inf, -np.inf], np.nan).fillna(global_std)
    out["amount_customer_z"] = (
        (out["amount"] - out["customer_prev_avg_amount"]) /
        out["customer_prev_std_amount"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Historical categorical novelty.
    out["new_device_for_customer"] = out.groupby(["customer_id", "device"], sort=False).cumcount().eq(0)
    out["new_location_for_customer"] = out.groupby(["customer_id", "location"], sort=False).cumcount().eq(0)
    out["new_merchant_for_customer"] = out.groupby(["customer_id", "merchant"], sort=False).cumcount().eq(0)

    # Time-based behavior.
    prev_timestamp = out.groupby("customer_id", sort=False)["timestamp"].shift(1)
    out["minutes_since_previous"] = (out["timestamp"] - prev_timestamp).dt.total_seconds() / 60.0
    out["rapid_transaction"] = out["minutes_since_previous"].between(0, 120, inclusive="both")

    prev_hour_sum = g_customer["hour"].cumsum() - out["hour"]
    prev_hour_mean = (prev_hour_sum / out["customer_prev_count"].replace(0, np.nan)).fillna(out["hour"].mean() if len(out) else 14.0)
    circular_diff = (out["hour"] - prev_hour_mean).abs()
    out["customer_hour_deviation"] = np.minimum(circular_diff, 24 - circular_diff).clip(lower=0)

    out["amount_log"] = np.log1p(out["amount"].clip(lower=0))
    out["amount_percentile"] = out["amount"].rank(pct=True, method="average") * 100
    out["global_amount_z"] = (out["amount"] - global_mean) / global_std
    out["same_day_customer_count_before"] = out.groupby(["customer_id", out["timestamp"].dt.date], sort=False).cumcount().astype(float)
    out["is_night"] = out["hour"].between(0, 5)
    out["is_weekend"] = out["timestamp"].dt.dayofweek >= 5
    out["weekday"] = out["timestamp"].dt.day_name()
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)

    return out
