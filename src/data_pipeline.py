from __future__ import annotations

import io
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "transaction_id", "customer_id", "date", "hour", "amount", "category",
    "merchant", "location", "payment_method", "device",
]

# Aliases intentionally include Vietnamese/English business terminology.
COLUMN_ALIASES = {
    "transaction_id": ["transaction_id", "transactionid", "transaction id", "ma giao dich", "mã giao dịch", "id giao dich", "id", "ma gd", "mã gd", "code", "reference", "ref"],
    "customer_id": ["customer_id", "customerid", "customer id", "ma khach hang", "mã khách hàng", "khach hang", "customer", "customer code", "user", "user id", "nguoi thuc hien", "người thực hiện", "employee", "employee id", "staff", "staff id", "account", "account id"],
    "date": ["date", "datetime", "timestamp", "time", "ngay", "ngày", "ngay giao dich", "ngày giao dịch", "transaction date", "transaction datetime", "created at", "created_at", "thoi gian", "thời gian"],
    "hour": ["hour", "gio", "giờ", "transaction_hour", "transaction hour", "gio giao dich", "giờ giao dịch"],
    "amount": ["amount", "transaction_amount", "transaction amount", "value", "gia tri", "giá trị", "so tien", "số tiền", "tien", "tiền", "total", "total amount", "price", "cost", "revenue", "doanh thu", "chi phi", "chi phí", "thanh tien", "thành tiền", "amount vnd", "amount usd"],
    "category": ["category", "danh muc", "danh mục", "loai giao dich", "loại giao dịch", "nhom chi phi", "nhóm chi phí", "transaction type", "product category"],
    "merchant": ["merchant", "nguoi ban", "người bán", "merchant_name", "merchant name", "vendor", "vendor name", "supplier", "supplier name", "nha cung cap", "nhà cung cấp", "store", "shop", "counterparty", "payee"],
    "location": ["location", "dia diem", "địa điểm", "city", "thanh pho", "thành phố", "address", "dia chi", "địa chỉ", "branch", "chi nhanh", "chi nhánh", "region", "province", "tinh", "tỉnh"],
    "payment_method": ["payment_method", "payment method", "phuong thuc thanh toan", "phương thức thanh toán", "method", "payment", "pay method", "hinh thuc thanh toan", "hình thức thanh toán"],
    "device": ["device", "thiet bi", "thiết bị", "device_id", "device id", "terminal", "terminal id", "channel", "kenh", "kênh", "source", "nguon", "nguồn"],
}
TEXT_DEFAULT = "Không xác định"

VALUE_TRANSLATIONS = {
    "category": {"Food": "Ăn uống", "Shopping": "Mua sắm", "Transport": "Di chuyển", "Bills": "Hóa đơn", "Entertainment": "Giải trí", "Healthcare": "Y tế", "Electronics": "Điện tử"},
    "merchant": {"Supermarket": "Siêu thị", "Restaurant": "Nhà hàng", "Online Shop": "Cửa hàng trực tuyến", "Taxi": "Taxi", "Utility": "Dịch vụ tiện ích", "Cinema": "Rạp chiếu phim", "Pharmacy": "Nhà thuốc", "Electronics Store": "Cửa hàng điện tử"},
    "location": {"Ha Noi": "Hà Nội", "Hanoi": "Hà Nội", "Ho Chi Minh City": "TP. Hồ Chí Minh", "Da Nang": "Đà Nẵng", "Hai Phong": "Hải Phòng", "Can Tho": "Cần Thơ"},
    "payment_method": {"Card": "Thẻ", "E-wallet": "Ví điện tử", "Bank Transfer": "Chuyển khoản", "Cash": "Tiền mặt"},
    "device": {"Known Device": "Thiết bị quen thuộc", "New Device": "Thiết bị mới"},
}


def _normalize_name(name: Any) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name).strip())
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text


def _name_similarity(column: Any, alias: str) -> float:
    c = _normalize_name(column)
    a = _normalize_name(alias)
    if not c or not a:
        return 0.0
    if c == a:
        return 1.0
    if a in c or c in a:
        return 0.88
    ct, at = set(c.split()), set(a.split())
    return len(ct & at) / max(len(at), 1) * 0.75


def parse_amount_value(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).strip()
    if not text:
        return np.nan
    text = re.sub(r"[₫đĐ]|VND|VNĐ|USD|EUR|GBP", "", text, flags=re.IGNORECASE).replace(" ", "")
    if not text:
        return np.nan
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".") if text.rfind(",") > text.rfind(".") else text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        text = "".join(parts) if len(parts[-1]) == 3 and all(p.isdigit() for p in parts) else text.replace(",", ".")
    elif "." in text:
        parts = text.split(".")
        text = "".join(parts) if len(parts[-1]) == 3 and all(p.isdigit() for p in parts) else text
    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_amount_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype(float)
    return series.map(parse_amount_value).astype(float)


def _candidate_score(series: pd.Series, kind: str) -> float:
    nonnull = series.dropna()
    if nonnull.empty:
        return 0.0
    if kind == "amount":
        parsed = parse_amount_series(series)
        valid = parsed.notna().mean()
        if valid < 0.65:
            return 0.0
        values = parsed.dropna()
        positive = (values >= 0).mean() if len(values) else 0
        # Prefer numeric-looking columns with meaningful variation.
        variation = min(values.nunique() / max(len(values), 1) * 4, 1.0)
        return float(valid * positive * (0.65 + 0.35 * variation))
    if kind == "date":
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
        return float(parsed.notna().mean())
    if kind == "id":
        unique_ratio = nonnull.astype(str).nunique() / max(len(nonnull), 1)
        return float(0.5 + 0.5 * unique_ratio) if unique_ratio > 0.25 else 0.0
    return 0.0


def standardize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    """Map known columns and infer reasonable transaction columns when names are unfamiliar."""
    out = df.copy()
    available = list(out.columns)
    used: set[Any] = set()
    rename_map: dict[Any, str] = {}
    confidence: dict[str, float] = {}

    # Pass 1: exact/fuzzy semantic name matching.
    for canonical, aliases in COLUMN_ALIASES.items():
        candidates = sorted(((max(_name_similarity(c, a) for a in aliases), c) for c in available if c not in used), reverse=True)
        if candidates and candidates[0][0] >= 0.72:
            score, source = candidates[0]
            rename_map[source] = canonical
            used.add(source)
            confidence[canonical] = round(float(score), 2)

    # Pass 2: data-shape inference for still-missing core columns.
    def infer(kind: str, exclude: set[Any]) -> Any | None:
        scores = [( _candidate_score(out[c], kind), c) for c in available if c not in exclude]
        scores.sort(reverse=True)
        return scores[0][1] if scores and scores[0][0] >= 0.65 else None

    if "amount" not in rename_map.values():
        c = infer("amount", used)
        if c is not None:
            rename_map[c] = "amount"; used.add(c); confidence["amount"] = 0.65
    if "date" not in rename_map.values():
        c = infer("date", used)
        if c is not None:
            rename_map[c] = "date"; used.add(c); confidence["date"] = 0.65
    if "transaction_id" not in rename_map.values():
        c = infer("id", used)
        if c is not None:
            rename_map[c] = "transaction_id"; used.add(c); confidence["transaction_id"] = 0.65

    if rename_map:
        out = out.rename(columns=rename_map)
    return out, {str(k): str(v) for k, v in rename_map.items()}, confidence


def _make_fallback_columns(out: pd.DataFrame, original_rows: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Fill optional schema fields without inventing transaction amounts."""
    inferred: list[str] = []
    warnings: list[str] = []
    n = len(out)

    if "transaction_id" not in out:
        out["transaction_id"] = [f"TX-{i+1:07d}" for i in range(n)]
        inferred.append("transaction_id")
    if "customer_id" not in out:
        # A stable fallback keeps the pipeline valid but disables true customer-history signals.
        out["customer_id"] = out["transaction_id"].astype("string")
        inferred.append("customer_id")
        warnings.append("Không tìm thấy cột khách hàng/người dùng; hệ thống dùng transaction_id làm nhóm tạm thời nên tín hiệu lịch sử khách hàng bị hạn chế.")
    if "date" not in out:
        out["date"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(np.arange(n), unit="m")
        inferred.append("date")
        warnings.append("Không tìm thấy cột ngày/thời gian; hệ thống tạo trục thời gian kỹ thuật để model vẫn chạy.")
    if "hour" not in out:
        out["hour"] = pd.Series(pd.to_datetime(out["date"], errors="coerce", dayfirst=True).dt.hour, index=out.index)
        inferred.append("hour")
    if "category" not in out:
        out["category"] = TEXT_DEFAULT; inferred.append("category")
    if "merchant" not in out:
        out["merchant"] = TEXT_DEFAULT; inferred.append("merchant")
    if "location" not in out:
        out["location"] = TEXT_DEFAULT; inferred.append("location")
    if "payment_method" not in out:
        out["payment_method"] = TEXT_DEFAULT; inferred.append("payment_method")
    if "device" not in out:
        out["device"] = TEXT_DEFAULT; inferred.append("device")
    return out, inferred, warnings


def read_csv_flexible(uploaded_bytes: bytes) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        for sep in (None, ",", ";", "\t", "|"):
            try:
                kwargs = {"encoding": encoding}
                if sep is None:
                    kwargs.update({"sep": None, "engine": "python"})
                else:
                    kwargs["sep"] = sep
                df = pd.read_csv(io.BytesIO(uploaded_bytes), **kwargs)
                if df.shape[1] >= 2:
                    return df
            except Exception as exc:
                errors.append(f"{encoding}/{sep}: {exc}")
    raise ValueError("Không thể đọc file CSV. Hãy kiểm tra encoding hoặc định dạng file.")


def read_uploaded_file(uploaded_bytes: bytes, filename: str) -> pd.DataFrame:
    """Read CSV/XLS/XLSX with automatic format detection."""
    lower = filename.lower()
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(uploaded_bytes))
    return read_csv_flexible(uploaded_bytes)


def localize_transaction_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in VALUE_TRANSLATIONS.items():
        if col in out.columns:
            original = out[col].astype("string")
            out[col] = original.map(mapping).fillna(original)
    return out


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """Automatically adapt a broad range of transaction tables to the model schema."""
    if df is None or df.empty:
        return None, {"input_rows": 0, "valid_rows": 0, "dropped_rows": 0, "missing_columns": ["amount"], "renamed_columns": {}, "inferred_columns": [], "warnings": ["File không có dữ liệu."]}

    raw = df.copy()
    out, renamed, confidence = standardize_columns(raw)
    out, inferred, warnings = _make_fallback_columns(out, len(raw))

    # Amount is the one genuinely essential field. Never fabricate it.
    if "amount" not in out.columns:
        return None, {
            "input_rows": int(len(raw)), "valid_rows": 0, "dropped_rows": int(len(raw)),
            "missing_columns": ["amount"], "renamed_columns": renamed, "inferred_columns": inferred,
            "mapping_confidence": confidence, "warnings": warnings + ["Không tìm thấy cột số tiền/giá trị. Hãy chỉ định một cột tiền trong file."],
        }

    out = out[CANONICAL_COLUMNS].copy()
    input_rows = len(out)
    out["transaction_id"] = out["transaction_id"].astype("string").str.strip()
    out["customer_id"] = out["customer_id"].astype("string").str.strip()
    out["date"] = pd.to_datetime(out["date"], errors="coerce", dayfirst=True)
    invalid_dates = int(out["date"].isna().sum())

    out["hour"] = pd.to_numeric(out["hour"], errors="coerce")
    missing_hours = out["hour"].isna() & out["date"].notna()
    out.loc[missing_hours, "hour"] = out.loc[missing_hours, "date"].dt.hour
    invalid_hours = int(out["hour"].isna().sum())
    out["hour"] = out["hour"].fillna(12).clip(0, 23)

    out["amount"] = parse_amount_series(out["amount"])
    invalid_amounts = int(out["amount"].isna().sum())
    negative_amounts = int((out["amount"] < 0).sum())

    for col in ["category", "merchant", "location", "payment_method", "device"]:
        out[col] = out[col].astype("string").fillna(TEXT_DEFAULT).str.strip()
        out.loc[out[col].eq(""), col] = TEXT_DEFAULT

    duplicate_ids = int(out["transaction_id"].duplicated(keep=False).sum())
    invalid_id = out["transaction_id"].isna() | out["transaction_id"].eq("")
    invalid_amount = out["amount"].isna() | (out["amount"] < 0)
    invalid = out["date"].isna() | invalid_amount | invalid_id
    out = out.loc[~invalid].copy()

    if out.empty:
        return None, {
            "input_rows": int(input_rows), "valid_rows": 0, "dropped_rows": int(input_rows),
            "missing_columns": [], "renamed_columns": renamed, "inferred_columns": inferred,
            "mapping_confidence": confidence, "warnings": warnings + ["Không còn dòng hợp lệ sau khi kiểm tra ngày và số tiền."],
            "invalid_amounts": invalid_amounts, "negative_amounts": negative_amounts, "invalid_dates": invalid_dates,
        }

    out["hour"] = out["hour"].round().astype(int)
    out["amount"] = out["amount"].astype(float)
    out = localize_transaction_values(out)
    out["timestamp"] = out["date"].dt.normalize() + pd.to_timedelta(out["hour"], unit="h")
    out["is_weekend"] = out["timestamp"].dt.dayofweek >= 5
    out["is_night"] = out["hour"].between(0, 5)

    quality = {
        "input_rows": int(input_rows), "valid_rows": int(len(out)), "dropped_rows": int(input_rows - len(out)),
        "missing_columns": [], "renamed_columns": renamed, "inferred_columns": inferred,
        "mapping_confidence": confidence, "warnings": warnings,
        "duplicate_transaction_ids": duplicate_ids, "invalid_amounts": invalid_amounts,
        "negative_amounts": negative_amounts, "invalid_dates": invalid_dates + invalid_hours,
    }
    return out.reset_index(drop=True), quality


def generate_demo_data(n: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=180, freq="D")
    date = pd.Series(rng.choice(dates, n))
    hour = np.clip(rng.normal(14, 4.8, n).round().astype(int), 0, 23)
    amount = np.maximum(rng.lognormal(np.log(230_000), 0.68, n).round(-2), 20_000)
    categories = ["Ăn uống", "Mua sắm", "Di chuyển", "Hóa đơn", "Giải trí", "Y tế", "Điện tử"]
    methods = ["Thẻ", "Ví điện tử", "Chuyển khoản", "Tiền mặt"]
    locations = ["Hà Nội", "TP. Hồ Chí Minh", "Đà Nẵng", "Hải Phòng", "Cần Thơ"]
    merchants = ["Siêu thị", "Nhà hàng", "Cửa hàng trực tuyến", "Taxi", "Dịch vụ tiện ích", "Rạp chiếu phim", "Nhà thuốc", "Cửa hàng điện tử"]
    df = pd.DataFrame({"transaction_id": [f"TX{100000+i}" for i in range(n)], "customer_id": rng.integers(1001, 1901, n).astype(str), "date": date, "hour": hour, "amount": amount.astype(float), "category": rng.choice(categories, n, p=[.27,.19,.16,.13,.09,.07,.09]), "merchant": rng.choice(merchants, n), "location": rng.choice(locations, n, p=[.27,.25,.18,.12,.18]), "payment_method": rng.choice(methods, n, p=[.42,.28,.20,.10]), "device": rng.choice(["Thiết bị quen thuộc", "Thiết bị quen thuộc", "Thiết bị quen thuộc", "Thiết bị mới"] )})
    anomaly_idx = rng.choice(n, size=max(1, int(n * 0.035)), replace=False)
    chunks = np.array_split(anomaly_idx, 5)
    for g in chunks:
        if len(g): df.loc[g, "amount"] *= rng.uniform(6, 35, len(g))
    for g in chunks[:4]:
        if len(g): df.loc[g, "hour"] = rng.choice([0,1,2,3,4,5], len(g))
    for g in chunks[:3]:
        if len(g): df.loc[g, "device"] = "Thiết bị mới"
    for g in chunks[1:4]:
        if len(g): df.loc[g, "location"] = rng.choice(locations, len(g))
    burst_idx = chunks[4] if len(chunks) > 4 else np.array([], dtype=int)
    if len(burst_idx) >= 2:
        for i in range(0, len(burst_idx)-1, 2):
            a, b = int(burst_idx[i]), int(burst_idx[i+1]); df.loc[b, "customer_id"] = df.loc[a, "customer_id"]; df.loc[b, "date"] = df.loc[a, "date"]; df.loc[b, "hour"] = df.loc[a, "hour"]
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["customer_id", "timestamp", "transaction_id"]).reset_index(drop=True)
    global_median = float(out["amount"].median()) if len(out) else 0.0
    global_mean = float(out["amount"].mean()) if len(out) else 0.0
    global_std = max(float(out["amount"].std(ddof=1)) if len(out) > 1 else 1.0, 1.0)
    g_customer = out.groupby("customer_id", sort=False)
    out["customer_prev_count"] = g_customer.cumcount().astype(float)
    cumulative_sum = g_customer["amount"].cumsum(); out["customer_prev_sum"] = cumulative_sum - out["amount"]
    out["customer_prev_avg_amount"] = (out["customer_prev_sum"] / out["customer_prev_count"].replace(0, np.nan)).fillna(global_median)
    out["amount_vs_customer_prev_avg"] = (out["amount"] / out["customer_prev_avg_amount"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    cumulative_sq = g_customer["amount"].transform(lambda s: s.pow(2).cumsum()); prev_sq_sum = cumulative_sq - out["amount"].pow(2); n = out["customer_prev_count"]
    prev_var = (prev_sq_sum - (out["customer_prev_sum"].pow(2) / n.replace(0, np.nan))) / (n - 1).replace(0, np.nan)
    out["customer_prev_std_amount"] = np.sqrt(prev_var.clip(lower=0)).replace([np.inf, -np.inf], np.nan).fillna(global_std)
    out["amount_customer_z"] = ((out["amount"] - out["customer_prev_avg_amount"]) / out["customer_prev_std_amount"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["new_device_for_customer"] = out.groupby(["customer_id", "device"], sort=False).cumcount().eq(0)
    out["new_location_for_customer"] = out.groupby(["customer_id", "location"], sort=False).cumcount().eq(0)
    out["new_merchant_for_customer"] = out.groupby(["customer_id", "merchant"], sort=False).cumcount().eq(0)
    prev_timestamp = out.groupby("customer_id", sort=False)["timestamp"].shift(1)
    out["minutes_since_previous"] = (out["timestamp"] - prev_timestamp).dt.total_seconds() / 60.0
    out["rapid_transaction"] = out["minutes_since_previous"].between(0, 120, inclusive="both")
    prev_hour_sum = g_customer["hour"].cumsum() - out["hour"]
    prev_hour_mean = (prev_hour_sum / out["customer_prev_count"].replace(0, np.nan)).fillna(out["hour"].mean() if len(out) else 14.0)
    circular_diff = (out["hour"] - prev_hour_mean).abs(); out["customer_hour_deviation"] = np.minimum(circular_diff, 24-circular_diff).clip(lower=0)
    out["amount_log"] = np.log1p(out["amount"].clip(lower=0)); out["amount_percentile"] = out["amount"].rank(pct=True, method="average") * 100; out["global_amount_z"] = (out["amount"]-global_mean)/global_std
    out["same_day_customer_count_before"] = out.groupby(["customer_id", out["timestamp"].dt.date], sort=False).cumcount().astype(float)
    out["is_night"] = out["hour"].between(0,5); out["is_weekend"] = out["timestamp"].dt.dayofweek >= 5; out["weekday"] = out["timestamp"].dt.day_name(); out["hour_sin"] = np.sin(2*np.pi*out["hour"]/24.0); out["hour_cos"] = np.cos(2*np.pi*out["hour"]/24.0)
    return out
