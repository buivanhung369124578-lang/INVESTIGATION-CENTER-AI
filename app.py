from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_pipeline import (
    CANONICAL_COLUMNS,
    generate_demo_data,
    prepare_data,
    read_csv_flexible,
)
from src.model_engine import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SIGNAL_WEIGHTS,
    run_detection,
)
from src.reporting import build_html_report, money, percent


st.set_page_config(
    page_title="INVESTIGATION CENTER AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Design system
# -----------------------------
st.markdown(
    """
<style>
:root { --ink:#162033; --muted:#657189; --border:#e6eaf0; --surface:#ffffff; --bg:#f6f8fb; }
.block-container { padding-top: 1.1rem; padding-bottom: 3.5rem; }
[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stSidebar"] { background:#0f172a; }
[data-testid="stSidebar"] * { color:#e5e7eb; }
[data-testid="stSidebar"] .stCaption { color:#94a3b8; }
.hero { background:linear-gradient(135deg,#ffffff 0%,#f1f5ff 100%); border:1px solid var(--border); border-radius:20px; padding:1.3rem 1.5rem; margin-bottom:1rem; box-shadow:0 8px 30px rgba(15,23,42,.04); }
.hero h1 { margin:0; font-size:2.05rem; letter-spacing:-.02em; color:var(--ink); }
.hero p { margin:.35rem 0 0; color:var(--muted); font-size:.98rem; }
.section-title { font-size:1.35rem; font-weight:750; color:var(--ink); margin:.2rem 0 .8rem; }
.section-subtitle { color:var(--muted); margin-top:-.45rem; margin-bottom:.9rem; }
.card-note { border:1px solid var(--border); border-radius:14px; padding:14px 16px; background:var(--surface); }
.metric-strip { border:1px solid var(--border); border-radius:14px; padding:.55rem .75rem; background:white; }
.small-note { color:var(--muted); font-size:.86rem; }
hr { border:0; border-top:1px solid var(--border); margin:1.3rem 0; }
.priority-critical { background:#fff1f2; border:1px solid #fecdd3; }
.priority-high { background:#fff7ed; border:1px solid #fed7aa; }
.priority-medium { background:#fffbeb; border:1px solid #fde68a; }
.priority-low { background:#f0fdf4; border:1px solid #bbf7d0; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Cached data/model work
# -----------------------------
@st.cache_data(show_spinner=False)
def cached_prepare(raw: pd.DataFrame):
    return prepare_data(raw)


@st.cache_data(show_spinner=False)
def cached_detection(data: pd.DataFrame, contamination: float, n_estimators: int, seed: int):
    return run_detection(data, contamination=contamination, n_estimators=n_estimators, random_state=seed)


# -----------------------------
# Helpers
# -----------------------------
def fmt_score(v: float) -> str:
    return f"{float(v):.1f}/100"


def get_risk_color(level: str) -> str:
    return {
        "Nghiêm trọng": "#b91c1c",
        "Cao": "#ea580c",
        "Trung bình": "#ca8a04",
        "Thấp": "#16a34a",
    }.get(str(level), "#64748b")


def safe_value(v) -> str:
    if pd.isna(v):
        return "—"
    return str(v)


def details_table(row: pd.Series) -> pd.DataFrame:
    # Keep display columns homogeneous to avoid the PyArrow mixed-object problem.
    details = [
        ("Transaction ID", safe_value(row["transaction_id"])),
        ("Customer ID", safe_value(row["customer_id"])),
        ("Thời gian", row["timestamp"].strftime("%Y-%m-%d %H:00") if not pd.isna(row["timestamp"]) else "—"),
        ("Danh mục", safe_value(row["category"])),
        ("Người bán", safe_value(row["merchant"])),
        ("Địa điểm", safe_value(row["location"])),
        ("Phương thức thanh toán", safe_value(row["payment_method"])),
        ("Thiết bị", safe_value(row["device"])),
        ("Số tiền", money(row["amount"])),
        ("Risk Score", fmt_score(row["risk_score"])),
        ("Mức rủi ro", safe_value(row["risk_level"])),
        ("Trạng thái model", safe_value(row["status"])),
        ("Mức độ ưu tiên", safe_value(row["priority"])),
        ("Tỷ lệ so với baseline khách hàng", f"{float(row['amount_vs_customer_prev_avg']):.2f}x"),
        ("Số giao dịch trước đó của khách hàng", f"{int(row['customer_prev_count'])}"),
        ("Số phút từ giao dịch trước", "Chưa có" if pd.isna(row["minutes_since_previous"]) else f"{float(row['minutes_since_previous']):.1f}"),
        ("Thiết bị mới", "Có" if bool(row["new_device_for_customer"]) else "Không"),
        ("Địa điểm mới", "Có" if bool(row["new_location_for_customer"]) else "Không"),
        ("Giao dịch ban đêm", "Có" if bool(row["is_night"]) else "Không"),
    ]
    return pd.DataFrame(details, columns=["Thuộc tính", "Giá trị"])


def make_risk_gauge(score: float) -> go.Figure:
    score = float(score)
    if score >= 95:
        label = "NGHIÊM TRỌNG"
    elif score >= 80:
        label = "CAO"
    elif score >= 50:
        label = "TRUNG BÌNH"
    else:
        label = "THẤP"
    color_key = {
        "NGHIÊM TRỌNG": "Nghiêm trọng",
        "CAO": "Cao",
        "TRUNG BÌNH": "Trung bình",
        "THẤP": "Thấp",
    }[label]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 38}},
            title={"text": f"Risk Score — {label}"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 50], "color": "#dcfce7"},
                    {"range": [50, 80], "color": "#fef3c7"},
                    {"range": [80, 95], "color": "#ffedd5"},
                    {"range": [95, 100], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": get_risk_color(color_key), "width": 4}, "value": score},
            },
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=65, b=10))
    return fig


def kpi_block(label: str, value: str, help_text: str | None = None):
    st.metric(label, value, help=help_text)


# -----------------------------
# Sidebar configuration
# -----------------------------
st.sidebar.markdown("## 🛡️ INVESTIGATION CENTER AI")
st.sidebar.caption("Transaction anomaly detection • investigation • customer behavior")
st.sidebar.markdown("---")

source = st.sidebar.radio("Nguồn dữ liệu", ["Dữ liệu mẫu", "Tải CSV"], index=0)

if source == "Dữ liệu mẫu":
    demo_rows = st.sidebar.select_slider("Quy mô dữ liệu mẫu", options=[2000, 5000, 12000, 20000], value=12000)
    demo_seed = st.sidebar.number_input("Seed dữ liệu mẫu", min_value=1, max_value=9999, value=42, step=1)
    raw_input = generate_demo_data(int(demo_rows), int(demo_seed))
else:
    uploaded = st.sidebar.file_uploader("Tải file CSV", type=["csv"], help="Hệ thống tự nhận diện một số tên cột tiếng Việt/Anh như Giá trị → amount.")
    raw_input = None
    if uploaded is not None:
        try:
            raw_input = read_csv_flexible(uploaded.getvalue())
        except Exception as exc:
            st.sidebar.error(f"Không thể đọc CSV: {exc}")

st.sidebar.markdown("---")
contamination = st.sidebar.slider(
    "Contamination của Isolation Forest",
    min_value=0.01,
    max_value=0.10,
    value=0.035,
    step=0.005,
    help="Tỷ lệ bất thường kỳ vọng của model, không phải xác suất gian lận.",
)
n_estimators = st.sidebar.slider("Số cây (n_estimators)", 100, 600, 300, 50)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

if raw_input is None:
    st.sidebar.info("Chọn Dữ liệu mẫu hoặc tải CSV để bắt đầu.")
    st.stop()

with st.spinner("Đang kiểm tra dữ liệu và chuẩn hóa schema…"):
    data, quality = cached_prepare(raw_input)

if data is None:
    st.error("Không thể phân tích dataset vì thiếu cột bắt buộc.")
    st.code("\n".join(CANONICAL_COLUMNS))
    if quality.get("missing_columns"):
        st.error("Thiếu: " + ", ".join(quality["missing_columns"]))
    st.stop()

with st.spinner("AI đang xây dựng baseline hành vi và chạy Isolation Forest…"):
    result, model_info = cached_detection(data, float(contamination), int(n_estimators), int(seed))

# -----------------------------
# Global metrics & navigation
# -----------------------------
total = len(result)
anomaly_df = result[result["is_anomaly"]].copy()
normal_df = result[~result["is_anomaly"]].copy()
anomaly_count = len(anomaly_df)
anomaly_rate = anomaly_count / total * 100 if total else 0
high_count = int(result["risk_level"].isin(["Cao", "Nghiêm trọng"]).sum())
critical_count = int(result["risk_level"].eq("Nghiêm trọng").sum())
customers = int(result["customer_id"].nunique())
anomaly_value = float(anomaly_df["amount"].sum())

st.markdown(
    f"""
<div class="hero">
  <h1>🛡️ INVESTIGATION CENTER AI</h1>
  <p>Hệ thống phân tích giao dịch, xếp hạng Risk Score, giám sát hàng đợi rủi ro và điều tra theo giao dịch/khách hàng.</p>
</div>
""",
    unsafe_allow_html=True,
)

cols = st.columns(7)
metrics = [
    ("Giao dịch", f"{total:,}"),
    ("Bất thường", f"{anomaly_count:,}"),
    ("Tỷ lệ bất thường", percent(anomaly_rate)),
    ("Cao + nghiêm trọng", f"{high_count:,}"),
    ("Nghiêm trọng", f"{critical_count:,}"),
    ("KH bị ảnh hưởng", f"{result.loc[result['is_anomaly'], 'customer_id'].nunique():,}"),
    ("Giá trị bất thường", money(anomaly_value)),
]
for col, (label, value) in zip(cols, metrics):
    with col:
        kpi_block(label, value)

st.caption("⚠️ Isolation Forest phát hiện điểm dữ liệu bất thường. Risk Score là thứ hạng tương đối trong dataset hiện tại; không phải xác suất gian lận.")

pages = [
    "📊 Dashboard",
    "🚨 Giám sát rủi ro",
    "🔍 Điều tra giao dịch",
    "👤 Hồ sơ khách hàng 360°",
    "📈 Phân tích dữ liệu",
    "🤖 Trung tâm mô hình AI",
    "📋 Dữ liệu & Báo cáo",
    "📚 Phương pháp",
]
st.sidebar.caption("Chọn màn hình cần làm việc")
page = st.sidebar.selectbox("Điều hướng", pages, index=0, key="main_navigation")

st.sidebar.markdown("---")
st.sidebar.markdown("**Data Health**")
st.sidebar.caption(f"Hợp lệ: {quality['valid_rows']:,}/{quality['input_rows']:,} dòng")
if quality["dropped_rows"] == 0 and quality["duplicate_transaction_ids"] == 0:
    st.sidebar.success("Dataset sạch")
else:
    st.sidebar.warning(
        f"Dropped: {quality['dropped_rows']:,} • Duplicate IDs: {quality['duplicate_transaction_ids']:,}"
    )
st.sidebar.caption(f"Cờ mô hình: {model_info['flagged_anomalies']:,} ({model_info['flagged_rate']:.2f}%)")


# -----------------------------
# Dashboard
# -----------------------------
if page == pages[0]:
    st.markdown('<div class="section-title">Tổng quan điều hành</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Tập trung vào quy mô rủi ro, xu hướng và các giao dịch cần ưu tiên.</div>', unsafe_allow_html=True)

    left, right = st.columns([1.15, 1.85])
    with left:
        status_df = pd.DataFrame({"Trạng thái": ["Bình thường", "Bất thường"], "Số giao dịch": [len(normal_df), anomaly_count]})
        fig = px.pie(status_df, names="Trạng thái", values="Số giao dịch", hole=.58, title="Phân loại mô hình")
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), legend_title_text="")
        st.plotly_chart(fig, width="stretch")
    with right:
        daily = result.assign(day=result["timestamp"].dt.date).groupby("day").agg(
            total=("transaction_id", "count"),
            anomaly=("is_anomaly", "sum"),
            avg_risk=("risk_score", "mean"),
        ).reset_index()
        daily["anomaly_rate"] = daily["anomaly"] / daily["total"] * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["day"], y=daily["anomaly_rate"], mode="lines+markers", name="Tỷ lệ bất thường (%)"))
        fig.add_trace(go.Scatter(x=daily["day"], y=daily["avg_risk"], mode="lines", name="Điểm rủi ro trung bình", yaxis="y2"))
        fig.update_layout(
            title="Xu hướng bất thường theo ngày",
            yaxis=dict(title="Tỷ lệ bất thường (%)"),
            yaxis2=dict(title="Điểm rủi ro", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=10, r=10, t=70, b=10),
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Hàng đợi cần ưu tiên")
    top_cols = ["transaction_id", "customer_id", "amount", "risk_score", "risk_level", "priority", "category", "location", "reason"]
    top = result[top_cols].head(20).copy()
    top["amount"] = top["amount"].map(money)
    top["risk_score"] = top["risk_score"].map(lambda x: f"{x:.1f}")
    st.dataframe(top, width="stretch", hide_index=True, height=500)

    c1, c2 = st.columns(2)
    with c1:
        signal_summary = []
        for key, (_, label) in SIGNAL_WEIGHTS.items():
            if key == "amount_top_1pct":
                count = int(result["amount"].ge(result["amount"].quantile(.99)).sum())
            elif key == "amount_vs_customer":
                count = int(result["amount_vs_customer_prev_avg"].ge(3).sum())
            elif key == "new_device":
                count = int(result["new_device_for_customer"].sum())
            elif key == "new_location":
                count = int(result["new_location_for_customer"].sum())
            elif key == "night":
                count = int(result["is_night"].sum())
            elif key == "rapid":
                count = int(result["rapid_transaction"].sum())
            elif key == "new_merchant":
                count = int(result["new_merchant_for_customer"].sum())
            else:
                count = int(result["is_weekend"].sum())
            signal_summary.append({"Tín hiệu": label, "Số giao dịch": count})
        sig_df = pd.DataFrame(signal_summary).sort_values("Số giao dịch", ascending=False)
        fig = px.bar(sig_df.head(8), x="Số giao dịch", y="Tín hiệu", orientation="h", title="Tín hiệu ngữ cảnh phổ biến")
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        customer_risk = result[result["is_anomaly"]].groupby("customer_id").agg(
            anomalies=("transaction_id", "count"), anomaly_value=("amount", "sum"), max_risk=("risk_score", "max")
        ).reset_index().sort_values(["anomalies", "anomaly_value"], ascending=False).head(10)
        fig = px.bar(customer_risk, x="anomaly_value", y="customer_id", orientation="h", title="Khách hàng có giá trị bất thường lớn nhất")
        fig.update_layout(margin=dict(l=10, r=10, t=55, b=10), xaxis_title="Giá trị bất thường (₫)")
        st.plotly_chart(fig, width="stretch")

    with st.expander("Chỉ số hệ thống"):
        s1, s2, s3 = st.columns(3)
        s1.metric("Risk Score trung bình", fmt_score(result["risk_score"].mean()))
        s2.metric("Risk Score cao nhất", fmt_score(result["risk_score"].max()))
        s3.metric("Giá trị / anomaly TB", money(anomaly_df["amount"].mean() if anomaly_count else 0))


# -----------------------------
# Risk Monitoring
# -----------------------------
elif page == pages[1]:
    st.markdown('<div class="section-title">🚨 Giám sát rủi ro</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Lọc hàng đợi theo Risk Score và bối cảnh để tìm các giao dịch cần kiểm tra.</div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        min_score = st.slider("Risk Score tối thiểu", 0, 100, 65)
    with f2:
        selected_levels = st.multiselect("Mức rủi ro", ["Nghiêm trọng", "Cao", "Trung bình", "Thấp"], default=["Nghiêm trọng", "Cao"])
    with f3:
        selected_categories = st.multiselect("Danh mục", sorted(result["category"].unique()), default=[])
    with f4:
        selected_methods = st.multiselect("Thanh toán", sorted(result["payment_method"].unique()), default=[])

    f5, f6, f7, f8 = st.columns(4)
    with f5:
        selected_locations = st.multiselect("Địa điểm", sorted(result["location"].unique()), default=[])
    with f6:
        status_choice = st.multiselect("Trạng thái model", ["Bất thường", "Bình thường"], default=["Bất thường"])
    with f7:
        customer_query = st.text_input("Tìm Mã khách hàng", placeholder="Nhập một phần mã…")
    with f8:
        start_date = st.date_input("Từ ngày", value=result["timestamp"].dt.date.min())
        end_date = st.date_input("Đến ngày", value=result["timestamp"].dt.date.max())

    filtered = result[result["risk_score"] >= min_score].copy()
    if selected_levels:
        filtered = filtered[filtered["risk_level"].astype(str).isin(selected_levels)]
    if selected_categories:
        filtered = filtered[filtered["category"].isin(selected_categories)]
    if selected_methods:
        filtered = filtered[filtered["payment_method"].isin(selected_methods)]
    if selected_locations:
        filtered = filtered[filtered["location"].isin(selected_locations)]
    if status_choice:
        filtered = filtered[filtered["status"].isin(status_choice)]
    if customer_query:
        filtered = filtered[filtered["customer_id"].astype(str).str.contains(customer_query.strip(), case=False, na=False)]
    filtered = filtered[filtered["timestamp"].dt.date.between(start_date, end_date)]

    k = st.columns(4)
    k[0].metric("Kết quả", f"{len(filtered):,}")
    k[1].metric("Tổng giá trị", money(filtered["amount"].sum()))
    k[2].metric("Risk TB", fmt_score(filtered["risk_score"].mean()) if len(filtered) else "0/100")
    k[3].metric("Anomaly trong queue", f"{int(filtered['is_anomaly'].sum()):,}")

    display_cols = ["transaction_id", "customer_id", "timestamp", "amount", "risk_score", "risk_level", "priority", "category", "location", "payment_method", "signal_count", "reason"]
    table = filtered[display_cols].head(500).copy()
    table["timestamp"] = table["timestamp"].dt.strftime("%Y-%m-%d %H:00")
    table["amount"] = table["amount"].map(money)
    table["risk_score"] = table["risk_score"].map(lambda x: f"{x:.1f}")
    table["customer_id"] = table["customer_id"].astype(str)
    st.dataframe(table, width="stretch", height=600, hide_index=True)

    st.download_button(
        "⬇️ Xuất queue hiện tại",
        filtered.to_csv(index=False).encode("utf-8-sig"),
        "risk_queue.csv",
        "text/csv",
    )

    st.markdown("### Phân bố queue")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(filtered, x="risk_score", color="risk_level", nbins=30, title="Risk Score của queue")
        st.plotly_chart(fig, width="stretch")
    with c2:
        by_cat = filtered.groupby("category").agg(anomalies=("is_anomaly", "sum"), value=("amount", "sum")).reset_index().sort_values("anomalies", ascending=False)
        fig = px.bar(by_cat, x="category", y="anomalies", title="Anomaly theo danh mục")
        st.plotly_chart(fig, width="stretch")


# -----------------------------
# Transaction Investigation
# -----------------------------
elif page == pages[2]:
    st.markdown('<div class="section-title">🔍 Điều tra giao dịch</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Đi từ một giao dịch cụ thể đến lý do bị gắn cờ và lịch sử hành vi của khách hàng.</div>', unsafe_allow_html=True)

    query = st.text_input("Tìm Transaction ID", placeholder="Ví dụ: TX100123")
    candidate = result
    if query.strip():
        candidate = result[result["transaction_id"].astype(str).str.contains(query.strip(), case=False, na=False)]
    candidate = candidate.head(1000)
    if candidate.empty:
        st.warning("Không tìm thấy giao dịch phù hợp.")
        st.stop()

    default_index = 0
    selected_tx = st.selectbox("Chọn giao dịch", candidate["transaction_id"].tolist(), index=default_index)
    row = result[result["transaction_id"].eq(selected_tx)].iloc[0]

    left, right = st.columns([1.0, 1.7])
    with left:
        st.plotly_chart(make_risk_gauge(row["risk_score"]), width="stretch")
        st.metric("Context signal score", f"{row['context_signal_score']:.1f}/100")
    with right:
        a, b, c, d = st.columns(4)
        a.metric("Số tiền", money(row["amount"]))
        b.metric("Risk Score", fmt_score(row["risk_score"]))
        c.metric("Mức rủi ro", safe_value(row["risk_level"]))
        d.metric("Model", safe_value(row["status"]))
        st.info(
            "**Why this transaction stands out:** " + safe_value(row["reason"])
        )
        st.caption("Context signals là lớp giải thích quy tắc để hỗ trợ điều tra; chúng không thay thế anomaly score của model.")

    st.markdown("### Hồ sơ giao dịch")
    st.dataframe(details_table(row), width="stretch", hide_index=True)

    st.markdown("### Lịch sử gần nhất của khách hàng")
    customer_hist = result[result["customer_id"].eq(row["customer_id"])].sort_values("timestamp", ascending=False).head(15).copy()
    hist_cols = ["transaction_id", "timestamp", "amount", "risk_score", "risk_level", "status", "location", "device", "category"]
    hist = customer_hist[hist_cols].copy()
    hist["timestamp"] = hist["timestamp"].dt.strftime("%Y-%m-%d %H:00")
    hist["amount"] = hist["amount"].map(money)
    hist["risk_score"] = hist["risk_score"].map(lambda x: f"{x:.1f}")
    st.dataframe(hist, width="stretch", hide_index=True)

    st.markdown("### So sánh với baseline khách hàng")
    baseline_cols = st.columns(4)
    baseline_cols[0].metric("Baseline amount", money(row["customer_prev_avg_amount"]))
    baseline_cols[1].metric("Current amount", money(row["amount"]))
    baseline_cols[2].metric("Deviation", f"{row['amount_vs_customer_prev_avg']:.2f}x")
    baseline_cols[3].metric("Prev transactions", f"{int(row['customer_prev_count']):,}")

    amount_sample = result[["amount"]].copy()
    fig = px.histogram(amount_sample, x="amount", nbins=60, title="Vị trí của giao dịch trong phân bố số tiền")
    fig.add_vline(x=float(row["amount"]), line_dash="dash", annotation_text="Giao dịch đang chọn")
    fig.update_xaxes(title="Số tiền (₫)")
    st.plotly_chart(fig, width="stretch")


# -----------------------------
# Customer Intelligence
# -----------------------------
elif page == pages[3]:
    st.markdown('<div class="section-title">👤 Customer Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Xem baseline, quy mô giao dịch, anomaly history và các thay đổi về thiết bị/địa điểm của từng khách hàng.</div>', unsafe_allow_html=True)

    customer_summary = result.groupby("customer_id").agg(
        transactions=("transaction_id", "count"),
        total_spend=("amount", "sum"),
        avg_spend=("amount", "mean"),
        max_risk=("risk_score", "max"),
        avg_risk=("risk_score", "mean"),
        anomalies=("is_anomaly", "sum"),
        anomaly_value=("amount", lambda s: float(s[result.loc[s.index, "is_anomaly"]].sum())),
        known_devices=("device", "nunique"),
        locations=("location", "nunique"),
    ).reset_index()
    customer_summary["anomaly_rate"] = customer_summary["anomalies"] / customer_summary["transactions"] * 100
    customer_summary = customer_summary.sort_values(["anomalies", "max_risk", "anomaly_value"], ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Khách hàng", f"{len(customer_summary):,}")
    c2.metric("Có anomaly", f"{int((customer_summary['anomalies'] > 0).sum()):,}")
    c3.metric("Anomaly rate KH TB", percent(customer_summary["anomaly_rate"].mean()))
    c4.metric("Giá trị anomaly", money(customer_summary["anomaly_value"].sum()))

    cust_query = st.text_input("Tìm Mã khách hàng", placeholder="Ví dụ: 1042")
    candidates = customer_summary
    if cust_query.strip():
        candidates = candidates[candidates["customer_id"].astype(str).str.contains(cust_query.strip(), case=False, na=False)]
    if candidates.empty:
        st.warning("Không tìm thấy khách hàng.")
        st.stop()
    selected_customer = st.selectbox("Khách hàng", candidates["customer_id"].astype(str).tolist())
    selected = result[result["customer_id"].astype(str).eq(str(selected_customer))].copy()

    agg = selected.iloc[0]
    p = st.columns(6)
    p[0].metric("Giao dịch", f"{len(selected):,}")
    p[1].metric("Tổng chi", money(selected["amount"].sum()))
    p[2].metric("Chi tiêu TB", money(selected["amount"].mean()))
    p[3].metric("Anomaly", f"{int(selected['is_anomaly'].sum()):,}")
    p[4].metric("Risk max", fmt_score(selected["risk_score"].max()))
    p[5].metric("Thiết bị", f"{selected['device'].nunique():,}")

    c1, c2 = st.columns(2)
    with c1:
        risk_ts = selected.sort_values("timestamp")
        fig = px.line(risk_ts, x="timestamp", y="risk_score", markers=True, title="Risk Score theo lịch sử khách hàng")
        st.plotly_chart(fig, width="stretch")
    with c2:
        spend_cat = selected.groupby("category")["amount"].sum().reset_index().sort_values("amount", ascending=False)
        fig = px.bar(spend_cat, x="category", y="amount", title="Chi tiêu theo danh mục")
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Giao dịch gần đây")
    recent = selected.sort_values("timestamp", ascending=False).head(30).copy()
    recent["timestamp"] = recent["timestamp"].dt.strftime("%Y-%m-%d %H:00")
    recent["amount"] = recent["amount"].map(money)
    recent["risk_score"] = recent["risk_score"].map(lambda x: f"{x:.1f}")
    show_cols = ["transaction_id", "timestamp", "amount", "risk_score", "risk_level", "status", "category", "location", "device", "reason"]
    st.dataframe(recent[show_cols], width="stretch", height=500, hide_index=True)

    st.markdown("### So sánh thiết bị & địa điểm")
    c1, c2 = st.columns(2)
    with c1:
        devices = selected.groupby("device").agg(transactions=("transaction_id", "count"), anomalies=("is_anomaly", "sum")).reset_index()
        fig = px.bar(devices, x="device", y=["transactions", "anomalies"], barmode="group", title="Thiết bị")
        st.plotly_chart(fig, width="stretch")
    with c2:
        locations = selected.groupby("location").agg(transactions=("transaction_id", "count"), anomalies=("is_anomaly", "sum")).reset_index()
        fig = px.bar(locations, x="location", y=["transactions", "anomalies"], barmode="group", title="Địa điểm")
        st.plotly_chart(fig, width="stretch")


# -----------------------------
# Data Analysis
# -----------------------------
elif page == pages[4]:
    st.markdown('<div class="section-title">📈 Phân tích dữ liệu</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Đào sâu vào thời gian, số tiền, danh mục, thanh toán và mức Risk Score.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        hourly = result.groupby(["hour", "status"]).size().reset_index(name="count")
        fig = px.bar(hourly, x="hour", y="count", color="status", barmode="group", title="Giao dịch theo giờ")
        st.plotly_chart(fig, width="stretch")
    with c2:
        pay = result.groupby(["payment_method", "status"]).size().reset_index(name="count")
        fig = px.bar(pay, x="payment_method", y="count", color="status", barmode="group", title="Phương thức thanh toán")
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        cat = result.groupby(["category", "status"]).size().reset_index(name="count")
        fig = px.bar(cat, x="category", y="count", color="status", barmode="group", title="Danh mục")
        st.plotly_chart(fig, width="stretch")
    with c2:
        sample = result.sample(min(6000, len(result)), random_state=42)
        fig = px.scatter(sample, x="amount", y="risk_score", color="status", hover_data=["transaction_id", "customer_id", "category", "location"], title="Giá trị giao dịch và điểm rủi ro")
        fig.update_xaxes(type="log", title="Số tiền (log scale)")
        st.plotly_chart(fig, width="stretch")

    # Heatmap of anomaly rate by weekday/hour.
    heat = result.groupby(["weekday", "hour"]).agg(
        transactions=("transaction_id", "count"), anomalies=("is_anomaly", "sum")
    ).reset_index()
    heat["anomaly_rate"] = heat["anomalies"] / heat["transactions"] * 100
    pivot = heat.pivot(index="weekday", columns="hour", values="anomaly_rate").reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    fig = px.imshow(pivot, aspect="auto", labels={"color": "Tỷ lệ bất thường (%)"}, title="Bản đồ nhiệt: tỷ lệ bất thường theo thứ × giờ")
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Thống kê mô tả")
    desc = result[["amount", "risk_score", "customer_prev_count", "amount_vs_customer_prev_avg", "context_signal_score"]].describe().T.reset_index(names="feature")
    st.dataframe(desc, width="stretch", hide_index=True)


# -----------------------------
# Model Center
# -----------------------------
elif page == pages[5]:
    st.markdown('<div class="section-title">🤖 Model Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Minh bạch về model, feature space, threshold và giới hạn của anomaly detection.</div>', unsafe_allow_html=True)

    c = st.columns(6)
    c[0].metric("Thuật toán", model_info["algorithm"])
    c[1].metric("Số cây", f"{model_info['n_estimators']:,}")
    c[2].metric("Tỷ lệ bất thường kỳ vọng", percent(model_info["contamination"] * 100))
    c[3].metric("Số dòng huấn luyện", f"{model_info['training_rows']:,}")
    c[4].metric("Đặc trưng gốc", f"{model_info['raw_feature_count']:,}")
    c[5].metric("Đặc trưng sau mã hóa", f"{model_info['transformed_feature_count']:,}" if model_info["transformed_feature_count"] else "—")

    st.info(
        "Isolation Forest là mô hình không giám sát. Model tìm các quan sát có tính chất dễ bị cô lập hơn trong không gian đặc trưng. Không có ground-truth fraud label trong dataset mẫu nên hệ thống không trình bày Accuracy/Precision/Recall như thể chúng đã được kiểm chứng."
    )

    st.markdown("### Nhóm đặc trưng")
    feature_info = pd.DataFrame({
        "Nhóm": ["Giá trị giao dịch", "Baseline khách hàng", "Thời gian", "Tính mới", "Ngữ cảnh phân loại"],
        "Features": [
            ", ".join(["amount_log", "amount_percentile", "global_amount_z"]),
            ", ".join(["customer_prev_count", "customer_prev_avg_amount", "amount_vs_customer_prev_avg", "amount_customer_z"]),
            ", ".join(["hour_sin", "hour_cos", "customer_hour_deviation", "minutes_since_previous", "same_day_customer_count_before"]),
            ", ".join(["new_device_for_customer", "new_location_for_customer", "new_merchant_for_customer"]),
            ", ".join(CATEGORICAL_FEATURES),
        ],
        "Vai trò": [
            "Mô tả quy mô giao dịch và vị trí trong toàn bộ phân bố.",
            "So sánh với lịch sử của chính khách hàng, tránh lấy giao dịch hiện tại làm baseline.",
            "Nhận diện thay đổi hành vi theo thời gian và tốc độ giao dịch.",
            "Đo mức độ mới của device/location/merchant đối với khách hàng.",
            "Mã hóa ngữ cảnh danh mục, người bán, vị trí và phương thức thanh toán.",
        ],
    })
    st.dataframe(feature_info, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        score_dist = px.histogram(result, x="risk_score", color="risk_level", nbins=40, title="Phân bố Risk Score")
        st.plotly_chart(score_dist, width="stretch")
    with c2:
        flag_dist = result["risk_level"].value_counts().reindex(["Nghiêm trọng", "Cao", "Trung bình", "Thấp"]).fillna(0).reset_index()
        flag_dist.columns = ["risk_level", "count"]
        fig = px.bar(flag_dist, x="risk_level", y="count", title="Phân bố Risk Level")
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Ngưỡng diễn giải")
    score_table = pd.DataFrame({
        "Risk Score": ["0–49.9", "50–79.9", "80–94.9", "95–100"],
        "Mức rủi ro": ["Thấp", "Trung bình", "Cao", "Nghiêm trọng"],
        "Ý nghĩa vận hành": [
            "Theo dõi bình thường.",
            "Xem thêm bối cảnh nếu có tín hiệu bổ sung.",
            "Ưu tiên xem xét.",
            "Đưa lên đầu hàng đợi điều tra.",
        ],
    })
    st.dataframe(score_table, width="stretch", hide_index=True)

    st.markdown("### Chẩn đoán mô hình")
    diagnostics = pd.DataFrame({
        "Chỉ số": ["Giao dịch bất thường do mô hình đánh dấu", "Tỷ lệ bị đánh dấu", "Ngưỡng quyết định", "Trung vị điểm bất thường gốc", "Điểm bất thường gốc P95", "Điểm bất thường gốc P99"],
        "Giá trị": [
            f"{model_info['flagged_anomalies']:,}",
            f"{model_info['flagged_rate']:.2f}%",
            f"{model_info['decision_threshold']:.6f}",
            f"{model_info['anomaly_strength_quantiles'][0]:.6f}",
            f"{model_info['anomaly_strength_quantiles'][2]:.6f}",
            f"{model_info['anomaly_strength_quantiles'][3]:.6f}",
        ],
    })
    st.dataframe(diagnostics, width="stretch", hide_index=True)


# -----------------------------
# Data & Reports
# -----------------------------
elif page == pages[6]:
    st.markdown('<div class="section-title">📋 Dữ liệu & Báo cáo</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Kiểm tra chất lượng input, xem schema sau chuẩn hóa và xuất kết quả phục vụ báo cáo.</div>', unsafe_allow_html=True)

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Số dòng đầu vào", f"{quality['input_rows']:,}")
    q2.metric("Số dòng hợp lệ", f"{quality['valid_rows']:,}")
    q3.metric("Số dòng bị loại", f"{quality['dropped_rows']:,}")
    q4.metric("ID giao dịch trùng", f"{quality['duplicate_transaction_ids']:,}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Giá trị không hợp lệ", f"{quality['invalid_amounts']:,}")
    c2.metric("Giá trị âm", f"{quality['negative_amounts']:,}")
    c3.metric("Ngày/giờ không hợp lệ", f"{quality['invalid_dates']:,}")

    if quality.get("renamed_columns"):
        st.success("Đã chuẩn hóa tên cột: " + ", ".join(f"{a} → {b}" for a, b in quality["renamed_columns"].items()))

    st.markdown("### Schema hiện tại")
    schema = pd.DataFrame({
        "Cột": data.columns,
        "Kiểu dữ liệu": [str(data[c].dtype) for c in data.columns],
        "Giá trị trống": [int(data[c].isna().sum()) for c in data.columns],
        "Giá trị duy nhất": [int(data[c].nunique(dropna=True)) for c in data.columns],
    })
    st.dataframe(schema, width="stretch", hide_index=True)

    st.markdown("### Preview dữ liệu đã chuẩn hóa")
    preview = data.head(50).copy()
    preview["timestamp"] = preview["timestamp"].dt.strftime("%Y-%m-%d %H:00")
    preview["amount"] = preview["amount"].map(money)
    st.dataframe(preview, width="stretch", height=420, hide_index=True)

    st.markdown("### Export")
    col1, col2, col3 = st.columns(3)
    col1.download_button(
        "⬇️ Kết quả đầy đủ CSV",
        result.to_csv(index=False).encode("utf-8-sig"),
        "transaction_anomaly_results.csv",
        "text/csv",
    )
    col2.download_button(
        "⬇️ Anomaly queue CSV",
        anomaly_df.to_csv(index=False).encode("utf-8-sig"),
        "anomaly_queue.csv",
        "text/csv",
    )
    html = build_html_report(result, quality, model_info)
    col3.download_button(
        "⬇️ Báo cáo HTML",
        html.encode("utf-8"),
        "transaction_anomaly_report.html",
        "text/html",
    )


# -----------------------------
# Methodology
# -----------------------------
else:
    st.markdown('<div class="section-title">📚 Phương pháp & hướng dẫn</div>', unsafe_allow_html=True)

    st.markdown("""
### 1. Bài toán
Hệ thống giải quyết bài toán **anomaly detection**: tìm các giao dịch có đặc điểm khác biệt đáng kể so với phần còn lại của dữ liệu.

### 2. Pipeline
**Raw CSV / dữ liệu mẫu → Schema validation → Data quality → Historical feature engineering → Isolation Forest → Risk Score → Context signals → Monitoring → Investigation → Reporting.**

### 3. Historical baseline
Baseline khách hàng được xây dựng từ các giao dịch **trước** giao dịch đang phân tích, để hạn chế việc lấy chính giao dịch hiện tại làm chuẩn so sánh.

### 4. Isolation Forest
Model không cần nhãn fraud/legitimate. Nó tìm các quan sát có cấu trúc khác biệt trong không gian đặc trưng. `contamination` điều khiển tỷ lệ bất thường mà model kỳ vọng.

### 5. Risk Score
Risk Score 0–100 là **thứ hạng tương đối của anomaly score trong dataset hiện tại**. Điểm 95 không có nghĩa là xác suất gian lận 95%.

### 6. Context signals
Các signal như số tiền tăng mạnh so với baseline, thiết bị mới, địa điểm mới, giao dịch ban đêm, tốc độ giao dịch… được dùng để giải thích bối cảnh cho người điều tra. Đây là lớp giải thích bổ sung, không thay thế model score.

### 7. Không dùng Accuracy giả
Nếu dataset không có ground-truth fraud label, ứng dụng không tự tạo Accuracy/Precision/Recall. Khi có nhãn thật, có thể bổ sung đánh giá supervised/benchmarking.

### 8. Cách demo đề tài
1. Chọn dữ liệu mẫu.
2. Giải thích contamination và Data Health.
3. Mở Dashboard để nói về quy mô và xu hướng.
4. Vào Giám sát rủi ro để lọc Risk Score cao.
5. Chọn một transaction trong Điều tra giao dịch.
6. Giải thích baseline khách hàng + context signals.
7. Sang Customer Intelligence để trình bày lịch sử.
8. Sang Model Center để giải thích feature space và diagnostics.
9. Xuất báo cáo HTML/CSV.
""")

st.markdown("---")
st.caption("AI Transaction Anomaly Detection • Professional academic/demo build • Không sử dụng kết quả như bằng chứng duy nhất để kết luận gian lận.")
