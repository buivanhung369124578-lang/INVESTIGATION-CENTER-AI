from __future__ import annotations

from html import escape

import pandas as pd


def money(value: float) -> str:
    try:
        return f"{float(value):,.0f} ₫"
    except (TypeError, ValueError):
        return "—"


def percent(value: float) -> str:
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "—"


def safe_str(value) -> str:
    return "—" if pd.isna(value) else str(value)


def build_html_report(result: pd.DataFrame, quality: dict, model_info: dict) -> str:
    total = len(result)
    anomaly = int(result["is_anomaly"].sum())
    anomaly_rate = anomaly / total * 100 if total else 0
    anomaly_value = result.loc[result["is_anomaly"], "amount"].sum()
    high = int(result["risk_level"].isin(["Cao", "Nghiêm trọng"]).sum())

    top = result.head(15)[
        ["transaction_id", "customer_id", "amount", "risk_score", "risk_level", "category", "location", "reason"]
    ].copy()
    top["amount"] = top["amount"].map(money)
    top["risk_score"] = top["risk_score"].map(lambda x: f"{x:.1f}")
    top_html = top.to_html(index=False, classes="table", border=0, escape=True)

    quality_rows = "".join(
        f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>"
        for k, v in [
            ("Input rows", quality.get("input_rows", 0)),
            ("Valid rows", quality.get("valid_rows", 0)),
            ("Dropped rows", quality.get("dropped_rows", 0)),
            ("Duplicate transaction IDs", quality.get("duplicate_transaction_ids", 0)),
            ("Invalid dates/hours", quality.get("invalid_dates", 0)),
            ("Invalid amounts", quality.get("invalid_amounts", 0)),
            ("Negative amounts", quality.get("negative_amounts", 0)),
        ]
    )

    return f"""<!doctype html>
<html lang='vi'>
<head>
<meta charset='utf-8'>
<title>AI Transaction Anomaly Detection Report</title>
<style>
body{{font-family:Arial,Helvetica,sans-serif;margin:36px;color:#172033;background:#f8fafc}}
.container{{max-width:1180px;margin:auto;background:white;padding:32px;border:1px solid #e5e7eb;border-radius:18px}}
h1{{margin:0 0 6px}} h2{{margin-top:28px}} p{{color:#56627a}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card{{border:1px solid #e5e7eb;border-radius:12px;padding:16px;background:#fff}}
.label{{font-size:12px;color:#6b7280}} .value{{font-size:24px;font-weight:700;margin-top:8px}}
.table{{border-collapse:collapse;width:100%;font-size:13px}} .table th,.table td{{border:1px solid #e5e7eb;padding:8px;text-align:left;vertical-align:top}} .table th{{background:#f1f5f9}}
.note{{padding:12px 14px;border-radius:10px;background:#fff7ed;border:1px solid #fed7aa}}
</style>
</head>
<body><div class='container'>
<h1>AI Transaction Anomaly Detection</h1>
<p>Báo cáo phân tích phát hiện bất thường được sinh từ tập dữ liệu hiện tại.</p>
<div class='grid'>
<div class='card'><div class='label'>Tổng giao dịch</div><div class='value'>{total:,}</div></div>
<div class='card'><div class='label'>Bất thường</div><div class='value'>{anomaly:,}</div></div>
<div class='card'><div class='label'>Tỷ lệ bất thường</div><div class='value'>{anomaly_rate:.2f}%</div></div>
<div class='card'><div class='label'>Cao/Nghiêm trọng</div><div class='value'>{high:,}</div></div>
</div>
<h2>Giá trị & mô hình</h2>
<p><b>Giá trị bất thường:</b> {money(anomaly_value)} &nbsp;|&nbsp; <b>Model:</b> {escape(str(model_info.get('algorithm')))} &nbsp;|&nbsp; <b>Contamination:</b> {float(model_info.get('contamination',0))*100:.1f}% &nbsp;|&nbsp; <b>Trees:</b> {int(model_info.get('n_estimators',0))}</p>
<h2>Chất lượng dữ liệu</h2><table class='table'><tbody>{quality_rows}</tbody></table>
<h2>Top giao dịch theo Risk Score</h2>{top_html}
<p class='note'><b>Lưu ý:</b> Isolation Forest phát hiện điểm dữ liệu bất thường. Risk Score là thứ hạng tương đối trong tập dữ liệu hiện tại, không phải xác suất gian lận và không thay thế quy trình điều tra thực tế.</p>
</div></body></html>"""
