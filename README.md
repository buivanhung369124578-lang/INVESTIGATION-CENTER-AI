# AI Transaction Anomaly Detection — Professional Build

Ứng dụng Streamlit phát hiện giao dịch bất thường bằng **Isolation Forest**, có thêm historical customer baseline, Risk Score 0–100, context signals, risk monitoring, transaction investigation, customer intelligence, data quality và reporting.

## Chạy local
```bash
python -m venv test_env
test_env\\Scripts\\activate
python -m pip install -r requirements.txt
streamlit run app.py
```
Mở `http://localhost:8501`.

## Kiến trúc
```text
app.py
src/
  data_pipeline.py     # schema mapping, CSV parsing, validation, demo data, feature engineering
  model_engine.py      # Isolation Forest, risk score, context signals, diagnostics
  reporting.py         # CSV/HTML report helpers
```

## Các màn hình
- Dashboard
- Giám sát rủi ro
- Điều tra giao dịch
- Customer Intelligence
- Phân tích dữ liệu
- Model Center
- Dữ liệu & Báo cáo
- Phương pháp

## Dữ liệu CSV
Schema chuẩn:
```text
transaction_id, customer_id, date, hour, amount,
category, merchant, location, payment_method, device
```

Một số alias tiếng Việt được nhận diện tự động, ví dụ `Giá trị` → `amount`.

## Lưu ý học thuật
Isolation Forest phát hiện **bất thường**, không chứng minh giao dịch là gian lận. Risk Score là thứ hạng tương đối trong dataset hiện tại, không phải xác suất gian lận.

## Cloud Run
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/transaction-anomaly
gcloud run deploy transaction-anomaly \\
  --image gcr.io/PROJECT_ID/transaction-anomaly \\
  --platform managed \\
  --region asia-southeast1 \\
  --allow-unauthenticated
```
