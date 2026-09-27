# Changelog — Professional Build

## Upgrade

### Nền dữ liệu
- Chuẩn hóa schema transaction/customer/date/hour/amount/category/merchant/location/payment_method/device.
- Tự nhận diện một số tên cột tiếng Việt/Anh, gồm `Giá trị` → `amount`.
- Parse số tiền có ký hiệu tiền tệ và dấu phân cách phổ biến.
- Báo cáo data quality: invalid rows, duplicate transaction IDs, invalid/negative amounts.

### Feature engineering
- Historical customer baseline dùng các giao dịch trước đó.
- Amount deviation, customer z-like score, transaction velocity.
- Device/location/merchant novelty.
- Hour cyclic features, weekend/night features.

### AI / Risk
- Isolation Forest được tách khỏi UI.
- Risk Score 0–100 là relative anomaly rank trong dataset.
- Risk levels: Thấp / Trung bình / Cao / Nghiêm trọng.
- Context signals được tách riêng khỏi model score để giải thích.
- Không tự tạo Accuracy/Precision/Recall khi không có ground-truth labels.

### UX / Investigation
- Sidebar navigation thay cho việc render đồng thời mọi tab.
- Dashboard điều hành.
- Risk Monitoring với nhiều bộ lọc.
- Transaction Investigation + customer history + baseline comparison.
- Customer Intelligence.
- Model Center và diagnostics.
- Data & Reports với CSV + HTML export.

### Technical fixes
- Loại bỏ `use_container_width`.
- Dùng `width="stretch"` cho bảng/biểu đồ cần full width.
- Fix lỗi Arrow/PyArrow do bảng detail dùng cột `Giá trị` chứa mixed Python types bằng cách ép toàn bộ giá trị hiển thị thành string.
- Dockerfile copy cả thư mục `src`.
