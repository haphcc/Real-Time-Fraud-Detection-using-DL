# Real-Time-Fraud-Detection-using-DL

> **[ITDE - NCKH 2025]** Real-Time Payment Fraud Detection with Temporal Graphs  
> Nghiên cứu về việc phát hiện giao dịch gian lận theo thời gian thực bằng biểu đồ thời gian

---

## 📌 Giới thiệu (Overview)

Dự án nghiên cứu và so sánh các phương pháp **Deep Learning** và **Machine Learning** trong bài toán phát hiện giao dịch gian lận tài chính theo thời gian thực. Hệ thống sử dụng nhiều mô hình khác nhau, từ mạng nơ-ron đơn giản (MLP) đến các kiến trúc nâng cao như GRU, Random Forest kết hợp SMOTE, và kỹ thuật phát hiện ngoài phân phối (ODIN).

This project researches and compares multiple **Deep Learning** and **Machine Learning** approaches for real-time financial fraud detection. It explores models ranging from simple neural networks (MLP) to advanced architectures including GRU, Random Forest with SMOTE balancing, and Out-of-Distribution detection (ODIN).

---

## 🗂️ Cấu trúc dự án (Project Structure)

```
Real-Time-Fraud-Detection-using-DL/
├── HTGNN.ipynb       # MLP-based fraud detection (scikit-learn)
├── TGCN1.ipynb       # Random Forest + SMOTE for class imbalance
├── TestGRU.ipynb     # GRU recurrent neural network (PyTorch)
├── TestODIN.ipynb    # ODIN out-of-distribution detection (PyTorch)
└── README.md
```

---

## 🔬 Các mô hình nghiên cứu (Models)

| Notebook | Mô hình | Framework | Đặc điểm nổi bật |
|----------|---------|-----------|------------------|
| `HTGNN.ipynb` | MLP (Multi-Layer Perceptron) | scikit-learn | Mạng nơ-ron đầy đủ kết nối cơ bản |
| `TGCN1.ipynb` | Random Forest + SMOTE | scikit-learn | Cân bằng lớp với SMOTE, đánh giá ROC/AUC |
| `TestGRU.ipynb` | GRU (Gated Recurrent Unit) | PyTorch | Mô hình tuần tự trên chuỗi giao dịch |
| `TestODIN.ipynb` | ODIN Neural Network | PyTorch | Phát hiện ngoài phân phối, nhiễu đầu vào & temperature scaling |

### Chi tiết từng mô hình

#### 1. HTGNN — MLP Classifier
- Sử dụng `MLPClassifier` của scikit-learn để phân loại nhị phân giao dịch gian lận.
- Trích xuất đặc trưng thời gian từ timestamp (giờ, ngày trong tuần, tháng).
- Chuẩn hoá dữ liệu với `StandardScaler`.
- Đánh giá: Accuracy, Precision, Recall, F1-Score.

#### 2. TGCN1 — Random Forest + SMOTE
- Sử dụng `RandomForestClassifier` với `class_weight='balanced'`.
- Áp dụng **SMOTE** (Synthetic Minority Over-sampling Technique) để xử lý mất cân bằng lớp.
- Mã hoá đặc trưng phân loại bằng `OneHotEncoder`.
- So sánh kết quả có và không có SMOTE.
- Đánh giá: Confusion Matrix, Accuracy, Precision, Recall, F1-Score, ROC/AUC.

#### 3. TestGRU — Gated Recurrent Unit
- Xây dựng chuỗi giao dịch tuần tự (sequence length = 10) sắp xếp theo thời gian.
- Huấn luyện mô hình GRU với PyTorch để phát hiện mẫu gian lận theo chuỗi.
- Đánh giá: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.

#### 4. TestODIN — Out-of-Distribution Detection
- Áp dụng kỹ thuật **ODIN** với perturbation đầu vào và temperature scaling.
- Sử dụng `WeightedRandomSampler` để xử lý mất cân bằng lớp.
- Huấn luyện với `BCEWithLogitsLoss`.
- Đánh giá: ROC Curve, AUC Score, Confusion Matrix, Classification Report.

---

## 📊 Tập dữ liệu (Dataset)

- **Tên file:** `financial_fraud_detection_dataset.csv`
- **Số lượng mẫu:** ~251,898 giao dịch
- **Các đặc trưng chính:** `transaction_id`, `amount`, `timestamp`, `sender_account`, `receiver_account`, `transaction_type`, `location`, `device_type`, `ip_address`, `device_hash`, `time_since_last_transaction`, `is_fraud`
- **Nhãn:** `is_fraud` (0 = hợp lệ, 1 = gian lận)

> ⚠️ Tập dữ liệu **không** được đính kèm trong repository (đã được đưa vào `.gitignore`). Vui lòng chuẩn bị dữ liệu theo cấu trúc trên trước khi chạy notebook.

---

## ⚙️ Yêu cầu cài đặt (Requirements)

```bash
pip install numpy pandas scikit-learn imbalanced-learn torch matplotlib seaborn
```

### Các thư viện chính:
- `numpy`, `pandas` — xử lý và phân tích dữ liệu
- `scikit-learn` — MLP, Random Forest, preprocessing, metrics
- `imbalanced-learn` — SMOTE
- `torch` (PyTorch) — GRU, ODIN
- `matplotlib`, `seaborn` — trực quan hoá

---

## 🚀 Hướng dẫn chạy (Usage)

1. Clone repository:
   ```bash
   git clone https://github.com/haphcc/Real-Time-Fraud-Detection-using-DL.git
   cd Real-Time-Fraud-Detection-using-DL
   ```

2. Cài đặt thư viện:
   ```bash
   pip install numpy pandas scikit-learn imbalanced-learn torch matplotlib seaborn
   ```

3. Đặt file dữ liệu `financial_fraud_detection_dataset.csv` vào thư mục gốc của dự án.

4. Mở và chạy notebook tương ứng:
   ```bash
   jupyter notebook
   ```

---

## 📈 Kết quả & So sánh (Results)

Các mô hình được đánh giá trên các tiêu chí:

| Tiêu chí | Mô tả |
|----------|-------|
| **Accuracy** | Tỷ lệ dự đoán đúng trên toàn bộ tập kiểm tra |
| **Precision** | Độ chính xác trong dự đoán giao dịch gian lận |
| **Recall** | Khả năng phát hiện đúng giao dịch gian lận |
| **F1-Score** | Trung bình điều hoà của Precision và Recall |
| **AUC** | Diện tích dưới đường ROC (chỉ áp dụng cho TGCN1 và ODIN) |

---

## 👥 Thành viên nhóm (Team)

Dự án thuộc chương trình **Nghiên cứu Khoa học (NCKH) 2025** — ITDE.

---

## 📄 Giấy phép (License)

Dự án được phát triển cho mục đích nghiên cứu học thuật (academic research).
