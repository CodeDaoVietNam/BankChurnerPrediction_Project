# Bank Customer Churn Prediction (Implemented from Scratch with NumPy)

## 1. Giới thiệu (Introduction)

### Mô tả bài toán
Trong lĩnh vực ngân hàng, "Churn" (rời bỏ) là hiện tượng khách hàng ngưng sử dụng dịch vụ. Dự án này tập trung vào việc dự đoán những khách hàng có nguy cơ đóng tài khoản thẻ tín dụng dựa trên dữ liệu hành vi và nhân khẩu học của họ.

### Động lực và Ứng dụng thực tế
* **Động lực:** Chi phí để tìm kiếm một khách hàng mới thường cao gấp 5-25 lần so với việc giữ chân khách hàng hiện tại.
* **Ứng dụng:** Giúp ngân hàng chủ động nhận diện khách hàng rủi ro cao để đưa ra các chính sách chăm sóc, khuyến mãi kịp thời nhằm giữ chân họ, tối ưu hóa lợi nhuận.

### Mục tiêu cụ thể
1.  Xây dựng quy trình xử lý dữ liệu (Data Pipeline) từ thô đến sạch.
2.  Thực hiện phân tích khám phá (EDA) để tìm ra các yếu tố tác động chính.
3.  **Xây dựng thuật toán Logistic Regression từ con số 0 (From Scratch)** chỉ sử dụng `NumPy` (không dùng Scikit-learn cho lõi thuật toán) để hiểu sâu bản chất toán học.
4.  Đạt độ chính xác (Accuracy) và Recall (khả năng phát hiện khách rời bỏ) ở mức chấp nhận được.

## 2. Mục lục (Table of Contents)
- [Giới thiệu](#1-giới-thiệu-introduction)
- [Dataset](#3-dataset)
- [Method (Phương pháp)](#4-method-phương-pháp)
- [Installation & Setup](#5-installation--setup)
- [Usage](#6-usage)
- [Results](#7-results-kết-quả)
- [Project Structure](#8-project-structure)
- [Challenges & Solutions](#9-challenges--solutions)
- [Future Improvements](#10-future-improvements)
- [Contributors & Author](#11-contributors--author-info)
- [License](#13-license)

## 3. Dataset

* **Nguồn dữ liệu:** [Kaggle - Credit Card Customers](https://www.kaggle.com/sakshigoyal7/credit-card-customers)
* **Kích thước:** ~10,000 dòng (mẫu) và 21 cột (đặc trưng).
* **Đặc điểm:** Dữ liệu bao gồm thông tin nhân khẩu học (Tuổi, Giới tính, Học vấn) và hành vi tiêu dùng (Số giao dịch, Tổng tiền, Dư nợ).
* **Các Features quan trọng sau khi chọn lọc:**
    * `Total_Trans_Ct`: Tổng số lần giao dịch (Feature quan trọng nhất).
    * `Has_Revolving_Bal`: Biến phái sinh, chỉ ra khách hàng có đang nợ tín dụng hay không.
    * `Contacts_Count_12_mon`: Số lần liên hệ với ngân hàng.

## 4. Method (Phương pháp)

### Quy trình xử lý dữ liệu
1.  **Preprocessing:** Xử lý Missing Value (Mean cho số, Mode cho chữ, giữ 'Unknown' cho thu nhập).
2.  **Feature Engineering:** Tạo biến mới (`Avg_Trans_Amount`, `Has_Revolving_Bal`), loại bỏ biến đa cộng tuyến (`Total_Trans_Amt`, `Avg_Utilization_Ratio`).
3.  **Encoding:** One-Hot Encoding thủ công cho biến phân loại.
4.  **Scaling:** Z-score Standardization ($\frac{x - \mu}{\sigma}$).

### Thuật toán sử dụng: Logistic Regression
Vì yêu cầu cài đặt thủ công, tôi sử dụng Logistic Regression với tối ưu hóa bằng Gradient Descent.

**1. Hàm Sigmoid (Activation Function):**
Chuyển đổi đầu ra tuyến tính thành xác suất (0-1).

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Trong đó $z = w \cdot x + b$.

**2. Hàm mất mát (Loss Function - Log Loss):**
Đo lường sai số giữa dự đoán và thực tế.
$$J(w,b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)})]$$

**3. Cập nhật trọng số (Gradient Descent):**
$$w := w - \alpha \frac{\partial J}{\partial w} = w - \alpha \frac{1}{m} X^T (\hat{y} - y)$$

$$b := b - \alpha \frac{\partial J}{\partial b} = b - \alpha \frac{1}{m} \sum (\hat{y} - y)$$

### Giải thích cách implement bằng NumPy
Thay vì dùng vòng lặp (rất chậm), tôi tận dụng khả năng tính toán Vectorization của NumPy:
* Dùng `np.dot(X, w)` để tính tích vô hướng cho toàn bộ ma trận dữ liệu cùng lúc.
* Dùng `np.exp()` và tính toán mảng (broadcasting) để tính Sigmoid và Gradient cực nhanh.

## 5. Installation & Setup

Yêu cầu: Python 3.8+
```bash
# Clone repository
git clone https://github.com/CodeDaoVietNam/BankChurnerPrediction_Project.git
# Di chuyển vào thư mục
cd bank-churn-project

# Cài đặt thư viện cần thiết
pip install -r requirements.txt
```

## 6. Usage
Bạn có thể chạy toàn bộ dự án bằng script tự động hoặc chạy từng notebook:

### **Cách 1: Chạy tự động (Khuyên dùng)**
```bash
# Cấp quyền thực thi (Linux/Mac)
chmod +x run_all.sh

# Chạy script
./run_all.sh
```

### **Cách 2: Chạy từng phần**
- Mở `notebooks/01_data_exploration.ipynb` để xem thống kê mô tả dữ liệu
- Mở `notebooks/02_preprocessing.ipynb` để thực hiện làm sạch và EDA
- Mở `notebooks/03_modeling.ipynb` để huấn luyện và đánh giá mô hình 

## 7. Results (Kết quả)
**Metrics đạt được trên tập test**
**Accuracy**: 0.8672260612043435

**Precision**: 0.5319148936170212

**Recall**: 0.8361204013377923

**F1-score**: 0.6501950585170798

**Trực quan hóa và Phân tích**
- **Phân cụm hành vi:** Biểu đồ scatter plot cho thấy khách hàng có **Tổng số giao dịch < 40** có nguy cơ rời bỏ rất cao
- **Tác động của nợ**: Nhóm khách hàng không có dư nợ xoay vòng (`Has_Revolving_Bal = 0`) chiếm phần lớn lượng khách rời bỏ.

## 8. Project Structure

```text
├── data/                   # Chứa dữ liệu của dự án
│   ├── raw/                # Dữ liệu thô ban đầu (BankChurners.csv)
│   └── processed/          # Dữ liệu đã qua xử lý, mã hóa và file .npy
├── notebooks/              # Các Jupyter Notebook theo quy trình Data Science
│   ├── 01_data_exploration.ipynb  # Khám phá sơ bộ dữ liệu
│   ├── 02_preprocessing.ipynb     # Làm sạch, Feature Engineering & EDA chuyên sâu
│   └── 03_modeling.ipynb          # Huấn luyện mô hình Logistic Regression (From scratch)
├── src/                    # Source code chứa các hàm tiện ích tự định nghĩa
│   ├── data_processing.py  # Các hàm xử lý dữ liệu, missing value, encoding
│   ├── visualization.py    # Các hàm trực quan hóa dữ liệu
│   └── models.py           # Class LogisticRegression tự viết
├── requirements.txt        # Các thư viện cần thiết
├── run_all.sh              # Script chạy toàn bộ dự án (nếu dùng Linux/Mac)
└── README.md               # Tài liệu hướng dẫn
```

## 9. Challenges & Solutions
Trong quá trình xây dựng pipeline xử lý dữ liệu thủ công bằng Numpy , tôi đã đối mặt và giải quyết các thách thức sau:
**A. Trong giai đoạn Tiền xử lý (Preprocessing) & Feature Engineering**
1. **Khó Khăn: Xử lý giá trị `NaN` trong mảng Numpy số học**
    - **Mô tả**: Không giống như Pandas có `fillna()`, Numpy thuần túy không hỗ trợ điền giá trị giá trị thiếu một cách thông minh 
    - **Giải pháp**: Tôi phải sử dụng `np.nanmean` để tính toán bỏ qua giá trị rỗng , sau đó kết hợp kỹ thuật **Boolean Indexing** (`arr[np.isnan(arr)] = mean_val`) để thay thế giá trị một cách vector hóa mà không dùng vòng lặp 
2. **Khó khắn: Rò rỉ dữ liệu (Data Leakage) khi chuẩn hóa (Scaling)**
    - **Mô tả**: Một sai lầm tôi gặp phải lúc đầu là tính Mean và Std trên toàn bộ tập dữ liệu (X_full) trước khi chia Train/Test. Điều này khiến thông tin từ tập Test bị "lộ" sang tập Train.
    - **Giải pháp** Viêt lại quy trình chuẩn hóa: Chỉ tính `Mean` và `Std` dựa trên `X_train` . Sau đó ,dùng đúng 2 giá trị này để biến đổi cho X_test
3. **Khó khăn: Lỗi toán học khi Log Transform**
    - **Mô tả**: Khi thực hiện giảm độ lệch (Skewness) cho các cột như `Total_Trans_Amt`, tôi gặp lỗi trả về `-inf`(âm vô cực) do một số dòng có giá trị bằng 0 (vì $\log(0)$ 
    không xác định).
    - **Giải pháp**: Chuyển sang sử dụng hàm `np.log1p(x)` (tương đương $\ln(1+x)$) thay vì `np.log(x)`. Việc này giúp xử lý mượt mà các giá trị 0 và giữ nguyên tính chất phân phối.

**B. Trong giai đoạn xây dựng thuật toán (Modeling with Numpy)**
1. **Khó khăn: Hiện tượng tràn số (Overflow/Underflow) trong hàm Sigmoid**
    - **Mô tả**: Với các giá trị $z$ quá lớn hoặc quá nhỏ (ví dụ $z = -1000$), hàm `np.exp(-z)` sẽ trả về `inf`, gây ra lỗi khi chia.
    - **Giải pháp**: Sử dụng kỹ thuật `np.clip(z, -500, 500)` để giới hạn miền giá trị của $z$ trong khoảng an toàn trước khi đưa vào hàm mũ, đảm bảo tính ổn định số học (Numerical Stability)
2. **Khó khăn: Kích thước ma trận không khớp (Dimension Mismatch)**
    - **Mô tả**: Lỗi phổ biến nhất là `ValueError: shapes (m,n) and (k,l) not aligned.` Đặc biệt là vector `y` thường có dạng `(m,)` (mảng 1 chiều) trong khi phép nhân ma trận yêu cầu `(m,1)` (vector cột).
    - **Giải pháp**: Luôn kiểm tra` X.shape` và `y.shape` trước khi training. Sử dụng `y.reshape(-1, 1)` để ép kiểu về vector cột tường minh.

## 10. Future Improvements
- Cài đặt thêm thuật toán Random Forest hoặc Neural Network (cũng theo phương pháp from scratch) để so sánh.
- Xây dựng API đơn giản bằng Flask để deploy mô hình.

## 11. Contributors & Author Info
- Họ và tên : Nguyễn Đức Tiến
- MSSV: 23120368
- Lớp/Môn Học: Lập trình cho khoa học dữ liệu 

## 12. Contact
Nếu có thắc mắc , vui lòng liên hệ qua email: tiennguyenbungbu1210@gmail.com

## 13. LICENSE
Dự án được cấp phép theo chuẩn MIT License.