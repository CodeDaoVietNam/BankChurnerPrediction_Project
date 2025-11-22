# Bank Customer Churn Prediction Project
**Thực hiện bởi** : Nguyễn Đức Tiến

**MSSV**: 23120368
## Tổng quan dự án (Project Overview)
Dự án này nhằm mục đích xây dựng một mô hình Machine Learning để dự đoán khả năng rời bỏ (Churn) của khách hàng ngân hàng.

**Điểm đặc biệt của dự án:**
Thay vì chỉ sử dụng các thư viện có sẵn như Scikit-Learn, dự án tập trung vào việc **xây dựng thủ công (implementation from scratch)** các thuật toán cốt lõi bằng **NumPy** để hiểu sâu về bản chất toán học bên dưới, bao gồm:
* Xử lý và làm sạch dữ liệu thủ công.
* Tính toán thống kê (Skewness, Z-score).
* Xây dựng mô hình Logistic Regression với thuật toán Gradient Descent.

## Cấu trúc thư mục (Project Structure)

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

## Hướng dẫn cài đặt và chạy (Installation & Usage)

1. **Cài đặt môi trường**
Yêu cầu Python 3.8+. Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```
2. **Quy trình thực hiện**
Dự án được chia thành 3 giai đoạn chính, tương ứng với 3 notebook:
- **Bước 1**: `notebooks/01_data_exploration.ipynb`.
    - Tải dữ liệu và kiểm tra cấu trúc.
    - Thống kê mô tả cơ bản.
- **Bước 2**: `notebooks/02_preprocessing.ipynp` (Quan trọng).
    - **Cleaning**: Xử lý giá trị thiếu (Mean cho số, Mode cho category data,giữ `Unknown` cho `Income_Category`).
    - **Feature Engineering**: 
        - Tạo biến `Avg_Trans_Amount` (Số tiền trung bình/Giao dịch).
        - Tạo biến `Has_Revolving_Bal` (Trạng thái nợ).
        - **Feature Selection**: Loại bỏ `Total_Trans_Amt` và `Avg_Utilization_Ratio` để tránh **đa cộng tuyến (Multicollinearity)**.
    - **Tranformation**: Kiểm tra độ lệch **(Skewness)** và áp dùng **Log Transform**.
    - **Encoding**: Thực hiện **One_hot_Encoding** cho các biến phân loại có nhiều giá trị khác nhau. 
    - **EDA**: Phân tích sâu các yếu tố tác động đến Churn **(Scatter plots, Box plots).**
- **Bước 3**: `notebooks/03_modeling.ipynb`
    - Chia tập dữ liệu huấn luyện Train/Test **(Tỷ lệ 80/20).**
    - Chuẩn hóa dữ liệu **(Z_score Scaling)** sử dụng **Numpy.**
    - Huấn luyện mô hình **LogisticRegression** tự xây dựng
    - Đánh giá mô hình bằng các chỉ số **Accuracy, Precision, Recall,F1-score và Confusion Matrix**

## Hướng dẫn sử dụng file `run_all.sh`
Để chạy file này trên Linux/MacOS hoặc Git Bash (Windows), bạn cần cấp quyền thực thi cho nó trước.
- **Bước 1**: **Cấp quyền thực thi** Mở terminal tại thư mục gốc của dự án và gõ:
```bash
chmod +x run_all.sh
```
- **Bước 2**: Chạy Script
```bash
./run_all.sh
```

## Các phát hiện chính **(Key Insights)**
Từ quá trình phân tích dữ liệu (EDA), tôi đã rút ra được những góc nhìn sau về bộ dữ liệu này về hành vi của khách hàng :
1. **Tần suất giao dịch là chìa khóa**: Khách hàng có tổng số giao dich thấp (dưới 40 lần/năm) có nguy cơ rời bỏ rất cao. Khi giao dịch > 60 lần, tỷ lệ rời bỏ gần như bằng 0.
2. **Trạng thái hoạt động tín dụng**: Khách hàng không có dư nợ xoay vòng (`Total_Revolving_Bal = 0`) có tỷ lệ rời bỏ cao hơn gấp 4 lần so với nhóm có dư nợ.
3. **Sự thay đổi hành vi**:  Nhóm khách hàng rời bỏ thường có xu hướng giảm mạnh giao dịch trong Q4 so với Q1 (Tỷ lệ `Total_Ct_Chng_Q4_Q1` thấp)

## Công nghệ sử dụng (Technologies)
- Ngôn ngữ : Python
- Thư viện chính: 
    - `Numpy`: Tính toán ma trân, xây dựng thuật toán Machine Learning, xử lý thống kê 
    - `Matplotlb`,`Seaborn` : Trực quan hóa dữ liệu 

## Kết quả mô hình 
Mô hình LogisticRegression đã đạt được kết quả khá khả quan trên tập dữ liệu kiểm thử (Test set):
- **Accuracy** (`~85%`)
- **Recall** (`Churn`) : Đã tối ưu hóa để phát hiện được khách hàng rời bỏ
- **Confusion Matrix**: Cho thấy khả năng phân tách lớp tốt giữa hai lớp khách hàng 
