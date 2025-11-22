#!/bin/bash



# Dừng script ngay lập tức nếu có bất kỳ lệnh nào bị lỗi

set -e



echo "======================================================="

echo "   BANK CHURN PREDICTION - AUTOMATED PIPELINE"

echo "======================================================="



# 1. Kiểm tra và tạo các thư mục cần thiết nếu chưa có

echo "Step 1: Checking directory structure..."

mkdir -p data/raw

mkdir -p data/processed

mkdir -p models

echo " Directories checked."



# 2. Cài đặt thư viện (Tùy chọn - Bỏ comment nếu muốn tự động cài)

# echo "Step 2: Installing dependencies..."

# pip install -r requirements.txt

# echo " Dependencies installed."



# 3. Chạy Notebook: 02_preprocessing.ipynb

# Lệnh này sẽ chạy notebook, thực thi code và lưu kết quả (đồ thị, logs) lại vào file

echo "-------------------------------------------------------"

echo "Step 3: Running Data Preprocessing & Feature Engineering..."

echo "Executing notebooks/02_preprocessing.ipynb..."



jupyter nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb



echo " Preprocessing completed. Data saved to 'data/processed/'."



# 4. Chạy Notebook: 03_modeling.ipynb

echo "-------------------------------------------------------"

echo "Step 4: Running Model Training & Evaluation..."

echo "Executing notebooks/03_modeling.ipynb..."



jupyter nbconvert --to notebook --execute --inplace notebooks/03_modeling.ipynb



echo " Modeling completed. Results saved to 'models/'."



# 5. Thông báo hoàn tất

echo "======================================================="

echo "PIPELINE FINISHED SUCCESSFULLY!"

echo "   - Processed data: data/processed/"

echo "   - Model outputs:  models/"

echo "   - Notebooks have been updated with latest run outputs."

echo "======================================================="
