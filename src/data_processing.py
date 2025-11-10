"""
data_processing.py

Các hàm xử lý dữ liệu bằng NumPy:
- đọc CSV
- xử lý missing values
- phát hiện & xử lý outliers
- chuẩn hóa dữ liệu số
- encode categorical (One-hot Encoding)
"""
#==========LOAD DATA============
import numpy as np
#Hàm này để tải dữ liệu từ file BankChurner.csv
def load_data(file_path):
    """
    :param file_path: đường dẫn của file BankChurner.csv
    :return: Trả về một mảng numpy 2 chiều.
    """
    with open(file_path,encoding='utf-8') as f:
        header = f.readline().strip().split(sep = ',')
        header = [col.strip('"') for col in header]
    data = np.genfromtxt(file_path,delimiter=',',dtype = 'U50',skip_header=1)
    # Làm sạch dấu ngoặc kép một lần ngay khi tải
    data = clean_quoted_strings(data)
    return data, header

def clean_quoted_strings(string_array):
    """
    Loại bỏ dấu ngoặc kép (") thừa khỏi mảng dữ liệu chuỗi.
    np.genfromtxt đôi khi giữ lại dấu ngoặc kép.
    """
    return np.char.strip(string_array.astype(str), '"')

#===========XỬ LÝ MISSING VALUE===============
#hàm điền giá trị missing value  cho các cột numerical bằng mean
def fill_missing_value_by_mean(column_data):
    vals = column_data
    #tạo một mask_missing để có thể thay thế missing value (Boolean)
    #Nếu 0: Dữ liệu không thiếu , 1: Dữ liệu bị thiếu
    mask_missing_value = np.array([(val.strip().lower() == 'unknown') or (val.strip() == '') for val in vals])
    #tính trung bình để điền vào giá trị thiệu
    numerical_vals = np.array([float(val) for val in vals[~mask_missing_value]])
    numerical_mean = np.mean(numerical_vals)
    #Thay giá trị missing_value bằng mean
    filled_vals = vals.copy()
    filled_vals[mask_missing_value] = str(numerical_mean)
    print(f"Đã thay thế giá trị trung bình bằng {numerical_mean}")
    return filled_vals

#Hàm điền missing value bằng giá trị mode cho categorical data
def fill_categorical_value_with_mode(column_data):
    vals = column_data
    mask_missing_value =  np.array([(val.strip().lower() == 'unknown') or (val.strip() == '') for val in vals])
    valid_data = vals[~mask_missing_value]
    #Lấy các giá trị khác nhau trong cột đó và trả về tần suất xuất hiện của nó
    unique_vals, counts = np.unique(valid_data,return_counts = True)
    #Lấy mode: Value có tần suất xuất hiện cao nhất
    mode_val = unique_vals[np.argmax(counts)]
    filled_vals = vals.copy()
    filled_vals[mask_missing_value] = mode_val
    print(f"Đã thay thế bằng giá trị mode:{mode_val}")
    return filled_vals


#=========CÁC HÀM XEM THÔNG TIN CHI TIẾT CỦA DATA=========
def _infer_dtype(column_data, missing_mask):
    valid_data = column_data[~missing_mask] #Lọc ra những dữ liệu không bị null
    if len(valid_data) == 0:
        return 'object'
    try:
        float_col = valid_data.astype(float)
        if np.all(float_col == float_col.astype(int)): #Kiểm tra nó thuộc kiểu integer hay không
            return 'int64'
        else: #nếu không trả về kiểu float
            return 'float64'
    except ValueError:
        return 'object'

#Hàm mô tả các thông tin về percentile , min,max,mean,std của một cột dữ liệu số
def describe_data(column_data):
    stats = {
        "count": len(column_data),
        "mean": np.mean(column_data),
        "std_dev": np.std(column_data),
        "min": np.min(column_data),
        "25% (Q1)": np.percentile(column_data,25),
        "median (50%)": np.median(column_data),
        "75% (Q3)": np.percentile(column_data, 75),
        "max": np.max(column_data)
    }
    for key,val in stats.items():
        if isinstance(val, (float, np.floating)):
            stats[key] = round(val, 2)
    return stats
def get_valid_numerical_array(column_data):
    """
    Hàm trợ giúp: Chuyển một cột chuỗi (có 'Unknown') thành
    một mảng số (float) và một mảng boolean (mặt nạ)
    chỉ ra đâu là giá trị 'Unknown'.
    """
    missing_mask = (np.char.lower(column_data) == 'unknown') | \
                   (np.char.strip(column_data) == '')
    numerical_array = np.where(missing_mask, '0', column_data).astype(float)
    return numerical_array, missing_mask
#Hàm để thống kê mô tả các thông tin cho tất cả các cột số
def numpy_describe(data,header):
    numerical_col_names = []
    stats_data = []
    stats_name = ['count', 'mean', 'std_dev', 'min', '25% (Q1)', 'median (50%)', '75% (Q3)', 'max']

    for i, col_name in enumerate(header):
        col = data[:,i] #Kiểm tra tưng cột
        missing_mask = (np.char.lower(col) == 'unknown') | (np.char.strip(col)=='')
        dtype = _infer_dtype(col,missing_mask)  #Kiểm tra nó có phải là dữ liệu numerical không ?

        #Nếu nó là cột số thi xử lý
        if dtype in ['float64','int64']:
            numerical_col_names.append(col_name)
            valid_col = col[~missing_mask].astype(float) #Chuyển về kiểu dữ liệu float

            #Tính toán các thông tin
            stats_dict = describe_data(valid_col)
            col_stats =np.array([stats_dict[name] for name in stats_name])
            stats_data.append(col_stats)
    stats_matrix = np.array(stats_data).T
    #In kết quả
    print("=" * (15 + 12 * len(numerical_col_names)))
    print(f"{'Statistics':<15}", end="")

    for name in numerical_col_names:
        print(f"{name[:12]:>12}", end=" ")  # in tên cột tối đa 12 ký tự, căn phải

    print()
    print("-" * 80)

    for i, label in enumerate(stats_name):
        print(f"{label:<15}", end="")
        for val in stats_matrix[i]:
            print(f"{val:12.2f}", end="")  # căn phải, 2 chữ số thập phân
        print()

    print("=" *80)

#Hàm này để cung cấp kiẻu dữ liệu, số lượng giá trị thiếu, tên các cột, dung lượng bộ nhớ sử dụng
def numpy_info(data, header):
    num_entries = data.shape[0]
    num_cols = data.shape[1]
    print("=" * 60)
    print(f"Entries: {num_entries}")
    print(f"Columns: {num_cols}")
    print("-" * 60)

    print(f"{'STT':<5} {'Column':<30} {'Non-Null Count':<18} {'Dtype'}")
    print("-" * 60)

    for i, col_name in enumerate(header):
        col = data[:, i]

        # Đếm "non-null" (không phải 'Unknown' hoặc rỗng)
        null_mask = (np.char.lower(col) == 'unknown') | \
                    (np.char.strip(col) == '')
        non_null_count = num_entries - np.sum(null_mask)

        # 3. Suy luận kiểu dữ liệu
        dtype = _infer_dtype(col, null_mask)

        # 4. In thông tin
        print(f" {i:<4} {col_name[:60]:<30} {non_null_count:<18} {dtype}")

    print("-" * 60)

    # 5. Tính toán bộ nhớ sử dụng
    memory_usage_mb = data.nbytes / (1024 * 1024)
    print(f"Memory Usage: {memory_usage_mb:.2f} MB")
    print("=" * 60)

#========XỬ LÝ OUTLIERS=========
#Hàm dùng để check outliers bằng tứ phân vị
def check_outliers_iqr(data_column):
    q1 = np.percentile(data_column,25) #Tứ phân vị thứ 1
    q3 = np.percentile(data_column,75) #Tứ phân vị thứ 3
    #Tính iqr
    iqr = q3-q1
    lower_bound = iqr - 1,5*q1
    upper_bound = iqr - 1.5*q3
    return (data_column < lower_bound) | (data_column > upper_bound) # Trả về mảng boolean những giá trị là outliers

#Hàm để loại bỏ các outliers gây cực đoan cho việc dự đoán
def remove_outliers(data,indices):
    """
    Dùng để xóa bỏ những giá trị ngoại lai không cần thiết
    :param data: Mảng dữ liệu 2D
    :param indices: các dòng là Outliers
    :return: Trả về một mảng 2D mới đã được làm sạch
    """
    cleaned_data = np.delete(data,indices,axis = 0) #Xóa theo dòng
    print(f"Đã xóa {len(indices)} outliers. Kích thước sau khi xóa là: {cleaned_data.shape}")
    return cleaned_data


#========CÁC HÀM CHUẨN HÓA DỮ LIỆU
#Hàm chuẩn hóa min max : Công thức (x-min_value)/(max-min)
def min_max_scaler(column_data):
    #Nó sẽ chuẩn hóa các dữ liệu về [0,1]
    min_val = np.min(column_data)
    max_val = np.max(column_data)
    diff = (max_val-min_val)
    if diff == 0 :
        return np.zeros_like(column_data)
    return (column_data-min_val)/diff #Trả về các giá trị của một cột trong khoảng [0,1]

#Hàm này dùng để chuẩn hóa các dữ liệu về phân phối chuẩn tắc với mean= 0 ,std = 1
#Công thức : (x-mean)/std
def z_score_standard(column_data):
    mean_val = np.mean(column_data)
    std_val = np.std(column_data)
    if std_val == 0:
        return np.zeros_like(column_data)
    return (column_data-mean_val)/std_val

#Hàm này dùng để biến đổi nếu dữ lỗi có skew bị lệch
def log_transformation(column_data):
    return np.log1p(column_data)


def decimal_scaling(column_data):
    """
    Chuẩn hóa Decimal Scaling.
    X_scaled = X / 10^j
    'j' là số nhỏ nhất sao cho max(|X_scaled|) < 1.
    """
    data = column_data.astype(float)
    max_abs_val = np.max(np.abs(data))
    if max_abs_val == 0:
        return data

    j = np.ceil(np.log10(max_abs_val))
    return data / (10 ** j)

#=======CÁC HÀM MÃ HÓA=======
def label_encoding(column_data, positive_class):
    """ Chuyển đổi cột nhị phân (chuỗi) thành 0 và 1. """
    return np.where(column_data == positive_class, 1, 0)

def one_hot_encoding(column_data):
    """ Implement One-Hot Encoding cho một cột phân loại (mảng 1D chuỗi). """
    categories = np.unique(column_data)
    num_categories = len(categories)
    ohe_matrix = np.zeros((len(column_data), num_categories), dtype=int)
    for i, category in enumerate(categories):
        ohe_matrix[column_data == category, i] = 1
    return ohe_matrix, categories