# src/visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_target_distribution_pie(target_column, ax, title='Phân bố Biến Mục tiêu'):
    """
    Vẽ biểu đồ tròn (Pie Chart) để thể hiện tỷ lệ phần trăm
    của biến mục tiêu (ví dụ: Existing vs. Attrited).

    Args:
        target_column (np.array): Mảng 1D chứa dữ liệu mục tiêu.
        ax (matplotlib.axes.Axes): Đối tượng Axes để vẽ lên.
        title (str): Tiêu đề của biểu đồ.
    """
    # 1. Đếm các giá trị duy nhất bằng NumPy
    labels, counts = np.unique(target_column, return_counts=True)

    # 2. Vẽ biểu đồ tròn
    ax.pie(counts,
           labels=labels,
           autopct='%1.1f%%',  # Hiển thị tỷ lệ phần trăm
           startangle=90,
           pctdistance=0.85,
           colors=sns.color_palette('pastel')[0:len(labels)])

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)

    ax.set_title(title, pad=20)
    ax.axis('equal')  # Đảm bảo biểu đồ tròn


def plot_numerical_distribution(data_column, ax, title='Phân phối Biến số', xlabel='Giá trị'):
    """
    Vẽ biểu đồ histogram kết hợp KDE (Kernel Density Estimate)
    để xem phân phối của một biến số.

    Args:
        data_column (np.array): Mảng 1D chứa dữ liệu số.
        ax (matplotlib.axes.Axes): Đối tượng Axes để vẽ lên.
        title (str): Tiêu đề của biểu đồ.
        xlabel (str): Nhãn cho trục X.
    """
    sns.histplot(data_column, kde=True, ax=ax, color='skyblue')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Số lượng')


def plot_categorical_distribution(data_column, ax, title='Phân phối Biến Phân loại', ylabel='Hạng mục'):
    """
    Vẽ biểu đồ cột (Count Plot) để xem phân phối của một
    biến phân loại. Vẽ theo chiều ngang (y) để dễ đọc nhãn.

    Args:
        data_column (np.array): Mảng 1D chứa dữ liệu phân loại (dạng chuỗi).
        ax (matplotlib.axes.Axes): Đối tượng Axes để vẽ lên.
        title (str): Tiêu đề của biểu đồ.
        ylabel (str): Nhãn cho trục Y.
    """
    # Sắp xếp các giá trị để biểu đồ đẹp hơn (theo tần suất)
    order = np.unique(data_column, return_counts=True)
    order_indices = np.argsort(order[1])[::-1]  # Sắp xếp giảm dần
    sorted_labels = order[0][order_indices]

    sns.countplot(y=data_column, ax=ax, palette='pastel', order=sorted_labels)
    ax.set_title(title)
    ax.set_xlabel('Số lượng')
    ax.set_ylabel(ylabel)


def plot_correlation_heatmap(corr_matrix, ax, title='Ma trận Tương quan', x_labels=None, y_labels=None):
    """
    Vẽ một biểu đồ nhiệt (Heatmap) cho ma trận tương quan.

    Args:
        corr_matrix (np.array): Ma trận tương quan 2D.
        ax (matplotlib.axes.Axes): Đối tượng Axes để vẽ lên.
        title (str): Tiêu đề của biểu đồ.
        x_labels (list): Nhãn cho trục X.
        y_labels (list): Nhãn cho trục Y.
    """
    if x_labels is None:
        x_labels = []
    if y_labels is None:
        y_labels = []

    sns.heatmap(corr_matrix,
                annot=True,  # Hiển thị giá trị
                fmt='.2f',  # Định dạng số
                cmap='Blues',  # Bảng màu
                ax=ax,
                vmin = -1,
                vmax = 1,
                xticklabels=x_labels,
                yticklabels=y_labels)

    ax.set_title(title)
    # Xoay nhãn trục X nếu cần
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')


def plot_scatter(x_data, y_data, ax, title='Biểu đồ Scatter', xlabel='Trục X', ylabel='Trục Y', hue_data=None):
    """
    Vẽ biểu đồ phân tán (Scatter Plot) để so sánh hai biến số.
    Có thể thêm biến thứ 3 (hue_data) để tô màu.

    Args:
        x_data (np.array): Dữ liệu trục X.
        y_data (np.array): Dữ liệu trục Y.
        ax (matplotlib.axes.Axes): Đối tượng Axes để vẽ lên.
        title (str): Tiêu đề.
        xlabel (str): Nhãn trục X.
        ylabel (str): Nhãn trục Y.
        hue_data (np.array, optional): Dữ liệu để tô màu.
    """
    sns.scatterplot(x=x_data, y=y_data, hue=hue_data, alpha=0.6, ax=ax, legend='auto')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)