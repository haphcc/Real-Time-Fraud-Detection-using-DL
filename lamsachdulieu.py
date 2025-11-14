import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

du_lieu = pd.read_csv("creditcard.csv")

# kieu tra du lieu thieu
print("Giá trị bị thiếu:")
print(du_lieu.isnull().sum())

# phat hien bat thuiong (outliers) o cot Amount
plt.boxplot(du_lieu['Amount'])
plt.title("Phân bố cột Amount")
plt.show()

# xac dinh nguong bat thuong theo z-score
z_score = np.abs((du_lieu['Amount'] - du_lieu['Amount'].mean()) / du_lieu['Amount'].std())
nguong = 3
so_outlier = (z_score > nguong).sum()
print(f"Số lượng giá trị bất thường trong Amount: {so_outlier}")

# loai bo outliers
du_lieu = du_lieu[z_score <= nguong]

# chuan hoa data
scaler = StandardScaler()
du_lieu['Amount'] = scaler.fit_transform(du_lieu[['Amount']])

# dua time ve gio
du_lieu['Time'] = du_lieu['Time'] / 3600

print("\nSau khi chuẩn hóa:")
print(du_lieu[['Time', 'Amount']].head())

# show data
print("\nThống kê mô tả sau khi làm sạch:")
print(du_lieu.describe())
