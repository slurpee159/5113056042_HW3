import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

# 生成 2D 資料 (圓形分布)
np.random.seed(0)
X_train = np.random.randn(100, 2)
y_train = np.array((X_train[:, 0]**2 + X_train[:, 1]**2) < 1, dtype=int)

# 建立 SVM 模型
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)

# Streamlit 部署
st.title("2D SVM with Circular Data Distribution")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 繪製 3D 分布
ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 0, color='red', label='Class 0')
ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 1, color='blue', label='Class 1')
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Class")
ax.set_title("3D Plot of 2D Data with SVM")
plt.legend()

st.pyplot(fig)