import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# 生成 2D 非圓形分布資料
X_train, y_train = make_moons(n_samples=100, noise=0.2, random_state=0)

# 建立 SVM 和 Logistic Regression 模型
svm_model = SVC(kernel='rbf')
log_reg_model = LogisticRegression()

# 訓練模型
svm_model.fit(X_train, y_train)
log_reg_model.fit(X_train, y_train)

# 設置顯示邊界
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 繪製決策邊界
plt.figure(figsize=(10, 6))
Z_svm = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)
plt.contourf(xx, yy, Z_svm, alpha=0.3, cmap='coolwarm')

# 資料點
plt.scatter(X_train[y_train==0][:, 0], X_train[y_train==0][:, 1], color='red', label='Class 0')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='blue', label='Class 1')
plt.title("SVM Decision Boundary on Non-Circular 2D Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()