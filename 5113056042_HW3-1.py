import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 生成 1D 資料
np.random.seed(0)
X_train = np.sort(np.random.rand(20) * 10).reshape(-1, 1)  # 一維特徵
y_train = (X_train > 5).astype(int).ravel()  # 標籤

# 建立模型
svm_model = SVC(kernel='linear')
log_reg_model = LogisticRegression()

# 訓練模型
svm_model.fit(X_train, y_train)
log_reg_model.fit(X_train, y_train)

# 預測
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_svm_pred = svm_model.predict(X_test)
y_log_reg_pred = log_reg_model.predict(X_test)

# 繪製結果
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='black', label="Training Data")
plt.plot(X_test, y_svm_pred, label="SVM Prediction", linestyle='--')
plt.plot(X_test, y_log_reg_pred, label="Logistic Regression Prediction", linestyle=':')
plt.xlabel("Feature")
plt.ylabel("Class")
plt.legend()
plt.title("1D Comparison: Logistic Regression vs. SVM")
plt.show()