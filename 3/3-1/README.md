## 3-1: 1D Comparison of Logistic Regression with SVM

### Question:
How can we visualize and compare the performance of Logistic Regression and Support Vector Machines (SVM) on one-dimensional data?

### Solution:
We generate a synthetic 1D dataset and train both models to observe their classification boundaries. A comparison plot is created to highlight the differences between the two approaches.

#### Code Snippet:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate 1D data
np.random.seed(0)
X_train = np.sort(np.random.rand(20) * 10).reshape(-1, 1)  # 1D feature
y_train = (X_train > 5).astype(int).ravel()  # Labels

# Build models
svm_model = SVC(kernel='linear')
log_reg_model = LogisticRegression()

# Train models
svm_model.fit(X_train, y_train)
log_reg_model.fit(X_train, y_train)

# Predictions
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_svm_pred = svm_model.predict(X_test)
y_log_reg_pred = log_reg_model.predict(X_test)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='black', label="Training Data")
plt.plot(X_test, y_svm_pred, label="SVM Prediction", linestyle='--')
plt.plot(X_test, y_log_reg_pred, label="Logistic Regression Prediction", linestyle=':')
plt.xlabel("Feature")
plt.ylabel("Class")
plt.legend()
plt.title("1D Comparison: Logistic Regression vs. SVM")
plt.show()
```

---

![image](https://github.com/user-attachments/assets/035877e6-4910-4db1-8e6a-25c5c437fc37)


