
# README for HW3: Machine Learning Comparisons and Visualizations

## HW3-1: Comparing Logistic Regression and SVM on 1D Data

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

## HW3-2: Visualizing SVM with Circular Data Distribution in 3D

### Question:
How can we visualize a 2D circular dataset in 3D and analyze SVM's performance with an RBF kernel?

### Solution:
We create a synthetic 2D dataset with circular distributions and use an SVM model with an RBF kernel to classify the data. A 3D plot visualizes the data and decision boundaries.

#### Code Snippet:
```python
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from mpl_toolkits.mplot3d import Axes3D

# Generate 2D circular data
np.random.seed(0)
X_train = np.random.randn(100, 2)
y_train = np.array((X_train[:, 0]**2 + X_train[:, 1]**2) < 1, dtype=int)

# Build SVM model
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)

# Streamlit deployment
st.title("2D SVM with Circular Data Distribution")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 3D plot
ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 0, color='red', label='Class 0')
ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 1, color='blue', label='Class 1')
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Class")
ax.set_title("3D Plot of 2D Data with SVM")
plt.legend()

st.pyplot(fig)
```

---

## HW3-3: Comparing SVM and Logistic Regression on Non-Circular 2D Data

### Question:
How can we compare the decision boundaries of SVM and Logistic Regression on non-circular 2D datasets?

### Solution:
We generate a 2D moon-shaped dataset and train both SVM and Logistic Regression models. Decision boundaries are visualized to highlight the classification capabilities of both methods.

#### Code Snippet:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Generate non-circular 2D data
X_train, y_train = make_moons(n_samples=100, noise=0.2, random_state=0)

# Build models
svm_model = SVC(kernel='rbf')
log_reg_model = LogisticRegression()

# Train models
svm_model.fit(X_train, y_train)
log_reg_model.fit(X_train, y_train)

# Set decision boundary
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Plot decision boundaries
plt.figure(figsize=(10, 6))
Z_svm = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)
plt.contourf(xx, yy, Z_svm, alpha=0.3, cmap='coolwarm')

# Plot data points
plt.scatter(X_train[y_train==0][:, 0], X_train[y_train==0][:, 1], color='red', label='Class 0')
plt.scatter(X_train[y_train==1][:, 0], X_train[y_train==1][:, 1], color='blue', label='Class 1')
plt.title("SVM Decision Boundary on Non-Circular 2D Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()
```

---

## Notes
- The first task visualizes differences between Logistic Regression and SVM on simple 1D data.
- The second task demonstrates a 3D visualization of 2D circular data classified by an SVM.
- The third task compares decision boundaries on complex 2D datasets.

## Contact
For questions or feedback, feel free to reach out to the project maintainer.
