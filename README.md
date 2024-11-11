
# Machine Learning Homework 3

This repository contains three tasks related to Support Vector Machines (SVM), Logistic Regression, and visualization techniques.

## Tasks Overview

### HW3-1: 1D Comparison of Logistic Regression with SVM
This task generates 1-dimensional data and compares Logistic Regression with SVM. The purpose is to visualize how these two models differ on a simple 1D dataset.

#### Code Highlights:
- Data generation for a simple 1D case.
- Logistic Regression and SVM model training.
- Visualization of the decision boundaries.

#### Example Code:
```python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Generate 1D data
np.random.seed(0)
X_train = np.sort(np.random.rand(20) * 10).reshape(-1, 1)
y_train = (X_train > 5).astype(int).ravel()

# Train models
svm_model = SVC(kernel='linear')
log_reg_model = LogisticRegression()
svm_model.fit(X_train, y_train)
log_reg_model.fit(X_train, y_train)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='black', label="Training Data")
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
plt.plot(X_test, svm_model.predict(X_test), label="SVM Prediction", linestyle='--')
plt.plot(X_test, log_reg_model.predict(X_test), label="Logistic Regression Prediction", linestyle=':')
plt.xlabel("Feature")
plt.ylabel("Class")
plt.legend()
plt.title("1D Comparison: Logistic Regression vs. SVM")
plt.show()
```
![image](https://github.com/user-attachments/assets/46f0d8be-3fed-489b-a275-74ea53e4ceff)



---

### HW3-2: 2D SVM with Streamlit Deployment (3D Plot - Circular Data Distribution)
This task demonstrates a circular distribution of data in a 2D feature space and visualizes it in 3D with Streamlit.

#### Code Highlights:
- 2D data generation with circular distribution.
- Training an SVM model with an RBF kernel.
- Deploying with Streamlit and visualizing data in 3D.

#### Streamlit Code:
To run this part, save it as `streamlit_app.py` and run with `streamlit run streamlit_app.py`.

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

# Train SVM model
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
![image](https://github.com/user-attachments/assets/9717898a-1d9a-412a-bcb6-20fa96d46ab4)



---

### HW3-3: 2D Dataset with Non-Circular Distribution on Feature Plane
In this task, a non-circular data distribution is generated, and an SVM model is used to classify it. The decision boundary is then visualized.

#### Code Highlights:
- 2D data generation with a non-circular distribution (using `make_moons`).
- Training an SVM model with an RBF kernel.
- Plotting the decision boundary and data points.

#### Example Code:
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# Generate 2D non-circular data
X_train, y_train = make_moons(n_samples=100, noise=0.2, random_state=0)

# Train SVM model
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train, y_train)

# Plot decision boundary
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='red', label='Class 0')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='blue', label='Class 1')
plt.title("SVM Decision Boundary on Non-Circular 2D Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

# Save plot as output.png
plt.savefig("output.png")
print("Figure saved as 'output.png'")
```

---

## Running the Scripts

To run the Streamlit application for HW3-2, use the following command:
```bash
streamlit run streamlit_app.py
```
![image](https://github.com/user-attachments/assets/325e9076-2d45-4558-aeb3-be7660b6c84d)


For HW3-1 and HW3-3, simply execute the scripts directly in a Python environment that supports plotting.

