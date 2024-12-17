## 3-2: 2D SVM with Streamlit Deployment (3D Plot - Circular Data Distribution)

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

![image](https://github.com/user-attachments/assets/57fe2c2a-d1cf-4089-b6ca-380ab955ce88)


