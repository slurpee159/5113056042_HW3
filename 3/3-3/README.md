## 3-3: 2D Dataset with Non-Circular Distribution on Feature Plane

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

![image](https://github.com/user-attachments/assets/133dd19d-aa5a-4478-920a-821f8c586eb8)

