import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. CREATE THE "IMPOSSIBLE" DATASET
# make_circles generates a large circle containing a smaller circle in 2D.
# This data is mathematically impossible to separate with a straight line.
X, y = make_circles(n_samples=500, noise=0.1, factor=0.3, random_state=42)

# Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- TESTING DIFFERENT SVM KERNELS ---")

# ==========================================
# TEST 1: Linear Kernel (No Kernel Trick)
# ==========================================
print("\n1. Linear Kernel (Straight Line)")
# The algorithm will try to draw a straight flat line across the circles.
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)

pred_linear = svm_linear.predict(X_test)
acc_linear = accuracy_score(y_test, pred_linear)
print(f"Accuracy: {acc_linear * 100:.2f}%")
print("Result: Failed! A straight line cannot cut a circle in half properly.")

# ==========================================
# TEST 2: Polynomial Kernel (Bending the math)
# ==========================================
print("\n2. Polynomial Kernel (Degree=2)")
# This tells the SVM to mathematically engineer features like X^2 (just like our HTML visualizer!)
svm_poly = SVC(kernel='poly', degree=2)
svm_poly.fit(X_train, y_train)

pred_poly = svm_poly.predict(X_test)
acc_poly = accuracy_score(y_test, pred_poly)
print(f"Accuracy: {acc_poly * 100:.2f}%")
print("Result: Success! By squaring the features, the circles became a 3D bowl.")

# ==========================================
# TEST 3: RBF Kernel (The Industry Standard)
# ==========================================
print("\n3. RBF Kernel (Radial Basis Function)")
# RBF is the default Kernel in Scikit-Learn. Instead of just making a 3D bowl, 
# RBF theoretically maps data into INFINITE dimensions to find the perfect separation plane.
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)

pred_rbf = svm_rbf.predict(X_test)
acc_rbf = accuracy_score(y_test, pred_rbf)
print(f"Accuracy: {acc_rbf * 100:.2f}%")
print("Result: Perfect! RBF easily isolates the inner circle.")

print("\n--- Final RBF Classification Report ---")
print(classification_report(y_test, pred_rbf))