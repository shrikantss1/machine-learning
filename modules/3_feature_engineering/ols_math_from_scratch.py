import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Create our "Toy" Data
# The hidden rule is y = 3x + 2 (so slope should be 3, intercept should be 2)
X = np.array([1, 2, 3, 4, 5])
y = np.array([5, 8, 11, 14, 17])

print("--- 1. OUR MATH FROM SCRATCH ---")

# Step A: Calculate the means (averages) of X and y
x_mean = np.mean(X)
y_mean = np.mean(y)

# Step B: Calculate the numerator and denominator for the slope (m)
# Numerator: sum of (x - x_mean) * (y - y_mean)
numerator = np.sum((X - x_mean) * (y - y_mean))

# Denominator: sum of (x - x_mean)^2
denominator = np.sum((X - x_mean) ** 2)

# Step C: Calculate Slope (m) and Intercept (b)
m = numerator / denominator
b = y_mean - (m * x_mean)

print(f"Calculated Slope (m): {m}")
print(f"Calculated Intercept (b): {b}")
print(f"Our Learned Equation: y = {m}x + {b}\n")


print("--- 2. SCIKIT-LEARN'S OLS ALGORITHM ---")
# Let's run the exact same data through Scikit-Learn to compare
# Note: Scikit-learn expects X to be a 2D array, so we reshape it
X_reshaped = X.reshape(-1, 1)

# Initialize and fit the model
model = LinearRegression()
model.fit(X_reshaped, y)

print(f"Sklearn Slope (m): {model.coef_[0]}")
print(f"Sklearn Intercept (b): {model.intercept_}")