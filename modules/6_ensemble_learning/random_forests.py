import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. CREATE DUMMY DATA (Predicting if someone defaults on a loan)
# Features: [Credit Score, Income, Age, Months at Current Job]
np.random.seed(42)
X = np.random.randint(100, 1000, size=(200, 4)) 

# Create a target variable (y) that is heavily influenced by Credit Score (index 0) 
# and Income (index 1), while Age and Months at Job are mostly noise.
y = ((X[:, 0] > 600) & (X[:, 1] > 400)).astype(int) 

# Split into Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. INITIALIZE AND TRAIN THE RANDOM FOREST
# n_estimators = 100 means we are building a forest of 100 decision trees!
rf_model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
rf_model.fit(X_train, y_train)

# 3. EVALUATE THE MODEL
predictions = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Random Forest Accuracy: {accuracy * 100:.2f}%\n")

# 4. THE SECRET SUPERPOWER: FEATURE IMPORTANCES
# Because the forest looks at thousands of random splits across all features, 
# it can tell you mathematically exactly which features are the most important!
feature_names = ['Credit Score', 'Income', 'Age', 'Months at Job']
importances = rf_model.feature_importances_

print("=== FEATURE IMPORTANCES ===")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance * 100:.1f}%")

# Notice how the model correctly figures out that Credit Score and Income 
# carry almost all the predictive power, and largely ignores the noise columns!