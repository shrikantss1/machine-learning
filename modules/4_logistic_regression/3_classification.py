import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# 1. THE TRAINING DATA (Hours studied vs. Pass(1)/Fail(0))
X_train = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y_train = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model trained!\n")

# 2. THE TESTING DATA (New students)
X_test = np.array([[2], [4], [5.5], [7], [8]])

# The ACTUAL answers for these 5 students (The Ground Truth)
# Notice student #2 (4 hours) actually passed, and student #3 (5.5 hours) actually failed!
y_test = np.array([0, 1, 0, 1, 1]) 

# 3. MAKE PREDICTIONS
predictions = model.predict(X_test)

print("--- The Predictions vs Reality ---")
print(f"Model Guesses:  {predictions}")
print(f"Actual Answers: {y_test}\n")

# 4. THE SCORECARD
print("=== CONFUSION MATRIX ===")
# Format: 
# [True Negatives, False Positives]
# [False Negatives, True Positives]
print(confusion_matrix(y_test, predictions))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, predictions))