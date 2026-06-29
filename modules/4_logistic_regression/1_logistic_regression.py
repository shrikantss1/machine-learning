import numpy as np
from sklearn.linear_model import LogisticRegression

# 1. Create our "Toy" Data
# X = Hours studied
X_train = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# y = Exam Result (0 = Fail, 1 = Pass)
# Notice the trend: less studying = 0, more studying = 1
y_train = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])

# 2. Initialize and train the model
# (This is where the algorithm calculates the best Sigmoid curve)
model = LogisticRegression()
model.fit(X_train, y_train)
print("Model successfully trained on student data!\n")

# 3. The Predictor Phase
# Let's test three new students who studied for 3, 5.5, and 8 hours
X_test = np.array([[3], [5.5], [8]])

# .predict() gives us the hard classifications (0 or 1) based on a 50% threshold
hard_predictions = model.predict(X_test)

# .predict_proba() gives us the exact percentages!
# It returns an array of two numbers: [Probability of 0, Probability of 1]
probabilities = model.predict_proba(X_test)

# 4. Display the results
for i in range(len(X_test)):
    hours = X_test[i][0]
    final_guess = hard_predictions[i]
    prob_fail = probabilities[i][0] * 100
    prob_pass = probabilities[i][1] * 100
    
    print(f"--- Student who studied {hours} hours ---")
    print(f"Probability of Failing (Class 0): {prob_fail:.1f}%")
    print(f"Probability of Passing (Class 1): {prob_pass:.1f}%")
    print(f"Final Model Prediction: Class {final_guess}")
    print("-" * 40)