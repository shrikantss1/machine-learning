import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

# 1. CREATE DUMMY DATA
# Features: [Age, Salary]
X = np.array([
    [25, 40000],
    [30, 45000],
    [22, 35000],
    [45, 90000],
    [55, 110000],
    [60, 85000],
    [35, 120000], # Young but rich! (Will buy)
    [50, 40000]   # Old but low salary (Won't buy)
])

# Target: 0 = Won't Buy, 1 = Will Buy
y = np.array([0, 0, 0, 1, 1, 1, 1, 0])

# 2. INITIALIZE AND TRAIN THE MODEL
# We set max_depth to 2 to stop it from over-complicating (overfitting) the tree
tree_model = DecisionTreeClassifier(max_depth=2, random_state=42)

# Notice we DO NOT need to scale the data! 
# Trees don't care about the difference between $120,000 and 35 years of age.
tree_model.fit(X, y)

print("Decision Tree Successfully Trained!\n")

# 3. PRINT THE FLOWCHART
# Scikit-Learn has a beautiful built-in function to print the actual rules the tree learned.
feature_names = ["Age", "Salary"]
tree_rules = export_text(tree_model, feature_names=feature_names)

print("=== THE LEARNED FLOWCHART ===")
print(tree_rules)

# 4. MAKE PREDICTIONS ON NEW DATA
# Let's test a 40-year-old making $100,000
new_person = np.array([[40, 100000]])
prediction = tree_model.predict(new_person)

print(f"\nPrediction for 40 yr old making $100k: Class {prediction[0]}")