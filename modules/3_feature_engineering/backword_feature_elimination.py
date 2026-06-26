import numpy as np
import statsmodels.api as sm

# 1. CREATE DUMMY DATA
np.random.seed(42) # Ensures we get the same random numbers every time

# Let's generate 50 houses with 5 features
sqft = np.random.randint(1000, 4000, 50)          # Strong predictor
bedrooms = np.random.randint(1, 6, 50)            # Moderate predictor
age = np.random.randint(1, 50, 50)                # Moderate predictor
noise1 = np.random.randint(1, 100, 50)            # Completely useless feature
noise2 = np.random.randint(1, 100, 50)            # Completely useless feature

# Target (y): Price is based on sqft, bedrooms, age, PLUS some random variance
price = 50000 + (sqft * 150) + (bedrooms * 10000) - (age * 500) + np.random.randint(-10000, 10000, 50)

# Combine all 5 features into our X matrix
X = np.column_stack((sqft, bedrooms, age, noise1, noise2))
y = price

# 2. PREP THE DATA FOR STATSMODELS
# Statsmodels requires a column of 1s at the very beginning to calculate the Intercept (b)
X = np.append(arr = np.ones((len(X), 1)).astype(int), values = X, axis = 1)


# 3. BACKWARD ELIMINATION LOOP
# Our indices are now: [0: Intercept, 1: SqFt, 2: Bedrooms, 3: Age, 4: Noise1, 5: Noise2]
active_features = [0, 1, 2, 3, 4, 5] 
SL = 0.05 # Significance Level (5%)

print("Starting Backward Elimination...\n")

while len(active_features) > 0:
    # Step 2: Fit the model with the currently active features
    X_opt = X[:, active_features]
    ols_model = sm.OLS(endog=y, exog=X_opt).fit()
    
    # Step 3: Get all the p-values
    p_values = ols_model.pvalues
    max_p_value = max(p_values)
    
    # Step 4: Check if the highest p-value is greater than our SL (0.05)
    if max_p_value > SL:
        # Find the index of the highest p-value
        max_index = np.argmax(p_values)
        removed_feature = active_features[max_index]
        
        print(f"Removing feature at index {removed_feature} (P-value: {max_p_value:.4f})")
        
        # Remove the weak link from our active list
        active_features.pop(max_index)
        
        # The loop will now repeat Step 5: re-train without this feature!
    else:
        # If the highest p-value is LESS than 0.05, we STOP!
        break

print("\nElimination Complete!")
print(f"Remaining Significant Features (Indices): {active_features}")
print("\nFinal Model Summary:")
print(ols_model.summary())