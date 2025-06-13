import sklearn
import pickle
import os
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Ensure the sklearn version is compatible with the joblib library
import joblib

# Load the diabetes dataset
diabetes = load_diabetes()

# Split the data into training and testing sets
X,y = diabetes.data, diabetes.target

print(f"Data shape: {X.shape}, Target shape: {y.shape}")
print(f"Feature names: {diabetes.feature_names}")
print(f"Target range: {y.min()} to {y.max()}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train shape: {X_train.shape[0]}, y_train shape: {y_train.shape[0]}")
print(f"X_test shape: {X_test.shape[0]}, y_test shape: {y_test.shape[0]}")

# Create a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, 
                              random_state=42,
                              max_depth=10,
                              min_samples_leaf=5)

# Train the model on the training data
model.fit(X_train, y_train)

#make predictions and evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# create models directory and save the model
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)
with open(os.path.join(model_dir, 'diabetes_model.pkl'), 'wb') as f:
    pickle.dump(model, f)

# Save the model using joblib for better performance with large models
joblib.dump(model, os.path.join(model_dir, 'diabetes_model.joblib'))
print(f"Model saved to {model_dir}/diabetes_model.pkl and {model_dir}/diabetes_model.joblib")

print("Model training and saving completed successfully.")
# Check the sklearn version
print(f"Scikit-learn version: {sklearn.__version__}")
# Check the joblib version
print(f"joblib version: {joblib.__version__}")

