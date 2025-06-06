import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

# Load data
train = pd.read_csv(r"C:\Users\dipan\OneDrive\Documents\hackathon\dataset\train.csv")
test = pd.read_csv(r"C:\Users\dipan\OneDrive\Documents\hackathon\dataset\test.csv")

target = "efficiency"

# Combine train and test for consistent preprocessing
train["is_train"] = 1
test["is_train"] = 0
test[target] = np.nan
combined = pd.concat([train, test], axis=0, ignore_index=True)

# Drop id
combined.drop("id", axis=1, inplace=True)

# Identify categorical and numerical columns
cat_cols = combined.select_dtypes(include=["object"]).columns.tolist()
num_cols = combined.select_dtypes(include=["float64", "int64"]).columns.tolist()

# Impute missing values for numerical columns
num_imputer = SimpleImputer(strategy="median")
combined[num_cols] = num_imputer.fit_transform(combined[num_cols])

# Impute missing values for categorical columns
cat_imputer = SimpleImputer(strategy="most_frequent")
combined[cat_cols] = cat_imputer.fit_transform(combined[cat_cols])

# Split back to train/test
train_clean = combined[combined["is_train"] == 1].drop(["is_train"], axis=1)
test_clean = combined[combined["is_train"] == 0].drop(["is_train", target], axis=1)

X = train_clean.drop(columns=[target])
y = train_clean[target]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert categorical feature names to indices for CatBoost
cat_features_idx = [X_train.columns.get_loc(col) for col in cat_cols]

# Initialize CatBoost model
model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.003,
    depth=6,
    eval_metric="RMSE",
    random_seed=42,
    early_stopping_rounds=100,
    task_type="GPU",  # Use GPU if available; else remove this line
    verbose=100,
)

# Fit the model
model.fit(
    X_train,
    y_train,
    cat_features=cat_features_idx,
    eval_set=(X_val, y_val),
    use_best_model=True,
)

# Validation predictions and score
val_preds = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
score = 100 * (1 - val_rmse)
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation Score: {score:.2f}")

# Predict on test set
test_preds = model.predict(test_clean)
print("Test predictions completed.")
