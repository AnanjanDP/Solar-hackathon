import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Load data
train = pd.read_csv(r"C:\Users\dipan\OneDrive\Documents\hackathon\dataset\train.csv")
test = pd.read_csv(r"C:\Users\dipan\OneDrive\Documents\hackathon\dataset\test.csv")

# Target variable
target = "efficiency"

# Combine train and test for unified preprocessing
train["is_train"] = 1
test["is_train"] = 0
test[target] = np.nan  # Add dummy target for merging
combined = pd.concat([train, test], axis=0, ignore_index=True)

# Drop 'id' column
combined.drop("id", axis=1, inplace=True)

# Handle missing values
# Numerical columns
num_cols = combined.select_dtypes(include=["float64", "int64"]).columns
num_imputer = SimpleImputer(strategy="median")
combined[num_cols] = num_imputer.fit_transform(combined[num_cols])

# Categorical columns
cat_cols = combined.select_dtypes(include=["object"]).columns
cat_imputer = SimpleImputer(strategy="most_frequent")
combined[cat_cols] = cat_imputer.fit_transform(combined[cat_cols])

# Encode categorical variables
for col in cat_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])

# Split combined back into train/test
train_clean = combined[combined["is_train"] == 1].drop(["is_train"], axis=1)
test_clean = combined[combined["is_train"] == 0].drop(["is_train", target], axis=1)

X = train_clean.drop(columns=[target])
y = train_clean[target]

# Optional: train-validation split (for offline score check)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.005,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    
)
model.fit(X_train, y_train)

# Evaluate on validation set
val_preds = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
score = 100 * (1 - val_rmse)
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation Score: {score:.2f}")


