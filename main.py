import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

train = pd.read_csv(r"C:\Users\dipan\OneDrive\Documents\hackathon\dataset\train.csv")
test = pd.read_csv(r"C:\Users\dipan\OneDrive\Documents\hackathon\dataset\test.csv")


target = "efficiency"


train["is_train"] = 1
test["is_train"] = 0
test[target] = np.nan 
combined = pd.concat([train, test], axis=0, ignore_index=True)


combined.drop("id", axis=1, inplace=True)

num_cols = combined.select_dtypes(include=["float64", "int64"]).columns
num_imputer = SimpleImputer(strategy="median")
combined[num_cols] = num_imputer.fit_transform(combined[num_cols])


cat_cols = combined.select_dtypes(include=["object"]).columns
cat_imputer = SimpleImputer(strategy="most_frequent")
combined[cat_cols] = cat_imputer.fit_transform(combined[cat_cols])

for col in cat_cols:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])


train_clean = combined[combined["is_train"] == 1].drop(["is_train"], axis=1)
test_clean = combined[combined["is_train"] == 0].drop(["is_train", target], axis=1)

X = train_clean.drop(columns=[target])
y = train_clean[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


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


val_preds = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
score = 100 * (1 - val_rmse)
print(f"Validation RMSE: {val_rmse:.4f}")
print(f"Validation Score: {score:.2f}")


