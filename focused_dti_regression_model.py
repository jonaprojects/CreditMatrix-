
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor

# Load data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# Focused target
target = 'dti'

# Merge y_train[target] for feature engineering
X_train_fe = X_train.copy()
X_test_fe = X_test.copy()
X_train_fe['annual_inc'] = y_train['annual_inc']
X_test_fe['annual_inc'] = y_test['annual_inc']

# Advanced features for dti only
for df in [X_train_fe, X_test_fe]:
    df['installment_to_income'] = df['installment'] / (df['annual_inc'] / 12 + 1)
    df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['revol_bal_to_income'] = df['revol_bal'] / (df['annual_inc'] + 1)
    df['total_debt'] = df['revol_bal'] + df['installment'] * (df['term'] / 12)
    df['debt_to_income'] = df['total_debt'] / (df['annual_inc'] + 1)

# Clean up: remove annual_inc after feature engineering
X_train_fe.drop(columns=['annual_inc'], inplace=True)
X_test_fe.drop(columns=['annual_inc'], inplace=True)

# Keep relevant engineered features
dti_features = list(X_train.columns) + [
    'installment_to_income', 'loan_to_income', 'revol_bal_to_income',
    'total_debt', 'debt_to_income'
]

X_train_final = X_train_fe[dti_features]
X_test_final = X_test_fe[dti_features]

# Remove outliers for dti only
q_high = y_train[target].quantile(0.95)
mask = y_train[target] <= q_high
X_train_final = X_train_final.loc[mask]
y_train_filtered = y_train.loc[mask]

# Log transform dti
y_train_filtered[target] = np.log1p(y_train_filtered[target])
y_test[target] = np.log1p(y_test[target])

# Scale features
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_final), columns=X_train_final.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_final), columns=X_test_final.columns)

# Train tuned model for dti
print("\n===== Focused Regression Model: dti =====")
model = LGBMRegressor(
    learning_rate=0.005,
    n_estimators=2000,
    num_leaves=16,
    max_depth=5,
    min_child_samples=30,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
model.fit(X_train_scaled, y_train_filtered[target])
y_pred_log = model.predict(X_test_scaled)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test[target])
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"Target: {target}")
print(f"→ Mean Squared Error: {mse:.2f}")
print(f"→ Accuracy (R²): {r2:.2%}\n")
