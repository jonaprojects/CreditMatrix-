
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler
from lightgbm import LGBMRegressor

# Load data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# =======================
# FEATURE ENGINEERING
# =======================
for df in [X_train, X_test]:
    df['payment_to_income'] = df['installment'] / (df['annual_inc'] / 12 + 1)
    df['revol_util_ratio'] = df['revol_bal'] / (df['annual_inc'] + 1)
    df['income_per_account'] = df['annual_inc'] / (df['open_acc'] + 1)

# =======================
# REMOVE OUTLIERS (95th percentile)
# =======================
def remove_outliers(df, cols, percentile=0.95):
    limits = {col: df[col].quantile(percentile) for col in cols}
    for col, limit in limits.items():
        df = df[df[col] <= limit]
    return df

# Apply to training targets
y_train_filtered = y_train.copy()
X_train_filtered = X_train.copy()

for col in ['annual_inc', 'dti', 'revol_util']:
    high_limit = y_train[col].quantile(0.95)
    mask = y_train[col] <= high_limit
    y_train_filtered = y_train_filtered[mask]
    X_train_filtered = X_train_filtered.loc[mask]

# =======================
# LOG TRANSFORM TARGETS
# =======================
for col in ['annual_inc', 'dti', 'revol_util']:
    y_train_filtered[col] = np.log1p(y_train_filtered[col])
    y_test[col] = np.log1p(y_test[col])  # transform test for comparison

# =======================
# SCALE FEATURES ROBUSTLY
# =======================
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_filtered), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# =======================
# TRAIN LIGHTGBM MODELS
# =======================
regression_targets = ['revol_util', 'annual_inc', 'dti']
models = {}
predictions = {}

for target in regression_targets:
    model = LGBMRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
    model.fit(X_train_scaled, y_train_filtered[target])
    y_pred_log = model.predict(X_test_scaled)
    y_pred = np.expm1(y_pred_log)  # reverse log transform
    y_true = np.expm1(y_test[target])
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    predictions[target] = y_pred
    print(f"Target: {target}")
    print(f"→ Mean Squared Error: {mse:.2f}")
    print(f"→ Accuracy (R²): {r2:.2%}\n")
