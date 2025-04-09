
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

# Merge X and y temporarily for feature engineering
X_train_fe = X_train.copy()
X_test_fe = X_test.copy()
X_train_fe['annual_inc'] = y_train['annual_inc']
X_test_fe['annual_inc'] = y_test['annual_inc']

# =======================
# FEATURE ENGINEERING
# =======================
for df in [X_train_fe, X_test_fe]:
    df['payment_to_income'] = df['installment'] / (df['annual_inc'] / 12 + 1)
    df['revol_util_ratio'] = df['revol_bal'] / (df['annual_inc'] + 1)
    df['income_per_account'] = df['annual_inc'] / (df['open_acc'] + 1)

# Drop added target columns to restore X
X_train = X_train_fe.drop(columns=['annual_inc'])
X_test = X_test_fe.drop(columns=['annual_inc'])

# Add engineered features to main X
X_train['payment_to_income'] = X_train_fe['payment_to_income']
X_train['revol_util_ratio'] = X_train_fe['revol_util_ratio']
X_train['income_per_account'] = X_train_fe['income_per_account']

X_test['payment_to_income'] = X_test_fe['payment_to_income']
X_test['revol_util_ratio'] = X_test_fe['revol_util_ratio']
X_test['income_per_account'] = X_test_fe['income_per_account']

# =======================
# REMOVE OUTLIERS (95th percentile)
# =======================
def remove_outliers(df_X, df_y, cols, percentile=0.95):
    mask = pd.Series([True] * len(df_y))
    for col in cols:
        limit = df_y[col].quantile(percentile)
        mask = mask & (df_y[col] <= limit)
    return df_X[mask], df_y[mask]

X_train_filtered, y_train_filtered = remove_outliers(X_train, y_train, ['annual_inc', 'dti', 'revol_util'])

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
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_filtered), columns=X_train_filtered.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# =======================
# TRAIN LIGHTGBM MODELS
# =======================
regression_targets = ['revol_util', 'annual_inc', 'dti']
from lightgbm import LGBMRegressor
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
