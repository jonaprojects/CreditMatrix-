
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor

# Load data
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# Merge y targets into X temporarily for feature engineering
targets = ['revol_util', 'annual_inc', 'dti']
X_train_fe = X_train.copy()
X_test_fe = X_test.copy()

for col in targets:
    X_train_fe[col] = y_train[col]
    X_test_fe[col] = y_test[col]

# Feature Engineering using available targets
for df in [X_train_fe, X_test_fe]:
    df['payment_to_income'] = df['installment'] / (df['annual_inc'] / 12 + 1)
    df['revol_util_ratio'] = df['revol_bal'] / (df['annual_inc'] + 1)
    df['income_per_account'] = df['annual_inc'] / (df['open_acc'] + 1)
    df['income_to_debt_ratio'] = df['annual_inc'] / (df['revol_bal'] + df['installment'] * df['term'] + 1)
    df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)
    df['credit_util_efficiency'] = df['revol_util'] / (df['open_acc'] + 1)

# Remove targets from features
X_train = X_train_fe.drop(columns=targets)
X_test = X_test_fe.drop(columns=targets)

# Keep engineered features
new_features = ['payment_to_income', 'revol_util_ratio', 'income_per_account',
                'income_to_debt_ratio', 'loan_to_income', 'credit_util_efficiency']
X_train[new_features] = X_train_fe[new_features]
X_test[new_features] = X_test_fe[new_features]

# Remove outliers
def remove_outliers(df_X, df_y, cols, percentile=0.95):
    mask = pd.Series([True] * len(df_y))
    for col in cols:
        limit = df_y[col].quantile(percentile)
        mask = mask & (df_y[col] <= limit)
    return df_X[mask], df_y[mask]

X_train_filtered, y_train_filtered = remove_outliers(X_train, y_train, targets)

# Log-transform targets
for col in targets:
    y_train_filtered[col] = np.log1p(y_train_filtered[col])
    y_test[col] = np.log1p(y_test[col])

# Scale features
scaler = RobustScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_filtered), columns=X_train_filtered.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Feature selection based on annual_inc
selector_model = LGBMRegressor(n_estimators=100, random_state=42)
selector_model.fit(X_train_scaled, y_train_filtered['annual_inc'])
importances = pd.Series(selector_model.feature_importances_, index=X_train_scaled.columns)
top_features = importances.sort_values(ascending=False).head(25).index.tolist()

# Final training and evaluation
print("\n===== Final Optimized Regression Results =====")
for target in targets:
    model = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=8,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
    model.fit(X_train_scaled[top_features], y_train_filtered[target])
    y_pred_log = model.predict(X_test_scaled[top_features])
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test[target])
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"Target: {target}")
    print(f"→ Mean Squared Error: {mse:.2f}")
    print(f"→ Accuracy (R²): {r2:.2%}\n")
