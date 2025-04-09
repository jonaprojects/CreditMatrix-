
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import xgboost as xgb

# Load training and test datasets
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# Define classification and regression targets
classification_targets = ['loan_status_binary', 'delinq_flag']
regression_targets = ['revol_util', 'annual_inc', 'dti']

# Train multi-output classification model using XGBoost
clf_model = MultiOutputClassifier(xgb.XGBClassifier(
    n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42))
clf_model.fit(X_train, y_train[classification_targets])
clf_preds = clf_model.predict(X_test)

# Train multi-output regression model using XGBoost
reg_model = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, random_state=42))
reg_model.fit(X_train, y_train[regression_targets])
reg_preds = reg_model.predict(X_test)

# Evaluate classification performance
print("\n========= CLASSIFICATION RESULTS =========")
for i, target in enumerate(classification_targets):
    report = classification_report(y_test[target], clf_preds[:, i], output_dict=True)
    accuracy = report['accuracy']
    print(f"Accuracy for {target}: {accuracy:.2%}")

# Evaluate regression performance using MSE and R²
print("\n========= REGRESSION RESULTS =========")
for i, target in enumerate(regression_targets):
    mse = mean_squared_error(y_test[target], reg_preds[:, i])
    r2 = r2_score(y_test[target], reg_preds[:, i])
    print(f"{target} — MSE: {mse:.2f}, Accuracy (R²): {r2:.2%}")
