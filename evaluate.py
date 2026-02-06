import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


def evaluate_model(model, X_test, y_test, y_train):
    """Evaluate regression model and print metrics vs baseline.

    Prints Test R2, Test MSE, Test RMSE, Test MAE, Baseline RMSE and percent
    improvement over baseline RMSE.
    """
    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Baseline: predict mean of training target
    baseline_value = np.mean(y_train)
    baseline_pred = np.full(shape=len(y_test), fill_value=baseline_value)
    baseline_mse = mean_squared_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(baseline_mse)

    # Percentage improvement over baseline RMSE
    if baseline_rmse == 0:
        improvement_pct = float('inf')
    else:
        improvement_pct = (baseline_rmse - rmse) / baseline_rmse * 100

    # Clear formatted output
    print("\nModel evaluation")
    print(f" - Test R2: {r2:.4f}")
    print(f" - Test MSE: {mse:.4f}")
    print(f" - Test RMSE: {rmse:.4f}")
    print(f" - Test MAE: {mae:.4f}")
    print(f" - Baseline RMSE (predict mean of y_train = {baseline_value:.4f}): {baseline_rmse:.4f}")
    print(f" - Percentage improvement over baseline RMSE: {improvement_pct:.2f}%\n")


if __name__ == '__main__':
    # Attempt to load model and data from workspace files if variables aren't provided.
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        raise SystemExit(f"Failed to load model.pkl: {e}")

    # Load dataset
    try:
        df = pd.read_csv('data.csv')
    except Exception as e:
        raise SystemExit(f"Failed to load data.csv: {e}")

    # Expect target column named 'price' (per notebook)
    if 'price' not in df.columns:
        raise SystemExit("Expected target column 'price' in data.csv")

    X = df.drop(columns=['price'])
    y = df['price']

    # Split to obtain X_test, y_test, and y_train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    evaluate_model(model, X_test, y_test, y_train)
