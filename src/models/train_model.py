import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def train():
    print("Loading processed data...")

    df = pd.read_csv("data/processed/processed_data.csv")

    X = df.drop("body_fat", axis=1)
    y = df["body_fat"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training models...")

    mlflow.set_experiment("fitness-model")

    with mlflow.start_run():

        #  Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = lr.predict(X_test)
        lr_rmse = mean_squared_error(y_test, lr_pred) ** 0.5

        #  Random Forest
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_rmse = mean_squared_error(y_test, rf_pred) ** 0.5

        # Log metrics
        mlflow.log_metric("lr_rmse", lr_rmse)
        mlflow.log_metric("rf_rmse", rf_rmse)

        print(f"LR RMSE: {lr_rmse}")
        print(f"RF RMSE: {rf_rmse}")

        # Select best model
        best_model = rf if rf_rmse < lr_rmse else lr

        # Log model
        mlflow.sklearn.log_model(best_model, "model")

        # Save locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/model.pkl")

        print(" Model saved + logged in MLflow")


if __name__ == "__main__":
    train()