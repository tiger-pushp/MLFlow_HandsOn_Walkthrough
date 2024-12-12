import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

# Load data
data = load_diabetes()
X = data.data
y = data.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start an MLFlow experiment
mlflow.set_experiment("MLFlow Walkthrough")

with mlflow.start_run():
    # Train a RandomForest model
    params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Log parameters
    mlflow.log_param("n_estimators", params["n_estimators"])
    mlflow.log_param("max_depth", params["max_depth"])

    # Make predictions and log metrics
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)

    # Log the model
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Model logged with MSE: {mse}")

