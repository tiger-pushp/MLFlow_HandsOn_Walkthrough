import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create a simple regression dataset
X, y = make_regression(n_samples=100, n_features=5, noise=0.1)

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start an MLFlow run
with mlflow.start_run():

    # Log parameters (for example, model hyperparameters)
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("test_size", 0.2)

    # Train a model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics and log them
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("mean_squared_error", mse)

    # Log the model
    mlflow.sklearn.log_model(model, "model")

    print(f"Logged model with MSE: {mse}")

