# ============================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================================

import numpy as np
import time

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


# ============================================================
# STEP 2: DEFINE LOSS FUNCTION AND GRADIENT (MSE)
# ============================================================

def mse_loss(y_true, y_pred):
    """
    Mean Squared Error loss function
    """
    return np.mean((y_true - y_pred) ** 2)


def mse_gradient(y_true, y_pred):
    """
    Gradient of MSE loss with respect to predictions
    (Negative gradient used in boosting)
    """
    return y_true - y_pred


# ============================================================
# STEP 3: CUSTOM GRADIENT BOOSTING REGRESSOR (FROM SCRATCH)
# ============================================================

class CustomGradientBoostingRegressor:
    """
    Custom implementation of Gradient Boosting Regressor
    using Decision Trees as weak learners.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []
        self.initial_prediction = None
        self.train_rmse = []

    def fit(self, X, y):
        """
        Train the Gradient Boosting model
        """
        # Initial prediction = mean of target values
        self.initial_prediction = np.mean(y)
        y_pred = np.full(y.shape, self.initial_prediction)

        for i in range(self.n_estimators):
            # Compute residuals (negative gradient)
            residuals = mse_gradient(y, y_pred)

            # Train weak learner on residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Update predictions
            update = tree.predict(X)
            y_pred += self.learning_rate * update

            # Store model
            self.models.append(tree)

            # Track convergence
            rmse = np.sqrt(mse_loss(y, y_pred))
            self.train_rmse.append(rmse)

    def predict(self, X):
        """
        Generate predictions
        """
        y_pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.models:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred


# ============================================================
# STEP 4: LOAD AND PREPROCESS DATASET
# ============================================================

# Load California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ============================================================
# STEP 5: TRAIN CUSTOM GBM MODEL
# ============================================================

custom_gbm = CustomGradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

start_time = time.time()
custom_gbm.fit(X_train, y_train)
custom_training_time = time.time() - start_time

custom_predictions = custom_gbm.predict(X_test)
custom_rmse = np.sqrt(mean_squared_error(y_test, custom_predictions))


# ============================================================
# STEP 6: TRAIN LIBRARY GBM MODEL (BENCHMARK)
# ============================================================

library_gbm = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

start_time = time.time()
library_gbm.fit(X_train, y_train)
library_training_time = time.time() - start_time

library_predictions = library_gbm.predict(X_test)
library_rmse = np.sqrt(mean_squared_error(y_test, library_predictions))


# ============================================================
# STEP 7: PERFORMANCE COMPARISON & CONVERGENCE ANALYSIS
# ============================================================

print("\n================ PERFORMANCE COMPARISON ================")
print(f"Custom GBM RMSE       : {custom_rmse:.4f}")
print(f"Library GBM RMSE      : {library_rmse:.4f}")
print(f"Custom Training Time  : {custom_training_time:.2f} seconds")
print(f"Library Training Time : {library_training_time:.2f} seconds")

print("\n================ CONVERGENCE ANALYSIS ==================")
print("First 10 iterations RMSE values:")
for i in range(10):
    print(f"Iteration {i + 1}: RMSE = {custom_gbm.train_rmse[i]:.4f}")

print("\nFinal RMSE after {} estimators: {:.4f}".format(
    len(custom_gbm.train_rmse),
    custom_gbm.train_rmse[-1]
))

