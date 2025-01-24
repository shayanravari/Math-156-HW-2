import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# (a) Load data
data = pd.read_csv('winequality-red.csv', sep=';')
X = data.drop('quality', axis=1).values
y = data['quality'].values

# (b) Split the dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42
)

# Helper functions
def add_intercept(X):
    intercept = np.ones((X.shape[0], 1))
    return np.hstack((intercept, X))

def closed_form_solution(X, y):
    xtx = X.T @ X
    xty = X.T @ y
    w = np.linalg.inv(xtx) @ xty
    return w

def predict(X, w):
    return X @ w

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def lms_gradient_descent(X, y, learning_rate=0.0001, epochs=1000):
    n_samples, n_features = X.shape
    w = np.random.randn(n_features)
    
    for _ in range(epochs):
        y_pred = X @ w
        error = y_pred - y
        grad = (X.T @ error) / n_samples
        w = w - learning_rate * grad
        
    return w

# (c) Closed-form Solution
X_train_aug = add_intercept(X_train)
X_val_aug = add_intercept(X_val)
X_test_aug = add_intercept(X_test)

w_closed_form = closed_form_solution(X_train_aug, y_train)
y_train_pred_cf = predict(X_train_aug, w_closed_form)
y_test_pred_cf  = predict(X_test_aug, w_closed_form)

# (d) Plot Actual vs Predicted on the Training Set
plt.figure(figsize=(6,6))
plt.scatter(y_train, y_train_pred_cf, alpha=0.5, color='blue')
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Closed-form: Actual vs Predicted (Train)")
min_val = min(y_train.min(), y_train_pred_cf.min())
max_val = max(y_train.max(), y_train_pred_cf.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.tight_layout()
plt.show()

# (e) RMSE for Closed-form
print("Closed-form solution RMSE:")
print(f"  Train RMSE: {rmse(y_train, y_train_pred_cf):.4f}")
print(f"  Test  RMSE:  {rmse(y_test, y_test_pred_cf):.4f}")

# (f) LMS Solution
w_lms = lms_gradient_descent(X_train_aug, y_train, learning_rate=0.0001, epochs=5000)
y_train_pred_lms = predict(X_train_aug, w_lms)
y_test_pred_lms  = predict(X_test_aug, w_lms)

# (g) RMSE for LMS
print("\nLMS solution RMSE:")
print(f"  Train RMSE: {rmse(y_train, y_train_pred_lms):.4f}")
print(f"  Test  RMSE:  {rmse(y_test, y_test_pred_lms):.4f}")