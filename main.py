import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# (a) Load data
data = pd.read_csv('winequality-red.csv', sep=';')
X = data.drop('quality', axis=1).values
y = data['quality']

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

# Scale the dataset to ensure that the features are the same scale and the lms linear regression converges
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# (c) Analytic Solution function initalization
def closed_form_solution(X, y):
    X_aug = np.hstack((np.ones((X.shape[0], 1)), X))
    xtx = X_aug.T @ X_aug
    xty = X_aug.T @ y
    w = np.linalg.inv(xtx) @ xty
    return w

def predict(X, w):
    X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
    return X_bias @ w

# RMSE function to calculate the root mean square error
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# (f) LMS function initialization
def lms_gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    X_aug = np.hstack((np.ones((X.shape[0], 1)), X))
    # w_0 randomly initialized
    w = np.random.rand(X_aug.shape[1])

    # Stochastic Gradient Descent
    for _ in range(epochs):
        indices = np.random.permutation(X_aug.shape[0])
        X_aug_shuffled = X_aug[indices]
        y_shuffled = y.iloc[indices]

        # Update weights for each sample
        for i in range(X_aug.shape[0]):
            X_i = X_aug_shuffled[i, :]
            y_i = y_shuffled.iloc[i]
            y_pred_i = np.dot(X_i, w)
            E_i = y_i - y_pred_i
            w += learning_rate * E_i * X_i

    return w

# (c) Closed-form Solution used for Linear Regression
w_closed_form = closed_form_solution(X_train, y_train)
y_train_pred_cf = predict(X_train, w_closed_form)
y_test_pred_cf  = predict(X_test, w_closed_form)

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
print(f"  Test RMSE:  {rmse(y_test, y_test_pred_cf):.4f}")

# (f) LMS function used for Linear Regression
w_lms = lms_gradient_descent(X_train, y_train, learning_rate=0.01, epochs=1000)
y_train_pred_lms = predict(X_train, w_lms)
y_val_pred = predict(X_val, w_lms)

# (g) RMSE for LMS
print("\nLMS solution RMSE:")
print(f"  Train RMSE: {rmse(y_train, y_train_pred_lms):.4f}")
print(f"  Validation RMSE:  {rmse(y_val, y_val_pred):.4f}")