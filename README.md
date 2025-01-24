**This script is structured into several key sections:**

**1. Data Loading and Preprocessing:**

- load_data: Loads the dataset from a CSV file.
- add_intercept: Adds an intercept term to the feature matrix.
- split_dataset: Splits data into training, validation, and test sets.

I begin by loading the dataset then splitting the data into a training set, a validation set, and a testing set. Train Set: 70%, Validation Set: 15%, Test Set: 15%
I additionally added a bias term using add_intercept transorming $X$ into an augmented matrix.

**2. Model Training:**

- closed_form_solution: Computes weights using the Normal Equation.
- lms_gradient_descent: Trains weights using Gradient Descent with specified learning rate and epochs.

For model implementation, there were two approaches: a closed-form approach that minimized the sum-of-squares error and a gradient descent algorithm approach.
For the closed-form approach, I computed $X^{T} X$ and $X^{T} y$, inverted $X^{T} X$, and multiplied it with $X^{T} y$ using the helper functions to obtain $w$.
This solution is beneficial because it provides an exact solution without the need for an iterative approach, however, it can be computationally intensive for a
large dataset due to the matrix inversion. 

On the other hand, for the gradient descent approach, there is a random initialization for $w$ which is then updated
for $5000$ iterations with a $\eta = 0.0001$ learning rate using the standard gradient descent update scheme. The gradient of E was calculated by: $\frac{1}{N} X^{T} (Xw - y)$,
where $N$ is the total number of samples. The advantages of this approach is that it scales well to large datasets, however, it requires careful tuning of the hyperparameters
or else it may converge slowly or get stuck in local minima.

**3. Evaluation:**

- rmse: Calculates Root Mean Squared Error.
- predict: Generates predictions using the learned weights.

For model evaluation, I use the RMSE measure from the helper rmse function to calculate how well the models are able to perform given both the training and test data.
We begin by running the predict function to test the model on both the train data and test data, then we compare the predictions of y vs the actual y to get the rmse values.
Generally, the analytical solution will have the lowest error (closed-form). The iterative approach can do well given the right hyperparameters, but it will only ever be as good
as the analytic approach.

Closed-form RMSE:
- *Train:* 0.6487
- *Test:* 0.6451

LMS RMSE:
- *Average Train:* 0.9746
- *Average Test:* 1.0104
  
**4. Visualization:**

- plot_actual_vs_predicted: Creates scatter plots comparing actual vs. predicted quality scores.

Interpretation of the plot:

- *Ideal Scenario:* Points lie on the diagonal line $y = x$, indicating perfect predictions.
- *Good Performance:* Points cluster closely around the diagonal, showing high accuracy.
- *Poor Performance:* Points scatter widely, indicating significant prediction errors.
