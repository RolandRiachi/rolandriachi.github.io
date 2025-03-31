---
layout: post
title: "Introduction to Linear Regression"
date: 2024-03-31
categories: [Machine Learning, Mathematics]
tags: [linear-regression, statistics, python]
author: Roland Riachi
---

Linear regression is one of the most fundamental and widely used algorithms in machine learning. In this post, we'll explore the mathematical foundations of linear regression and implement it from scratch in Python.

## Mathematical Foundation

Linear regression models the relationship between a dependent variable \(y\) and one or more independent variables \(X\) using a linear function:

\[ y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon \]

where:
- \(\beta_0\) is the y-intercept
- \(\beta_i\) are the coefficients for each feature
- \(\epsilon\) is the error term

The goal is to find the values of \(\beta\) that minimize the sum of squared errors:

\[ \min_{\beta} \sum_{i=1}^n (y_i - \hat{y}_i)^2 \]

## Implementation in Python

Here's a simple implementation of linear regression using NumPy:

```python
import numpy as np

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calculate coefficients using normal equation
        self.coefficients = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]
    
    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept
```

## Example Usage

Let's create some sample data and test our implementation:

```python
# Generate sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

print(f"Intercept: {model.intercept:.2f}")
print(f"Coefficients: {model.coefficients[0]:.2f}")
```

## Conclusion

Linear regression is a powerful and interpretable model that serves as a foundation for understanding more complex machine learning algorithms. In future posts, we'll explore extensions like ridge regression, lasso regression, and polynomial regression.

Stay tuned for more content about machine learning and mathematics! 