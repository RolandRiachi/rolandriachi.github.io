---
layout: post
title: "Introduction to Linear Regression"
date: 2024-03-31
categories: [Machine Learning, Mathematics]
tags: [linear-regression, statistics, python]
author: Roland Riachi
use_pyodide: true
---

Linear regression is one of the most fundamental and widely used algorithms in machine learning. In this post, we'll explore the mathematical foundations of linear regression and implement it from scratch in Python.

## Mathematical Foundation

Linear regression models the relationship between a dependent variable \(y\) and one or more independent variables \(X\) using a linear function:

$$ 
    y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon 
$$

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

Try modifying the parameters below to see how they affect the linear regression:

<style>
.interactive-code .CodeMirror {
    height: calc(50 * 24px);"
}

.interactive-code {
    position: relative;
    display: flex;
    flex-direction: column;
    isolation: isolate;
}

#output-regression-example {
    position: relative;
    display: block;
    margin-top: 10px;
    margin-bottom: 20px;
    clear: both;
}

.run-button {
    position: relative;
    margin: 10px 0;
}
</style>

<div class="interactive-code" id="regression-example">
<div id="editor-regression-example" data-code="import numpy as np
from matplotlib import pyplot as plt

# Generate sample data
def generate_data(a=2, b=4, c=3):
    np.random.seed(42)
    X = a * np.random.rand(100, 1)
    y = b + c * X + np.random.randn(100, 1)
    return X, y

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

# Create and fit the model
X, y = generate_data(a=2, b=4, c=3)
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.5)
plt.plot(X_new, y_pred, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Linear Regression\nIntercept: {model.intercept.item():.2f}, Slope: {model.coefficients[0].item():.2f}')
plt.grid(True)
plt.show()

print(f'Intercept: {model.intercept.item():.2f}')
print(f'Coefficients: {model.coefficients[0].item():.2f}')"></div>
    <button id="run-regression-example" class="run-button">Run</button>
    <div id="output-regression-example" class="output"></div>
</div>

## Conclusion

Linear regression is a powerful and interpretable model that serves as a foundation for understanding more complex machine learning algorithms. In future posts, we'll explore extensions like ridge regression, lasso regression, and polynomial regression.

Stay tuned for more content about machine learning and mathematics! 