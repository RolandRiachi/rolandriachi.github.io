---
layout: post
title: "Understanding Model Regularization"
date: 2024-04-01
categories: [Machine Learning, Mathematics]
tags: [regularization, ridge-regression, lasso-regression, python]
author: Roland Riachi
---

In our [previous post on linear regression]({% post_url 2024-03-31-introduction-to-linear-regression %}), we explored the foundations of linear models. Today, we'll dive into regularization techniques that help prevent overfitting in these models.

## Why Regularization?

While [linear regression]({% post_url 2024-03-31-introduction-to-linear-regression %}#mathematical-foundation) is powerful in its simplicity, it can sometimes fit noise in the training data too closely. This is where regularization comes in - it adds constraints to the model to prevent this overfitting.

## Types of Regularization

### Ridge Regression (L2)

Ridge regression adds the L2 norm of the coefficients to the cost function:

$$ 
\min_{\beta} \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p \beta_j^2 
$$

where \(\lambda\) is the regularization strength.

### Lasso Regression (L1)

Lasso (Least Absolute Shrinkage and Selection Operator) uses the L1 norm instead:

$$ 
\min_{\beta} \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^p |\beta_j| 
$$

## Implementation in Python

Building on our [previous implementation]({% post_url 2024-03-31-introduction-to-linear-regression %}#implementation-in-python), here's how we can add ridge regularization:

```python
import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coefficients = None
        self.intercept = None
    
    def fit(self, X, y):
        # Add bias term
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Calculate coefficients using normal equation with regularization
        n_features = X_b.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0  # Don't regularize the bias term
        
        self.coefficients = np.linalg.inv(
            X_b.T.dot(X_b) + self.alpha * I
        ).dot(X_b.T).dot(y)
        
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]
```

## When to Use Each Type

- **Ridge Regression**: When you suspect many features contribute a bit to the outcome
- **Lasso Regression**: When you want feature selection (some coefficients become exactly zero)
- **Elastic Net**: When you want a combination of both (we'll cover this in a future post)

## Next Steps

In future posts, we'll explore:
- [Elastic Net Regression][elastic-net] (coming soon)
- [Cross-validation for Model Selection][cross-val] (coming soon)
- [Feature Engineering][feature-eng] (coming soon)

## Conclusion

Regularization is a crucial technique in the machine learning practitioner's toolbox. By understanding when and how to apply different types of regularization, we can build more robust and generalizable models.

[elastic-net]: /coming-soon "Elastic Net: Combining L1 and L2 Regularization"
[cross-val]: /coming-soon "Cross-validation Techniques for Model Selection"
[feature-eng]: /coming-soon "Advanced Feature Engineering Techniques"