"""Linear models.

Grows one file per algorithm as M1 progresses (LogisticRegression next).
Only what is implemented is exported.
"""

from scratchgrad.linear.lasso import Lasso
from scratchgrad.linear.linear_regression import LinearRegression
from scratchgrad.linear.ridge import Ridge

__all__ = ["Lasso", "LinearRegression", "Ridge"]
