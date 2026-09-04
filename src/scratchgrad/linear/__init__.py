"""Linear models.

Grows one file per algorithm as M1 progresses (Lasso, LogisticRegression).
Only what is implemented is exported.
"""

from scratchgrad.linear.linear_regression import LinearRegression
from scratchgrad.linear.ridge import Ridge

__all__ = ["LinearRegression", "Ridge"]
