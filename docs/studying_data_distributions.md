# Studying Data Distributions

In a data science / ML context, "studying the distribution" of a feature rarely means identifying the exact probability distribution family. It means answering these practical questions:

## 1. Is it symmetric or skewed?
Skewed features can distort linear regression because extreme values pull the mean (and the regression line) disproportionately. Right-skewed data (long tail to the right) is very common in real datasets. A log transform on a right-skewed feature often fixes this.

## 2. Does it have outliers?
Extreme values that sit far from the bulk of the data. Detected visually via boxplots or numerically via the IQR rule:
- IQR = Q3 - Q1
- Outlier if value < Q1 - 1.5×IQR or value > Q3 + 1.5×IQR

## 3. Is it roughly normal?
Linear regression doesn't require features to be normal, but it does assume the **residuals** (errors) are. Features that are wildly non-normal often produce non-normal residuals.

## 4. What is its range and scale?
Features on very different scales cause problems when comparing coefficients and slow down gradient descent. This motivates **feature scaling** (standardization).

## 5. Are there anomalies?
Spikes at specific values, gaps, or impossible values that suggest data collection issues.

---

## On formally identifying the distribution family

There is no clean automatic process. In practice:
- Look at the shape (histogram + KDE) and make an educated guess
- Use a **statistical test** when rigor is needed: Shapiro-Wilk for normality, KS test to compare against any distribution
- In most ML workflows, the practical question is: "is this normal enough, or do I need to transform it?"

```python
from scipy import stats

stat, p = stats.shapiro(df["column"])
print(f"p-value: {p:.4f}")
# p < 0.05 → reject normality
```

---

## Summary

"Study the distribution" in ML means understand the *shape and behavior* of your data, not name the distribution family. The practical goal is to catch skew, outliers, and scale differences before they hurt your model.
