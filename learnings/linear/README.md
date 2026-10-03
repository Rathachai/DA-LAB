# DA-LAB · Interactive Learning Labs

A collection of browser-based, zero-install simulators that teach core data-analytics and machine-learning concepts through hands-on experimentation. Every lab is a single self-contained HTML file (HTML + CSS + JavaScript), is responsive on desktop and mobile, and is written for engineering students who are new to data science.

**Live site:** <https://rathachai.github.io/DA-LAB/learnings/linear/>

---

## Linear Models

| Lab | Topic | Open |
|-----|-------|------|
| **LM101** | Linear regression fundamentals | [Launch](https://rathachai.github.io/DA-LAB/learnings/linear/lm101.html) |
| **CORR101** | Pearson correlation | [Launch](https://rathachai.github.io/DA-LAB/learnings/linear/corr101.html) |
| **LM201** | Feature selection for linear models | [Launch](https://rathachai.github.io/DA-LAB/learnings/linear/lm201.html) |
| **LM301** | Train–test split | [Launch](https://rathachai.github.io/DA-LAB/learnings/linear/lm301.html) |
| **LM302** | Train–test split with feature selection | [Launch](https://rathachai.github.io/DA-LAB/learnings/linear/lm302.html) |
| **LM401** | Time series with sliding windows | [Launch](https://rathachai.github.io/DA-LAB/learnings/linear/lm401.html) |

### LM101 · Linear Regression Simulator
Build intuition for how a straight line is fitted to data.
- Add random linear data or paint points freehand with an adjustable spray tool.
- Drag the model line (slope and intercept) and see the residuals update live.
- Train with gradient descent: step-by-step or continuous play, adjustable learning rate, random model restarts.
- Track the fitted equation `y = mx + c`, MAE, RMSE, MAPE, MSE and the loss curve.

### CORR101 · Pearson Correlation Demo
Explore what the correlation coefficient *r* does — and does not — measure.
- Choose from preset patterns (positive, negative, none, perfect line, circle, U-shape, sine wave, clusters, single outlier) or spray your own.
- Watch *r*, *r²*, covariance, means and standard deviations respond instantly, with the correlation line drawn on the scatter plot.
- Learn the limits of *r*: non-linear relationships, outliers and clustering effects.

### LM201 · Feature Selection for Linear Models
Decide which variables belong in a multiple linear regression.
- Explore a 100-sample table (`id`, `x1`–`x7`, `y`) and click any column to inspect its relationship with `y`.
- Tick columns to include as features, and optionally apply a log transform to linearize curved relationships.
- Train a multi-feature model with gradient descent (or solve it directly) and read the equation `β₀ + β₁x₁ + …`.
- Compare candidate models side by side using MAE, RMSE, MAPE, R² and adjusted R², alongside the loss curve and a *y* vs. *ŷ* plot.

### LM301 · Train–Test Split
See why a model must be judged on data it has never seen.
- Split points into train and test sets at any ratio from 10 % to 90 %, and re-shuffle the assignment at will.
- Fit on the training set only; compare train vs. test MAE, RMSE, MAPE, MSE and R².
- Toggle train/test visibility and run a 50-split experiment to see how the split ratio affects stability.

### LM302 · Train–Test Split with Feature Selection
Combine feature selection with honest, held-out evaluation.
- The LM201 workflow plus an adjustable 10–90 % train/test split of the 100-sample table.
- Compare train and test MAE and MAPE across feature sets and split ratios to spot overfitting.
- Toggle train/test points and residuals on the charts; test points are shown as diamonds.

### LM401 · Time Series with Linear Models
Turn a time series into a supervised-learning problem and forecast it with a linear (autoregressive) model.
- Pick from ten 200-point patterns (sine, trend, trend + season, random walk, AR(1), level shifts, damped wave, exponential, sawtooth, white noise) and set the noise level.
- Click the line and drag it up or down — neighbouring points follow smoothly, fading out with distance.
- Choose a sliding-window size (up to 10) and inspect the resulting table of `x₁ … x_w → y`, sliding by one step.
- Split chronologically or randomly (10–90 %), then train with gradient descent or solve directly.
- Compare train and test errors (MAE, RMSE, MAPE, MSE, R²) against a naive baseline. In a chronological split the test period is evaluated two ways: **chronological** (each prediction built from the model's earlier predictions) and **using actual data** (one step ahead).
- Read the chart: train predictions (red, 1 px), test predictions using actual data (red, 2 px) and the chronological multi-step test forecast (red dashed, 2 px).

---

## Using the Labs

- Open any link above in a modern browser — no installation, accounts or server required.
- To run locally, clone the repository and open the `.html` files directly.

## Repository Layout

```
learnings/
└── linear/
    ├── README.md
    ├── lm101.html
    ├── corr101.html
    ├── lm201.html
    ├── lm301.html
    ├── lm302.html
    └── lm401.html
```

---

<sub>Powered by: Sonnet &nbsp;|&nbsp; AI Whisperer: [rathachai.github.io](https://rathachai.github.io)</sub>
