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
| **LM501** | Predictive maintenance: RUL from sensors, split by machine | [Launch](https://rathachai.github.io/DA-LAB/learnings/linear/lm501.html) |
| **LM502** | Predictive maintenance: degradation curve & threshold | [Launch](https://rathachai.github.io/DA-LAB/learnings/linear/lm502.html) |
| **LM503** | Predictive maintenance: early warning with residuals | [Launch](https://rathachai.github.io/DA-LAB/learnings/linear/lm503.html) |
| **LM504** | Predictive maintenance: the cost of wrong predictions | [Launch](https://rathachai.github.io/DA-LAB/learnings/linear/lm504.html) |

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

### LM501 · Remaining Useful Life (RUL) from Sensors
Predict how many hours a machine has left from seven sensor readings.
- Explore a 120-row table (12 machines × 10 snapshots) with `id`, `machine`, sensors `s1`–`s7` and the target `y` = RUL; some sensors are strong, one is pure noise, one is U-shaped and one is exponential (use the per-column log option).
- Select features, train with gradient descent or solve directly, and compare train and test MAE and MAPE across feature sets.
- Learn the key lesson on splitting: **by machine** (correct) versus **by row** (leaky), with a leakage demo that runs 30 random splits of each mode.

### LM502 · Degradation Curve & Maintenance Threshold
Predict when a machine should be repaired from its declining health index.
- Choose a degradation pattern (linear, accelerating, fast early wear, late-onset fault, step shocks), set the sensor noise, and generate a new machine.
- Scrub **now** along the timeline: the model sees only the readings up to that moment.
- Fit a trend line on a recent window (gradient descent or exact solution), extend it to the maintenance threshold θ and to failure, and read the predicted remaining useful life (RUL).
- Track RUL predictions over time against the truth, and see whether a straight line is too optimistic or too pessimistic.
- Set repair, breakdown and wasted-life costs, and explore how the choice of θ trades early repairs against breakdowns.

### LM503 · Early Warning with a Sliding Window
Detect the start of a fault from the residuals of a linear autoregressive model.
- Train a sliding-window model (window up to 10) on the healthy start of a vibration signal; the residual (actual − predicted) becomes the health signal.
- Raise an alarm when the residual exceeds k·σ for m consecutive points; review detection delay, false alarms and the k trade-off table.
- Choose fault types (drift, level shift, growing variance, bearing oscillation) or inject your own anomaly by dragging the signal.

### LM504 · The Cost of Wrong Predictions
See why the model with the lowest error is not always the best decision-maker.
- Simulate a 200-machine fleet with adjustable prediction noise and bias, and spray in extra machines.
- Set a repair threshold, inspection interval and the costs of repairs, wasted life and breakdowns; see outcomes and the cost-versus-threshold curve.
- Save scenarios to compare the lowest-RMSE model with the lowest-cost one.

---

## Textbook Chapters

Each chapter explains the theory and links to the matching interactive labs.

| Chapter | Topic | Labs |
|---------|-------|------|
| [01 · Introduction to Linear Models](en-01-linear-intro.md) | regression, residuals, error metrics, gradient descent, mathematics of least squares | LM101 |
| [02 · Correlation and Feature Selection](en-02-feature-selection.md) | Pearson correlation, multiple regression, screening, log transform, adjusted R² | CORR101, LM201 |
| [03 · Machine Learning and Train–Test Split](en-03-machine-learning.md) | generalisation, overfitting, splitting, leakage, cross-validation | LM301, LM302 |
| [04 · Time Series](en-04-time-series.md) | sliding windows, AR models, chronological split, recursive forecasting | LM401 |
| [05 · Predictive Maintenance](en-05-predictive-maintainance.md) | RUL, degradation thresholds, residual alarms, cost-based decisions | LM501–LM504 |

---

## Using the Labs

- Open any link above in a modern browser — no installation, accounts or server required.
- To run locally, clone the repository and open the `.html` files directly.

## Repository Layout

```
learnings/
└── linear/
    ├── README.md
    ├── en-01-linear-intro.md
    ├── en-02-feature-selection.md
    ├── en-03-machine-learning.md
    ├── en-04-time-series.md
    ├── en-05-predictive-maintainance.md
    ├── lm101.html
    ├── corr101.html
    ├── lm201.html
    ├── lm301.html
    ├── lm302.html
    ├── lm401.html
    ├── lm501.html
    ├── lm502.html
    ├── lm503.html
    └── lm504.html
```

---

<sub>Powered by: Sonnet &nbsp;|&nbsp; AI Whisperer: [rathachai.github.io](https://rathachai.github.io)</sub>
