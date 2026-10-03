# DA-LAB · Linear Models for Engineers

A course in five topics for engineering students who are new to data science. Each topic has a **lecture** (a short textbook chapter), interactive **labs** that run in the browser with no installation, **data** (CSV files) and **Jupyter notebooks** (demos and exercises) that run in Google Colab.

**Live site:** <https://rathachai.github.io/DA-LAB/learnings/linear/>

---

## 1 · Introduction to Linear Models

*Data tables (`id`, `X`, `y`), the line $\hat y = mx + c$, residuals, error metrics, gradient descent and hyperparameters, the mathematics of least squares.*

* **Lecture** — [Chapter 1 · Introduction to Linear Models](en-01-linear-intro.md) · [ภาษาไทย](th-01-linear-intro.md)
* **Labs**
  * [LM101 · Linear Regression Simulator](https://rathachai.github.io/DA-LAB/learnings/linear/lm101.html) — spray points, drag the line, watch residuals, train with gradient descent
* **Data**
  * [lm101_points.csv](https://rathachai.github.io/DA-LAB/learnings/linear/data/lm101_points.csv) — 40 noisy points · `id, x, y`
* **Notebooks**
  * [Demo · Linear regression step by step](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/nb-01-linear-intro.ipynb) — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/nb-01-linear-intro.ipynb)

---

## 2 · Correlation and Feature Selection

*Pearson correlation, multiple regression, screening features, log transform, adjusted R², standardisation.*

* **Lecture** — [Chapter 2 · Correlation and Feature Selection](en-02-feature-selection.md) · [ภาษาไทย](th-02-feature-selection.md)
* **Labs**
  * [CORR101 · Pearson Correlation Demo](https://rathachai.github.io/DA-LAB/learnings/linear/corr101.html) — preset or sprayed patterns; watch *r*, *r²* and the correlation line
  * [LM201 · Feature Selection for Linear Models](https://rathachai.github.io/DA-LAB/learnings/linear/lm201.html) — choose columns, apply `log`, compare models
* **Data**
  * [corr101_patterns.csv](https://rathachai.github.io/DA-LAB/learnings/linear/data/corr101_patterns.csv) — nine scatter patterns · `pattern, id, x, y` · 900 rows
  * [lm201_data.csv](https://rathachai.github.io/DA-LAB/learnings/linear/data/lm201_data.csv) — 100 samples · `id, x1…x7, y` (`x3` U-shaped, `x4` noise, `x5` exponential)
* **Notebooks**
  * [Demo · Correlation and feature selection](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/nb-02-feature-selection.ipynb) — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/nb-02-feature-selection.ipynb)

---

## 3 · Machine Learning and Train–Test Split

*Supervised learning, generalisation, overfitting, train and test sets, split ratio, data leakage, repeated splits.*

* **Lecture** — [Chapter 3 · Machine Learning and Train–Test Split](en-03-machine-learning.md) · [ภาษาไทย](th-03-machine-learning.md)
* **Labs**
  * [LM301 · Train–Test Split](https://rathachai.github.io/DA-LAB/learnings/linear/lm301.html) — split points 10–90 %, compare train and test error, run 50 random splits
  * [LM302 · Train–Test Split with Feature Selection](https://rathachai.github.io/DA-LAB/learnings/linear/lm302.html) — the LM201 table with an adjustable split
* **Data**
  * [lm301_points.csv](https://rathachai.github.io/DA-LAB/learnings/linear/data/lm301_points.csv) — 40 points with a 70 / 30 split · `id, x, y, split`
  * [lm302_data.csv](https://rathachai.github.io/DA-LAB/learnings/linear/data/lm302_data.csv) — the LM201 table plus `split`
  * [quiz03_data.csv](https://rathachai.github.io/DA-LAB/learnings/linear/data/quiz03_data.csv) — quiz data for `qz-03` · `id, x1…x5, y` · 200 rows (`x2` needs a log)
* **Notebooks**
  * [Demo · Train–test split, overfitting and leakage](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/nb-03-machine-learning.ipynb) — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/nb-03-machine-learning.ipynb)
  * [Quiz · Choose X or ln_X, split and predict (empty code cells)](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/qz-03-machine-learning.ipynb) — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/qz-03-machine-learning.ipynb)
  * [Solutions · Choose X or ln_X, split and predict](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/sol-03-machine-learning.ipynb) — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/sol-03-machine-learning.ipynb)

---

## 4 · Time Series

*Trend, seasonality and noise, sliding windows, autoregressive models, chronological split, one-step vs. multi-step forecasts.*

* **Lecture** — [Chapter 4 · Time Series with Linear Models](en-04-time-series.md) · [ภาษาไทย](th-04-time-series.md)
* **Labs**
  * [LM401 · Time Series with Linear Models](https://rathachai.github.io/DA-LAB/learnings/linear/lm401.html) — ten 200-point patterns, drag the curve, window table, train/test forecast
* **Data**
  * [lm401_timeseries.csv](https://rathachai.github.io/DA-LAB/learnings/linear/data/lm401_timeseries.csv) — ten series of 200 points · `t` + one column per pattern
  * [lm401_window_w5_trend_season.csv](https://rathachai.github.io/DA-LAB/learnings/linear/data/lm401_window_w5_trend_season.csv) — sliding-window table (window 5) · `t, x1…x5, y, split`
* **Notebooks**
  * [Demo · Sliding windows and forecasting](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/nb-04-time-series.ipynb) — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/nb-04-time-series.ipynb)

---

## 5 · Predictive Maintenance

*Remaining useful life (RUL), splitting by machine, degradation thresholds, residual alarms, cost-based decisions.*

* **Lecture** — [Chapter 5 · Predictive Maintenance](en-05-predictive-maintainance.md) · [ภาษาไทย](th-05-predictive-maintainance.md)
* **Labs**
  * [LM500 · A Machine with Sensors](https://rathachai.github.io/DA-LAB/learnings/linear/lm500-machine.html) — a pump with five live sensors; make them drift, let the machine fail, then generate RUL and download the data
  * [LM501 · Remaining Useful Life from Sensors](https://rathachai.github.io/DA-LAB/learnings/linear/lm501.html) — 60 machines, split by machine vs. by row
  * [LM502 · Degradation Curve and Maintenance Threshold](https://rathachai.github.io/DA-LAB/learnings/linear/lm502.html) — extrapolate a trend to a threshold and weigh the cost
  * [LM503 · Early Warning with a Sliding Window](https://rathachai.github.io/DA-LAB/learnings/linear/lm503.html) — detect faults from residuals
  * [LM504 · The Cost of Wrong Predictions](https://rathachai.github.io/DA-LAB/learnings/linear/lm504.html) — why the lowest-error model is not always the cheapest
* **Data**
  * [lm501_data.csv](https://rathachai.github.io/DA-LAB/learnings/linear/data/lm501_data.csv) — 2,400 snapshots of 60 machines · `id, machine, s1…s7, y` (RUL, hours)
  * [lm502_degradation.csv](https://rathachai.github.io/DA-LAB/learnings/linear/data/lm502_degradation.csv) — five degradation patterns · `pattern, t, health_true, health_sensor, failure_time`
  * [lm503_signals.csv](https://rathachai.github.io/DA-LAB/learnings/linear/data/lm503_signals.csv) — five fault scenarios · `scenario, t, signal, is_fault`
  * [lm504_fleet.csv](https://rathachai.github.io/DA-LAB/learnings/linear/data/lm504_fleet.csv) — 200 machines, actual and predicted RUL · `machine_id, actual_rul, predicted_*`
  * [quiz05_sensors.csv](https://rathachai.github.io/DA-LAB/learnings/linear/data/quiz05_sensors.csv) — quiz data for `qz-05` · `id, machine, s1…s6, y` (RUL, hours) · 1,000 rows
* **Notebooks**
  * [Demo · RUL, degradation, early warning and costs](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/nb-05-predictive-maintenance.ipynb) — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/nb-05-predictive-maintenance.ipynb)
  * [Quiz · Predict RUL from selected sensors (empty code cells)](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/qz-05-predictive-maintenance.ipynb) — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/qz-05-predictive-maintenance.ipynb)
  * [Solutions · Predict RUL from selected sensors](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/sol-05-predictive-maintenance.ipynb) — [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Rathachai/DA-LAB/blob/gh-pages/learnings/linear/notebooks/sol-05-predictive-maintenance.ipynb)

---

## About the Data

`lm201_data.csv`, `lm302_data.csv` and `lm501_data.csv` are exact copies of the data shown in the labs (the labs use a fixed random seed). The other files are example datasets generated with the same patterns and noise levels as their labs; the labs themselves create fresh random data each time, so the numbers differ.

### Using a dataset in Google Colab

```python
import pandas as pd

url = "https://rathachai.github.io/DA-LAB/learnings/linear/data/lm201_data.csv"
df = pd.read_csv(url)

X = df[["x1", "x2", "x6", "x7"]]   # features
y = df["y"]                       # target

from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(X, y)
print(model.intercept_, model.coef_)
```

```python
# Time series: build the sliding-window table yourself (LM401)
s = pd.read_csv("https://rathachai.github.io/DA-LAB/learnings/linear/data/lm401_timeseries.csv")["trend_season"]
w = 5
window = pd.DataFrame({f"x{j}": s.shift(w - j + 1) for j in range(1, w + 1)})
window["y"] = s
window = window.dropna()
```

---

## Using the Labs

* Open any lab link in a modern browser — no installation, account or server required; every lab is one self-contained HTML file, responsive on desktop and mobile.
* To run locally, clone the repository and open the `.html` files directly.

## Repository Layout

```
learnings/
└── linear/
    ├── README.md
    ├── en-01-linear-intro.md          lecture 1
    ├── en-02-feature-selection.md     lecture 2
    ├── en-03-machine-learning.md      lecture 3
    ├── en-04-time-series.md           lecture 4
    ├── en-05-predictive-maintainance.md  lecture 5
    ├── th-01 … th-05 (*.md)             lectures 1–5 in Thai
    ├── lm101.html  corr101.html  lm201.html  lm301.html  lm302.html
    ├── lm401.html  lm500-machine.html  lm501.html  lm502.html  lm503.html  lm504.html
    ├── data/                          CSV datasets
    └── notebooks/                     Jupyter demos (nb-*), quizzes (qz-*) and solutions (sol-*)
```

---

<sub>Powered by: Sonnet &nbsp;|&nbsp; AI Whisperer: [rathachai.github.io](https://rathachai.github.io)</sub>
