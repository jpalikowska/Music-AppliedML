# 🚲 Homework 6 — Poisson Regression on Seoul Bike Sharing Dataset

In this homework, you'll implement **Poisson regression** and apply it to the   [Seoul Bike Sharing Demand Dataset](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand). The dataset contains the **daily number of rented bikes**, along with features describing the day (e.g. weather, season, holiday, etc.). Your goal is to **predict the rental count** (number of bikes rented) using all other features.

## 🔧 Data Preprocessing

The dataset includes **categorical variables** such as `Seasons` or `Holiday`. These must be converted into numeric format before training.

### 🔹 Step 1: One-hot encode categorical variables

Use `pd.get_dummies(...)` to convert each categorical column into binary columns.  
For example, the column `Seasons` should become:

```
Seasons_Autumn, Seasons_Spring, Seasons_Summer, Seasons_Winter
```

You can use:

```python
pd.get_dummies(df, columns=['Hour', 'Seasons', 'Holiday', 'Functioning Day'])
```

### 🔹 Step 2: Normalize features

Apply **min-max normalization** to all numerical features to ensure stable training. To do so, implement min-max normalization as a reusable function in `courselib.utils.normalization`.


## 📈 Poisson Regression

Poisson regression is a GLM for **count data** (such as bike rental count in this task). It uses:

- **Exponential activation**: $h(x) = e^{\langle w, x \rangle + b}$
- **Poisson loss function**: $ \mathcal{L}(y, \hat{y}) = \hat{y} - y \log \hat{y} $


## ✅ Task 1: Implement `PoissonRegression`

- Create a new class `PoissonRegression` that inherits from `TrainableModel`.
- Implement:
  - `decision_function`
  - `loss_grad`
  - `_get_params`
- Use exponential activation and Poisson loss as defined above, and derive the gradient expressions.
- Use `courselib` components where appropriate.


## ✅ Task 2: Implement Mean Absolute Error (MAE)

Implement the **Mean Absolute Error** (MAE) metric, defined as:

$$
E(X, Y) = \frac{1}{n} \sum_{i=1}^n |h(x_i) - y_i|
$$

Add this as a reusable function in:

```python
courselib.utils.metrics
```

Once implemented, you can include it in the training loop:

```python
metrics_dict = {
    "MAE": mean_abs_error,
    ...
}
```


## ✅ Task 3: Train and Evaluate

- Train your `PoissonRegression` model on the processed dataset.
- Plot a **learning curve** showing the value of a chosen metric (e.g. MAE and loss) over epochs.


## ✅ Task 4: Compare with Linear Regression

- Train a **linear regression** model on the same dataset.
- Compute and compare **train/test MAE** between the linear and Poisson models.


## 🤔 Discussion

- Which model performs better on this dataset?
- Why is Poisson regression more appropriate for predicting counts?
- What differences do you observe in predictions or training behavior?
