# 🔢 Homework 10 — MNIST Classification with MLP

In this assignment, you will apply our **multi-layer perceptron (MLP)** model to the classic **MNIST dataset** of handwritten digits.

## 🗂 Dataset

The [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) contains grayscale images of handwritten digits (0–9), each of size **28×28 pixels**, flattened into 784-dimensional vectors. It is widely used for benchmarking image classification models.

We'll use the preprocessed version provided by `scikit-learn`. To load the data, use:

```python
from sklearn.datasets import fetch_openml

X, Y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
```

## Task 1: Implement Activations and Losses

Add the following components to your codebase:

- **`Sigmoid` activation**
  $$
  \sigma(x) = \frac{1}{1 + e^{-x}}, \quad \sigma'(x) = \sigma(x)(1 - \sigma(x))
  $$

- **`CrossEntropy` loss** (for one-hot labels)
  $$
  \mathcal{L}(Y, \hat{Y}) = - \sum_k Y[k] \log(\hat{Y}[k]),
  $$

These should be implemented as classes, similar to `ReLU` and `L2`.


## Task 2: Experiment with Architectures & Training

Try different combinations of:

- 🔢 **Hidden layers**: depth and width (e.g. 1 layer vs 3 layers, 64 vs 128 neurons)
- ⚙️ **Activation functions**: ReLU vs Sigmoid
- 🎯 **Loss functions**: MSE vs CrossEntropy
- 📦 **Batch sizes**: try at least two values (e.g. 64 and 256)

For each configuration, record:

- Final **train and test accuracy**
- Training time and learning behavior


## Task 3: Visualize Misclassifications

For your best-performing model:

- Display a few **misclassified examples** from the test set
- Count how many times each digit (0–9) is misclassified
- Discuss which digits are most confusing and why (e.g. 4 vs 9, 5 vs 3)


## 🎯 Goal

**Test accuracy of ≥94%** should be feasible even with small MLPs and proper activation/loss choices.

