# 🔠 Homework 9 — Softmax Regression on Letter Recognition Dataset

In this assignment, you'll implement the **softmax regression** model and apply it to the [Letter Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/Letter+Recognition).


## 📚 Dataset Description

This dataset contains **16 numerical features** extracted from pixel-based images of **uppercase English letters**.  
Your goal is to classify each sample into one of **26 classes**, corresponding to the letters **A–Z**.

📎 Data link:  
https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data


## 🛠️ Task 1: Implement `SoftmaxRegression` Class

The softmax regression model outputs a probability distribution over $K$ classes using the **softmax activation function**:

$$
h(x) = \text{softmax}(W^\top x + B) \in \mathbb{R}^K
$$

Component-wise, this is:

$$
h_k(x) = \frac{\exp(w_k^\top x + b_k)}{\sum_{j=1}^K \exp(w_j^\top x + b_j)}, \quad 1 \leq k \leq K
$$

Where:
- $K$: number of classes (26)
- $d$: number of features (16)
- $W \in \mathbb{R}^{d \times K}$: weight matrix
- $B \in \mathbb{R}^{K}$: bias vector


### 📉 Cross-Entropy Loss

The model is trained by minimizing the **cross-entropy loss**.  
Assuming labels are one-hot encoded ($y \in \{0,1\}^K$), the loss is defined as:

$$
\mathcal{L}(y, \hat{y}) = - \sum_{k=1}^K y[k] \log(\hat{y}[k])
$$


## 📌 Notes

- Softmax regression is a **multiclass generalization of logistic regression**, modeling outcomes via a **multinomial distribution** instead of a binomial one.
- The **softmax function** ensures predicted probabilities for all classes sum to one.
- The **cross-entropy loss** is a natural choise for this model, it generalizes binary cross-entropy and ensures well-behaved gradients for optimization.

> 💡 As an extra exercise: Derive softmax regression from a maximum likelihood perspective under a multinomial model.


## 🛠️ Task 2: Train and Evaluate

1. **Load and preprocess** the dataset:
   - Shuffle and split into train/test sets
   - One-hot encode the class labels

2. **Train** the softmax regression model using **full-batch gradient descent**.

3. **Evaluate** performance:
   - Plot the **learning curve** (loss vs. iterations)
   - Compute **accuracy** on both training and test sets
   - Optionally compute a **confusion matrix** to inspect misclassifications

4. **Compare**:
   - With **linear least squares (OLS)** classification
   - (Optional) With **one-vs-rest logistic regression**




