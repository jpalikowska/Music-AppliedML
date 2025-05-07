# 📦 Homework 4 — Mini-batch SGD

The goal of this task is to implement and test the **Mini-batch Stochastic Gradient Descent (SGD)** algorithm.

As we discussed in the lecture, performing a single iteration of full-batch Gradient Descent (GD) can become computationally expensive for **large datasets**, since it requires calculating gradients over the entire dataset at each step.

To address this, practitioners often use **stochastic variants** of GD, which use only a subset of the data — called a **batch** — during each update step. This leads to faster iterations and faster learning.

In this exercise, you will:
- Modify your GD implementation to support **mini-batch updates**.
- Experiment with different batch sizes to explore their impact on performance and convergence.

## Algorithm 

Your implementation should follow the pseudocode below.

> ## ⚙️ Stochastic Gradient Descent Algorithm
>
> 0. **Choose hyperparameters**:
>    - Learning rate $\eta \in \mathbb{R}_+$
>    - Number of epochs $K \in \mathbb{N}$
>    - Batch size $B \in \mathbb{N}$
> 
> 1. **Initialize** parameters $w^{(0)}$, set $k = 0$.
> 
> 2. **For** each epoch $k < K$: 
>    - Shuffle the dataset $(X,Y)$
>    - Split $(X,Y)$ dataset into batches of size $B$
>    - **For** each batch $(X_i,Y_i)$:
>       - Compute the SGD update:  $w^{next} = w - \eta \nabla_w \ell (w, X_i, Y_i)$
> 
> 3. **Return** the final parameters $w$.


📌 Note:
- Batches should be disjoint within each epoch (i.e., sampled *without replacement*).
- Each epoch should reshuffle the dataset and sample new batches *independently*.

## Data

To better observe the effects of different batch sizes on the computational performance of SGD, we need a dataset significantly larger than the Irises.

In this task, we will use the **HTRU2: Pulsar Candidate Dataset**, which is publicly available on the UCI Machine Learning Repository. This dataset contains **17,898** samples and only **numerical features**, so no additional preprocessing is required.

🔗 **Dataset link:**  
[HTRU2: Pulsar Candidates Dataset (UCI)](https://archive.ics.uci.edu/dataset/372/htru2)

Please download the dataset manually from the link above and load the `.csv` file into your notebook to proceed with the task.

## Tasks

- Implement `SGDClassifier` as a child class of `LinearBinaryClassification`. Follow the same conventions that we used in the lecture to implement the `GDClassifier` class.
- Compute and plot learning curves (for training loss and accuracy) for a variety of batch sizes from 1 (corresponding to vanilla SGD) to the size of the training set (corresponding to full-batch GD). Compare the results using two different strategies:
    - **Fixed number of epochs:** Choose a fixed number of epochs (e.g., 10) and run the training for each batch size. Here, the number of updates will be different for each batch size.
    - **Fixed number of updates:** Choose a fixed total number of updates (e.g., equal to the number of samples in the train set) and run the training for each batch size for this number of steps. Here, the number of epochs will depend on the batch size.
- Observe how the convergence speed and the stability of training depend on the batch size.
- Measure the time taken by your training code in each of the scenarios (you can use Python’s built-in `time` package or Jupyter magic commands `%%time`/`%timeit`).

📌 When you debug and analyze your results, mind the importance of the learning rate.