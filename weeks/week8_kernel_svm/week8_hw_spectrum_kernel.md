# 🧪 Homework 8 — SVM with Spectrum Kernel on Gene Sequence Data

In this exercise, you will implement and experiment with the **spectrum kernel** for sequence classification using Support Vector Machines.

We will use the **UCI Splice-junction Gene Sequences** dataset, which consists of short DNA sequences labeled as either:

- `"EI"` – exon/intron boundary (positive class)
- `"IE"` or `"N"` – intron/exon or neither (negative class)

## 📦 Task 1: Load and Preprocess the Data

The dataset is available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences).

Your tasks:

- **Download the file manually** from:  
  [https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences](https://archive.ics.uci.edu/dataset/69/molecular+biology+splice+junction+gene+sequences)

- Work with the file `splice.data`. Each line in the file has the format:  

  `LABEL,ID,SEQUENCE`

- **Preprocess the data**:
  - Keep sequences that contain **only the letters A, G, C, T** 
  - Discard sequences that contain ambiguous characters (e.g. D, N, S, R)
  - Convert labels to binary classification:
    - `"EI"` → `+1`
    - `"IE"` and `"N"` → `-1`

Once cleaned, you should obtain:

```python
X = np.array(list_of_sequences)
Y = np.array(labels)  # values in {+1, -1}
```

## 🧠 Task 2: Implement the Spectrum Kernel

The **spectrum kernel** of order $k$ compares sequences by counting how many substrings of length $k$ (called $k$-mers) they share.

>**Definition**:
>
>Let $\Phi_k(s)$ be the vector of $k$-mer counts for a sequence $s$. Then:
>
>$$
>K_k(s_i, s_j) = \langle \Phi_k(s_i), \Phi_k(s_j) \rangle
>= \sum_{u \in \mathcal{A}^k} \text{count}_u(s_i) \cdot \text{count}_u(s_j)
>$$
>
>Where:
>- $\mathcal{A} = \{A, C, G, T\}$
>- $\text{count}_u(s)$ is the number of times the $k$-mer $u$ appears in sequence $s$

### Your task:

Implement a Python class `SpectrumKernel` with parameter `k`. Follow the same format as the kernel classes in the lecture.

You will use this class with your custom SVM:

```python
svm = BinaryKernelSVM(kernel='custom', kernel_function=SpectrumKernel(k))
```

## 📊 Part 3: Train and Evaluate with Different k

Evaluate the performance of the kernel SVM with different values of $k$ (spectrum kernel order).

### Steps:

1. Split your dataset into **training** and **test** sets.
2. For a range of $k$ values (e.g. $k = 2$ to $k = 8$):
   - Train your `BinaryKernelSVM` using this `SpectrumKernel(k)`
   - Compute and store **test accuracy**
3. Collect results for all values of $k$
4. Plot **test accuracy as a function of $k$**

### Discussion:

- What value of $k$ gives the best test accuracy?
- How does the accuracy change as $k$ increases?
- How does computational complexity change?



