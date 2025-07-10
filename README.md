# 🎵 Music Genre Classification using Machine Learning

This repository contains the official implementation of our **music genre classification** project using machine learning techniques.  
The project was developed as part of the *Applied Machine Learning* course and makes use of the custom *courselib* library created during lectures.


## 📊 Overview

We evaluate and compare the performance of two different models — an SVM with RBF kernel and a Multilayer Perceptron (MLP) neural network —  using MFCC-based features extracted from the GTZAN dataset.

The main objectives of the project are:

- Analyze overall classification accuracy  
- Assess per-genre performance  
- Investigate feature strengths and limitations  


## 📦 Requirements

To install all required dependencies, run:

```bash
pip install -r requirements.txt
```

## 📦 Virtual Environment

We recommend creating a virtual environment first:

<pre><code class="bash">
python -m venv test_env
source test_env/bin/activate        # Linux/macOS
test_env\Scripts\activate.bat       # Windows
</code></pre>


## 🏋️‍♂️ Training & Evaluation

All training and evaluation steps are included in the Jupyter notebook `notebook.ipynb`.

Simply run all cells to:

- Load and preprocess the data  
- Extract MFCC features  
- Train two models: SVM with RBF kernel and a Multilayer Perceptron (MLP)  
- Evaluate models using accuracy, confusion matrix, and per-genre performance  
- Analyze feature limitations and genre-specific classification challenges  

## ✅ Results

The models achieved the following performance in classifying music genres based on MFCC features:

| Model               | Accuracy     |
|--------------------|--------------|
| SVM (RBF kernel)   | XX.X%        |
| MLP Neural Network | XX.X% *(TBD)*|

Additional results, including the confusion matrix and per-genre accuracy, are discussed in detail in the attached PDF report.
