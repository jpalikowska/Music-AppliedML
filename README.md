# 🎵 Music Genre Classification using Machine Learning

This repository contains the official implementation of the **Music Genres Classification from Audio Features** project using machine learning techniques.  
The project was developed as part of the *Applied Machine Learning* course at *LMU SoSe 2025* and makes use of the custom *courselib* library created during lectures.


## 📊 Overview

This project focuses on music genre classification using machine learning techniques.

We use the <a href="https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification" target="_blank">GTZAN Dataset on Kaggle</a>, which contains 1000 audio tracks evenly distributed across 10 genres:  
*blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock*

The dataset provides:
- **Precomputed features** such as MFCCs, chroma, and spectral contrast in CSV format
- **Mel spectrograms** already saved as image files

These representations are used as inputs for our classification models.

We trained and evaluated the following models:

- A **Support Vector Machine (SVM)** with **Gaussian (RBF)** kernel, trained on MFCC features.  
  We explored two multi-class classification strategies:  
  - **One-vs-Rest (OvR)** — a separate classifier is trained for each genre vs. the rest  
  - **One-vs-One (OvO)** — classifiers are trained for every pair of genres

- A **Multilayer Perceptron (MLP)** neural network, also trained on MFCC features.

- A **Convolutional Neural Network (CNN)** trained on mel spectrogram images.

Relevant result visualizations and performance metrics are presented throughout the notebook.

## 🔍 Project Structure

```
├── courselib/              # Custom course library developed during lectures
├── data/                   # Folder containing data from the GTZAN dataset
├── pretrained_models/      # Pretrained CNN models for faster evaluation
├── project_notebook.ipynb  # Main notebook with experiments and analysis
├── requirements.txt        # Python dependencies
├── report.pdf              # Project report
```


## 📦 Requirements

- Python **>=3.10**
- Recommended: virtual environment (see below)

To install all required dependencies:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
```

## 🛠️ Virtual Environment

We recommend creating a virtual environment first:

<pre><code class="bash">
python -m venv test_env

# Linux/macOS:
source venv/bin/activate

# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# Windows CMD:
venv\Scripts\activate.bat
</code></pre>


## 🏋️‍♂️ Training & Evaluation

All training, evaluation, and result visualization steps are documented in the Jupyter notebook  
[`project_notebook`](./project_notebook.ipynb).

Running the notebook allows you to:

- Load and preprocess the dataset  
- Extract the MFCC features  
- Train two models on MFCCs:
  - Support Vector Machine (SVM) with RBF kernel (in both One-vs-Rest and One-vs-One settings)
  - Multilayer Perceptron (MLP) neural network
- Train a Convolutional Neural Network (CNN) using mel spectrogram images  
- Evaluate all models using accuracy, confusion matrices, and learning curves  
- Load pretrained CNN models from the [`pretrained_models`](./pretrained_models/) folder

> ⚠️ Training CNNs for many epochs was time-consuming, so we decided to save pretrained models to make the notebook easier to run and reproduce.

## 🧠 Pretrained Models

We provide pretrained CNN models in the [`pretrained_models`](./pretrained_models/) folder to speed up evaluation and avoid long training times.

All models were trained using the configuration defined in the `config_cnn` dictionary from the notebook.  
The only variation is the **number of training epochs**, which is indicated in the filename.

Additionally, the naming convention reflects how `torchvision.transforms` were applied to the dataset:

- Models **without `separate_transform`** in the filename were trained using a single transform pipeline applied to the entire dataset (both training and test data).
- The model **with `separate_transform`** in the name was trained using the current notebook setup, where different transforms are applied to the training and test sets.

> The training transform is intentionally stronger and includes augmentations such as horizontal flipping and affine transformations, while the test transform applies only resizing and normalization.  
> This change was introduced to explore whether separate transforms could improve generalization and accuracy — however, the improvement turned out to be marginal.


## ✅ Final Results

The following table summarizes the final classification accuracy of the models evaluated in this project:

| Model                         | Input Type     | Accuracy     |
|------------------------------|----------------|--------------|
| SVM (RBF kernel, One-vs-Rest)| MFCC features  | 68.5%        |
| SVM (RBF kernel, One-vs-One) | MFCC features  | 71.0%        |
| MLP Neural Network           | MFCC features  | 74.5%        |
| CNN (25 epochs, `separate_transform`) | Spectrograms | 58.0%        |
| CNN (100 epochs, `separate_transform`) | Spectrograms | 64.5%        |
| CNN (25 epochs) | Spectrograms | 59.0%        |
| CNN (100 epochs) | Spectrograms | 62.5%        |

All models were evaluated using accuracy scores, confusion matrices, and per-genre performance metrics.

> Detailed results, including plots and evaluation curves, can be found in the [project_notebook](./project_notebook.ipynb).   
> A PDF report is also provided, summarizing the task, chosen methods, key results, and discussion.
