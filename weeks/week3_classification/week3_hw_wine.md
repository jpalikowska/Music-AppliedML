# 🍷 Homework 3 — Wine classification

The goal of this task is to classify wine based on the chemical analysis data using **Ordinary Least Squares (OLS)**.

## Load the data
Load the [**wine dataset**](https://archive.ics.uci.edu/ml/datasets/Wine) from the UCI Machine Learning Repository. You can find the description of the dataset under the link. 

If you load the dataset as a Pandas DataFrame, you can use the following line to specify the column names:
``column_names = ['Class','Alcohol', 'Malic acid','Ash', 'Alcalinity of ash', 'Magnesium',
               'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 
                'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']``
                
## Assign class labels
The dataset contains three classes of wine. In this task, we want to work with **binary classification**. Therefore, you should to **discard class "3" from your data and only work with classes "1" and "2"**.

## Shuffle and split the data

Shuffle the data and separate it into train and test subsets. The train set should constitute 80% of the data.

## Visualize

Choose a reasonable **two-dimensional projection** of the data (e.g. columns "Alcohol" and "Proline") and plot the data. The difference between the two classes should be visible in your projection. Plot test and train data using different markers. 


## Implement OLS classifier

Implement a OLS classifier **with intercept** (e.g. the bias $b$ value may be non-zero) as a **child class** of `LinearBinaryClassification` class implemented in the lecture. The class should contain a method `fit(X,Y)`, which fits the parameters of the linear classifier using the theoretical solutions of the OLS problem.

#### Two-dimensional projection of the data

First train your classifier on two-dimensional train data. 
- Plot the **decision regions** of the classifier as in the lecture.
- Evaluate accuracy.

#### Full dataset

Then train an OLS classifier on **all the 13 variables** of the wine dataset.
- Evaluate accuracy.
- Is the dataset linearly separable? 
