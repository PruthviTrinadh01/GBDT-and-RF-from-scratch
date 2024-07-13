# README for Implementing Random Forest and Gradient Boosting Decision Trees

## Overview
This README provides instructions on how Random Forest (RF) and Gradient Boosting Decision Trees (GBDT) for regression and binary classification tasks are implemented using Python. The implementation includes the necessary steps, from data preparation to model evaluation. This was developed as part of EE511 at the University of Washington under the guidance of Professor J. Bilmes

## Prerequisites
Ensure you have the following packages installed:
- Python 3.6 or above
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

You can install the required packages using Anaconda or pip:
```bash
conda install numpy pandas matplotlib seaborn scikit-learn jupyter
```
or
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Data Preparation
1. **Load the datasets:**
   - Boston house price dataset for regression.
   - Credit-g and Breast Cancer diagnostic datasets for binary classification.

2. **Preprocess the data:**
   - Handle missing values.
   - Normalize/standardize features if necessary.
   - Split the data into training and testing sets.

## Implementation

### Random Forest (RF)
#### Regression
1. **Bootstrap Sampling:**
   - Generate B datasets by sampling with replacement from the original training data using `numpy.random.choice`.

2. **Train Decision Trees:**
   - Independently train B decision trees on the bootstrapped datasets.
   - Use a random subset of features for each split in the tree.

3. **Prediction:**
   - Aggregate the predictions of all B trees by averaging for regression tasks.

4. **Evaluate Performance:**
   - Calculate training and test RMSE (Root Mean Squared Error).

#### Binary Classification
1. **Bootstrap Sampling:**
   - Same as in regression.

2. **Train Decision Trees:**
   - Same as in regression but adapted for classification tasks.

3. **Prediction:**
   - Aggregate the predictions by taking the majority vote for classification tasks.

4. **Evaluate Performance:**
   - Calculate training and test accuracy.

### Gradient Boosting Decision Trees (GBDT)
#### Regression
1. **Initial Prediction:**
   - Start with an initial prediction, usually the mean of the target values.

2. **Iterative Tree Addition:**
   - Add trees sequentially, where each new tree corrects the errors of the previous ensemble.

3. **Gradient Calculation:**
   - Compute gradients and hessians based on the loss function to optimize the tree splits.

4. **Tree Optimization:**
   - Use the calculated gradients and hessians to find the best splits and optimize the tree structure.

5. **Learning Rate:**
   - Apply a learning rate to scale the contribution of each tree.

6. **Evaluate Performance:**
   - Calculate training and test RMSE.

#### Binary Classification
1. **Initial Prediction:**
   - Same as in regression but with the average of 0/1 labels.

2. **Iterative Tree Addition:**
   - Same as in regression.

3. **Gradient Calculation:**
   - Compute gradients and hessians using the logistic loss function.

4. **Tree Optimization:**
   - Same as in regression but adapted for classification.

5. **Learning Rate:**
   - Same as in regression.

6. **Evaluate Performance:**
   - Calculate training and test accuracy.
