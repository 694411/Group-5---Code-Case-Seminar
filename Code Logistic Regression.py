%reset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pandas.tseries.offsets import MonthEnd
from scipy import stats
from statsmodels.tsa.seasonal import STL
from itertools import groupby
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import itertools
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc, roc_curve
from itertools import product
from collections import OrderedDict
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import seaborn as sns
import random

# Set the seed
random.seed(5)
np.random.seed(5)


############# Phase 1 - Import the dataframe #############

# Load the file path and dataframe
filepath_LR = "C:/Users/maaik/Documents/Business analytics and quantitative marketing/Seminar/20240307_lr.csv"
df_LR = pd.read_csv(filepath_LR)
df_LR = df_LR.drop(columns = ['period'], axis=1)


############# Phase 2 - Train, Test & Validation #############

# Function to make the train/test split
def f_train_test_kfold(df, k=3):
    # Group by unique_id and shuffle
    groups = [group for _, group in df.groupby('unique_id')]
    np.random.shuffle(groups)

    # Determine the size of the training set (80%)
    total_rows = len(df)
    train_target = total_rows * 0.8

    # Empty vectors for the train and test set
    train_rows = 0
    train_groups = []
    test_groups = []

    # Assign customers to train or test set (this is based on unique_id)
    for group in groups:
        if train_rows + len(group) <= train_target:
            train_groups.append(group)
            train_rows += len(group)
        else:
            test_groups.append(group)

    # Concatenate groups into DataFrames for train and test sets
    train_df = pd.concat(train_groups).reset_index(drop=True)
    test_df = pd.concat(test_groups).reset_index(drop=True)

    # Split train_df into k-fold sets
    kfolds = []
    np.random.shuffle(train_groups)  # Shuffle again to randomize for k-fold
    fold_size = len(train_df) // k  # Target number of rows for each fold

    # Initialize counters and containers for k-folds
    current_fold_rows = 0
    current_fold_groups = []

    for group in train_groups:
        if current_fold_rows + len(group) <= fold_size:
            current_fold_groups.append(group)
            current_fold_rows += len(group)
        else:
            # If fold reaches its desired size, we add it to the kfolds and start a new fold
            kfolds.append(pd.concat(current_fold_groups).reset_index(drop=True))
            current_fold_groups = [group]  # Start new fold with the current group
            current_fold_rows = len(group)

    # Ensure the last fold is added as well, since it may be smaller than the others
    if current_fold_groups:
        kfolds.append(pd.concat(current_fold_groups).reset_index(drop=True))

    # Adjust if the last fold is significantly smaller than the rest
    if len(kfolds) > k:
        # Merge the last two folds if more than k folds are created
        kfolds[-2] = pd.concat([kfolds[-2], kfolds[-1]]).reset_index(drop=True)
        kfolds.pop()

    return train_df, test_df, kfolds

train_df, test_df, kfolds = f_train_test_kfold(df_LR, k=3)

# Calculate the proportion of non churn in the testset
n_churn = test_df['final_churn'].value_counts()[1]
n_non_churn = test_df['final_churn'].value_counts()[0]
proportion_non_churn = n_non_churn / (n_churn+n_non_churn)

# Drop the unique_id column since this is information leakage
train_df = train_df.drop(columns=['unique_id'])
test_df = test_df.drop(columns=['unique_id'])
for i in range(len(kfolds)):
    kfolds[i] = kfolds[i].drop(columns=['unique_id'])


############# Phase 3 - Parameter Grid #############

# Hyperparameter grid
param_grid = {
    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],  # Penalty strength (inverse)
    'penalty': ['elasticnet'],  # Penalty type
    'solver': ['saga'],  # Saga is the solver used for large data sets. Furthermore, the only solver compatible with L1 penalty
    'l1_ratio': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]  # Ratio of L1 penalty in Elastic Net
}

# Function to get all combinations of parameters
def get_param_combinations(param_grid):
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    return combinations

param_combinations = get_param_combinations(param_grid)


############# Phase 4 - K-Fold Cross Validation #############

# Store the average F1 score for each parameter setting and its corresponding best threshold
average_performances = []

# Grid search:
for params in param_combinations:
    print("Evaluating parameters:", params)
    # Logistic Regression model for each parameter setting
    model = LogisticRegression(**params, max_iter=500, warm_start=True)
    
    # Empty dictionary to store F1 scores for each threshold across all folds
    f1_scores_by_threshold = {}
    
    # Start training and validation phase
    for i in range(len(kfolds)):
        val_df = kfolds[i]
        train_dfs = [kfolds[j] for j in range(len(kfolds)) if j != i]
        train_df = pd.concat(train_dfs).reset_index(drop=True)
        # Drop the outcome variable final churn (leakage)
        X_train, y_train = train_df.drop('final_churn', axis=1), train_df['final_churn']
        X_val, y_val = val_df.drop('final_churn', axis=1), val_df['final_churn']
        
        # Apply SMOTE to the training data
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Train the model
        model.fit(X_train_smote, y_train_smote)
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Loop over probability thresholds
        for threshold in np.linspace(0.01, 0.99, 99):
            y_pred_threshold = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred_threshold)
            
            if threshold not in f1_scores_by_threshold:
                f1_scores_by_threshold[threshold] = []
            f1_scores_by_threshold[threshold].append(f1)
    
    # Calculate the average F1 score for each threshold
    average_f1_by_threshold = {threshold: np.mean(f1_scores) for threshold, f1_scores in f1_scores_by_threshold.items()}
    
    # Find the threshold with the highest average F1 score
    best_threshold = max(average_f1_by_threshold, key=average_f1_by_threshold.get)
    best_average_f1 = average_f1_by_threshold[best_threshold]
    
    # Store the parameters, best threshold, and best average F1 score
    average_performances.append((params, best_threshold, best_average_f1))
    
    print(f"Best average F1 score {best_average_f1} achieved with threshold {best_threshold} for parameters: {params}")

# After evaluating all parameter sets, we print the setting with the highest F1 score
best_params, best_threshold, best_f1_score = max(average_performances, key=lambda x: x[2])

print("Best parameters based on F1 score:", best_params)
print(f"Best threshold for this parameter set: {best_threshold}")
print(f"Best F1 score: {best_f1_score}")


############# Phase 5 - Evaluation #############

# If we skip the grid search, we can set best configuration manually
# best_params = {'C': 0.001, 'penalty': 'elasticnet', 'solver': 'saga', 'l1_ratio': 0.2}
# best_threshold = 0.77
# X_train, y_train = train_df.drop('final_churn', axis=1), train_df['final_churn']

# Prepare the test set
X_test, y_test = test_df.drop('final_churn', axis=1), test_df['final_churn']

# Train the model with the best parameters on the SMOTE train set
best_model = LogisticRegression(**best_params, max_iter=1000)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
best_model.fit(X_train_smote, y_train_smote)

# Predict probabilities on the test set
y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Convert probabilities to binary output based on the best threshold
y_test_pred = (y_test_pred_proba >= best_threshold).astype(int)

# Compute test metrics using the predictions
test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc_roc = roc_auc_score(y_test, y_test_pred_proba)
test_precision, test_recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
test_auc_pr = auc(test_recall, test_precision)
test_f1 = f1_score(y_test, y_test_pred)

print(f"Test Set Performance: Accuracy={test_accuracy}, AUC-ROC={test_auc_roc}, AUC-PR={test_auc_pr}, F1-Score={test_f1}")

# Create dataframe of true y and predicted y to plot the curves
data = {'y_test': y_test, 'y_test_pred_proba': y_test_pred_proba}
df = pd.DataFrame(data)
df.to_csv('probabilities_lr1103.csv', index=False)


############# Phase 6 - Regression coefficients and bootstrap #############

# Extract the regression coefficients
coefficients = best_model.coef_[0]
feature_names = X_train.columns

# Plot the regression coefficients
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
feature_importance_df = feature_importance_df.sort_values(by='Coefficient', ascending=False)

# Set the threshold for substantial importance to filter out less important features
threshold = 0.2
important_features_df = feature_importance_df[abs(feature_importance_df['Coefficient']) >= threshold]

# Plot feature importance
plt.figure(figsize=(10,8))
sns.barplot(x='Coefficient', y='Feature', data=important_features_df, color='dodgerblue')
plt.xlabel('Coefficient')
plt.ylabel('Feature')
plt.savefig("C:/Users/maaik/Documents/Business analytics and quantitative marketing/Seminar/Plots/Coefficients/Importance.pdf", format="pdf", bbox_inches='tight')
plt.show()

# Function to perform the bootstrap process for Logistic Regression coefficients
def bootstrap_coefficients(df, n_bootstraps, sample_size):
    # Store coefficients per feature for each bootstraps
    coefficients_summary = pd.DataFrame()

    for i in range(n_bootstraps):
        # Select unique_id values
        progress = ((i+1) / n_bootstraps) * 100  # Progress as a percentage
        print(f"Bootstrap progress: {progress:.2f}% ({i+1}/{n_bootstraps})")
        unique_ids = np.random.choice(df['unique_id'].unique(), size=sample_size, replace=False)
       
        # Select all rows for these unique_id values and drop the unique_id column
        df_bootstrap = df[df['unique_id'].isin(unique_ids)].drop(columns=['unique_id'])
       
        # Prepare the data for training
        X = df_bootstrap.drop(columns=['final_churn'])
        y = df_bootstrap['final_churn']
       
        # Apply SMOTE to the training data
        smote = SMOTE(random_state=i)
        X_res, y_res = smote.fit_resample(X, y)
       
        # Train the Logistic Regression model
        model = LogisticRegression(**best_params, max_iter=1000)
        model.fit(X_res, y_res)

        # Extract coefficients
        coefficients = model.coef_[0]
       
        column_name = f'Bootstrap_{i+1}'
        if i == 0:
            coefficients_summary = pd.DataFrame(coefficients, index=X.columns, columns=[column_name])
        else:
            coefficients_summary[column_name] = coefficients

    return coefficients_summary

# Apply the bootstrap function to the Logistic Regression. We obtain a dataframe 
# with each row representing a feature and each column representing a bootstrap
unique_ids = df_LR['unique_id'].unique()
coefficients_summary = bootstrap_coefficients(df_LR, 1000, int(0.1*len(unique_ids)))
mean_coefficients_summary = coefficients_summary.abs().mean(axis=1)

# Function to plot coefficient distribution
def plot_coefficient_distribution(coefficients_summary, feature_name, title='Coefficient Distribution'):
   
    # Extract the coefficients for a feature across all bootstraps
    feature_coefficients = coefficients_summary.loc[feature_name]
   
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(feature_coefficients, kde=True, bins=20)
   
    plt.title(title)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Plot distribution for a specific feature
plot_coefficient_distribution(coefficients_summary, 'remaing_months_contract', 'Distribution of Coefficients for Remaining Months in Contract')


############# Phase 7 - Post-hoc explanation via SHAP #############

import shap

# Calculate the SHAP values
explainer = shap.LinearExplainer(best_model, X_train_smote, int=1000, feature_perturbation="correlation_dependent")
shap_values = explainer.shap_values(X_train_smote)

# Plot the mean absolute SHAP values and the summary plot
shap.summary_plot(shap_values, X_train_smote, plot_type ="bar", show=False)
fig = plt.gcf()
fig.axes[0].set_xlabel('Mean Absolute SHAP Value')
fig.axes[0].set_ylabel('Feature')
plt.savefig("C:/Users/maaik/Documents/Business analytics and quantitative marketing/Seminar/Plots/Shap/LR_SHAP_Mean_Absolute.pdf", format="pdf", bbox_inches='tight')

shap.summary_plot(shap_values, X_train_smote, plot_type ="dot", show=False)
fig = plt.gcf()
fig.axes[0].set_xlabel('SHAP Value')
fig.axes[0].set_ylabel('Feature')
plt.savefig("C:/Users/maaik/Documents/Business analytics and quantitative marketing/Seminar/Plots/Shap/LR_SHAP_Summary_Plot.pdf", format="pdf", bbox_inches='tight')


