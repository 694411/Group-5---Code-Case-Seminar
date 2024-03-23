%reset -f


############# Phase 1 - Data Set #############

path_xgb = '/Users/julianjager/20240307_xgb.csv'
df_xgb = pd.read_csv(path_xgb)

%reset -f
#pip install --upgrade pandas
pip install xgboost
pip install imbalanced-learn

!pip uninstall scikit-learn --yes
!pip install scikit-learn==1.2.2


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.metrics import precision_recall_curve, auc
from xgboost import plot_importance
import matplotlib.pyplot as plt


############# Phase 1 - Data Set #############

# Define the start and end of the subset we want to run
a = 901
b = 972

############# Phase 2 - Train, Test & Validation #############

# Define a function for k-fold cross validation
def f_train_test_kfold(df, k=3):
    # Group by ID and shuffle
    groups = [group for _, group in df.groupby('unique_id')]
    np.random.shuffle(groups)

    # Calculate the target number of rows for the train and test splits
    total_rows = len(df)
    train_target = total_rows * 0.8

    # Initialize counters and containers for the splits
    train_rows = 0
    train_groups = []
    test_groups = []

    # Assign groups to train or test splits
    for group in groups:
        if train_rows + len(group) <= train_target:
            train_groups.append(group)
            train_rows += len(group)
        else:
            test_groups.append(group)

    # Concatenate groups into DataFrames for train and test sets
    train_df = pd.concat(train_groups).reset_index(drop=True)
    test_df = pd.concat(test_groups).reset_index(drop=True)

    # Split the train_df into k-fold sets
    kfolds = []
    np.random.shuffle(train_groups)  
    fold_size = len(train_df) // k  

    # Initialize counters and containers for k-folds
    current_fold_rows = 0
    current_fold_groups = []

    for group in train_groups:
        if current_fold_rows + len(group) <= fold_size:
            current_fold_groups.append(group)
            current_fold_rows += len(group)
        else:
            # Once a fold reaches its target size, add it to the kfolds and start a new fold
            kfolds.append(pd.concat(current_fold_groups).reset_index(drop=True))
            current_fold_groups = [group]  
            current_fold_rows = len(group)

    # Ensure the last fold is added (it may be smaller than the others)
    if current_fold_groups:
        kfolds.append(pd.concat(current_fold_groups).reset_index(drop=True))

    # Adjust if the last fold is significantly smaller than the rest
    if len(kfolds) > k:
        # Merge the last two folds if more than k folds are created
        kfolds[-2] = pd.concat([kfolds[-2], kfolds[-1]]).reset_index(drop=True)
        kfolds.pop()

    return train_df, test_df, kfolds

# Usage of the function
train_df, test_df, kfolds = f_train_test_kfold(df_xgb, k=3)

# Drop ID-columns
train_df = train_df.drop(columns=['unique_id', 'period'])
test_df = test_df.drop(columns=['unique_id', 'period'])
for i in range(len(kfolds)):
    kfolds[i] = kfolds[i].drop(columns=['unique_id', 'period'])

############# Phase 3 - Parameter Grid #############

# Define the parameter grid
param_grid = {
    'max_depth': [7, 9, 12],
    'learning_rate': [0.01, 0.025, 0.05],
    'n_estimators': [400, 600, 800],
    'subsample': [0.7, 0.9],
    'min_child_weight': [1, 5],
    'reg_lambda': [0, 1, 2],  
    'reg_alpha': [0.5, 1, 2]  
}

# Function to get all combinations of parameters
def get_param_combinations(param_grid):
    from itertools import product
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    return combinations

# Get all combinations of parameters
param_combinations = get_param_combinations(param_grid)

# Define a subset, if needed
param_combinations = param_combinations[a:b]

############# Phase 4 - K-Fold Cross Validation #############

# Initialize a list to store performance of each parameter set
performance_tracking = []

#pip install --upgrade scikit-learn
#pip install --upgrade imbalanced-learn  # This is the package name for imblearn
#pip install --upgrade threadpoolctl

for params in param_combinations:
    print("Evaluating parameters:", params)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **params)
    
    # Initialize a dictionary to store F1 scores for each threshold across all folds
    f1_scores_by_threshold = {}
    
    for i, val_df in enumerate(kfolds):
        print(f"Processing fold {i+1}/{len(kfolds)}")
        train_dfs = [kfolds[j] for j in range(len(kfolds)) if j != i]
        train_df = pd.concat(train_dfs).reset_index(drop=True)
        
        X_train, y_train = train_df.drop('final_churn', axis=1), train_df['final_churn']
        X_val, y_val = val_df.drop('final_churn', axis=1), val_df['final_churn']
        
        # Apply SMOTE to the training data
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Train the model
        model.fit(X_train_smote, y_train_smote)
        
        # Predict probabilities
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Loop over thresholds
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
    
    # Store the parameters, best threshold, and best F1 score
    performance_tracking.append((params, best_threshold, best_average_f1))
    
    print(f"Best average F1 score {best_average_f1} achieved with threshold {best_threshold} for parameters: {params}")

# After evaluating all parameter sets, find the set with the highest F1 score
best_params, best_threshold, best_f1_score = max(performance_tracking, key=lambda x: x[2])

# Print the results
print("Best parameter set based on F1 score:", best_params)
print(f"Best threshold for this parameter set: {best_threshold}")
print(f"Best F1 score: {best_f1_score}")

############# Phase 5 - Evaluation #############

# Prepare the test set
X_test, y_test = test_df.drop('final_churn', axis=1), test_df['final_churn']

# Initialize and train the model with the best parameters on the SMOTE train set
best_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **best_params)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
best_model.fit(X_train_smote, y_train_smote)

# Predict probabilities on the test set
y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Convert probabilities to binary output based on the best threshold
y_test_pred = (y_test_pred_proba >= best_threshold).astype(int)

# Compute test metrics using the predictions adjusted by the best threshold
test_accuracy = accuracy_score(y_test, y_test_pred)
test_auc_roc = roc_auc_score(y_test, y_test_pred_proba)
test_precision, test_recall, _ = precision_recall_curve(y_test, y_test_pred_proba)
test_auc_pr = auc(test_recall, test_precision)
test_f1 = f1_score(y_test, y_test_pred)

print(f"Test Set Performance: Accuracy={test_accuracy}, AUC-ROC={test_auc_roc}, AUC-PR={test_auc_pr}, F1-Score={test_f1}")

# Extract the probabilities
data = {'y_test': y_test, 'y_test_pred_proba': y_test_pred_proba}
df = pd.DataFrame(data)
df.to_csv('20240314_probabiilities_xgb.csv', index=False)

############# Phase 6 - Bootstrap #############

# Define the best parameter grid
param_grid = {
    'max_depth': 12,  
    'learning_rate': 0.025,
    'n_estimators': 800,  
    'subsample': 0.9,  
    'min_child_weight': 1,  
    'reg_lambda': 2,  
    'reg_alpha': 0.5  
}

df_xgb = df_xgb.drop('period', axis=1)

# Function to perform the bootstrap process
def bootstrap_shap(df, n_bootstraps, sample_size, shap_size):
    # Store mean absolute SHAP values per feature across bootstraps
    shap_values_summary = pd.DataFrame()

    for i in range(n_bootstraps):
        # Randomly choose 1000 unique_id values
        progress = ((i+1) / n_bootstraps) * 100  # Progress as a percentage
        print(f"Bootstrap progress: {progress:.2f}% ({i+1}/{n_bootstraps})")
        unique_ids = np.random.choice(df['unique_id'].unique(), size=sample_size, replace=False)
        
        # Select all rows for these unique_id values and drop the unique_id column
        df_bootstrap = df[df['unique_id'].isin(unique_ids)].drop(columns=['unique_id'])
        
        # Prepare the data for training
        X = df_bootstrap.drop(columns=['final_churn'])
        y = df_bootstrap['final_churn']
        
        # Train the XGBoost model
        smote = SMOTE(random_state=i)
        X_res, y_res = smote.fit_resample(X, y)
        
        model = xgb.XGBClassifier(**param_grid, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_res, y_res)

        # Compute SHAP values for 100 random observations
        explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
        X_random_sample = X_res.sample(n=shap_size, random_state=i)
        shap_values = explainer.shap_values(X_random_sample)
        
        # Define the binary features
        binary_features = ['fixed_contract_electricity_ind',
                           'electricity_ind',
                           'fixed_contract_gas_ind',
                           'gas_ind',
                           'changed_policy_ind',
                           'no_fixed_contracts_offered_ind',
                           'spring_ind',
                           'autumn_ind',
                           'winter_ind',
                           'price_cap_ind',
                           'fine_for_churning_ind',
                           'saleschannel_auction_ind',
                           'saleschannel_ownwebsite_ind',
                           'saleschannel_pricecomparisonwebsite_ind',
                           'solar_panels_ind']

        # Define the integer features
        integer_features = ['remaining_months_contract']
        
        # Find all indices that are added with SMOTE
        for feature in binary_features:
            feature_index = X_res.columns.get_loc(feature)
            non_binary_indices = X_res.index[(X_res[feature] != 0) & (X_res[feature] != 1)]
            indices_to_remove.update(non_binary_indices)

        for feature in integer_features:
            feature_index = X_res.columns.get_loc(feature)
            non_integer_indices = X_res.index[X_res[feature].apply(lambda x: x != np.floor(x))]
            indices_to_remove.update(non_integer_indices)

        # Convert indices_to_remove to a sorted list to ensure row removal is handled correctly
        sorted_indices_to_remove = sorted(list(indices_to_remove), reverse=True)
        
        # Remove observations from SMOTE
        X_res_filtered = X_res.drop(sorted_indices_to_remove)
        shap_values_filtered = np.delete(shap_values, sorted_indices_to_remove, axis=0)
        
        mean_abs_shap = np.abs(shap_values_filtered).mean(axis=0)
        
        column_name = f'Bootstrap_{i+1}'
        if i == 0:
            shap_values_summary = pd.DataFrame(mean_abs_shap, index=X.columns, columns=[column_name])
        else:
            shap_values_summary[column_name] = mean_abs_shap

    return shap_values_summary

# Apply the bootstrap
df_xgb_bootstrap = bootstrap_shap(df_xgb, 1000, round(df_xgb['unique_id'].nunique()*0.1), 1000)
df_mean_shap_per_feature = df_xgb_bootstrap.mean(axis=1)

def plot_shap_distribution(shap_values_summary, feature_name, title='SHAP Value Distribution'):

    # Check if the feature exists in the DataFrame
    if feature_name not in shap_values_summary.index:
        print(f"Feature '{feature_name}' not found in SHAP values summary.")
        return

    # Extract the SHAP values for the specified feature across all bootstraps
    feature_shap_values = shap_values_summary.loc[feature_name]
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(feature_shap_values, kde=True, bins=20)
    
    plt.title(title)
    plt.xlabel('Mean Absolute SHAP Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

# Apply the function
plot_shap_distribution(df_xgb_bootstrap, 'time_with_coolblue', 'Distribution of SHAP Values for Remaining Months Contract')

# Save to csv
df_xgb_bootstrap.to_csv('20240312_xgb_shap_bootstrap.csv', index=False)

############# Phase 7 - Plots #############

# Prepare the data for training
X = df_xgb.drop(columns=['final_churn'])
y = df_xgb['final_churn']
        
# Train the XGBoost model
smote = SMOTE(random_state=1)
X_res, y_res = smote.fit_resample(X, y)
        
model = XGBClassifier(**param_grid, use_label_encoder=False, eval_metric='logloss')
model.fit(X_res, y_res)

# SHAP Explanation object with TreeExplainer
explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
shap_values = explainer.shap_values(X_res)

# Make a copy
X_res_copy = X_res.copy()
shap_values_copy = shap_values.copy()

# Delete SMOTE features
binary_features = ['fixed_contract_electricity_ind',
                   'electricity_ind',
                   'fixed_contract_gas_ind',
                   'gas_ind',
                   'changed_policy_ind',
                   'no_fixed_contracts_offered_ind',
                   'spring_ind',
                   'autumn_ind',
                   'winter_ind',
                   'price_cap_ind',
                   'fine_for_churning_ind',
                   'saleschannel_auction_ind',
                   'saleschannel_ownwebsite_ind',
                   'saleschannel_pricecomparisonwebsite_ind',
                   'solar_panels_ind']
integer_features = ['remaining_months_contract']

# Identifying indices to remove based on binary feature criteria
non_binary_indices = X_res_copy[binary_features].apply(lambda x: ((x != 0) & (x != 1)).any(), axis=1)
binary_to_remove = non_binary_indices[non_binary_indices].index

# Identifying indices to remove based on integer feature criteria
non_integer_indices = X_res_copy[integer_features].applymap(lambda x: x != np.floor(x)).any(axis=1)
integer_to_remove = non_integer_indices[non_integer_indices].index

# Combine the indices and remove duplicates
indices_to_remove = binary_to_remove.union(integer_to_remove)

# Remove these observations from X_res_copy
X_res_filtered = X_res_copy.drop(index=indices_to_remove)

# Convert DataFrame indices to remove into positions for numpy array
positions_to_remove = [X_res_copy.index.get_loc(idx) for idx in indices_to_remove]

# Remove these observations from shap_values_copy
shap_values_filtered = np.delete(shap_values_copy, positions_to_remove, axis=0)

# Make summary plot
shap.summary_plot(shap_values_filtered, X_res_filtered, plot_type="dot", show=False)
plt.xlabel("Mean Absolute SHAP Value")
plt.tight_layout()
plt.savefig("20240317_shap_summary_plot.pdf")
plt.close()

# Bar plot 
shap.summary_plot(shap_values_filtered, X_res_filtered, plot_type="bar", show=False)
plt.xlabel("Mean Absolute SHAP Value")
plt.tight_layout()
plt.savefig("20240317_shap_bar_plot.pdf")
plt.close()

# Generate the SHAP summary plot with wider format
shap.summary_plot(shap_values_filtered, X_res_filtered, plot_type="dot", show=False)
plt.xlabel("Mean Absolute SHAP Value")
fig = plt.gcf()
fig.set_size_inches(fig.get_size_inches()[0]*2, fig.get_size_inches()[1])  
plt.tight_layout()
plt.savefig("20240317_shap_summary_plot_2.pdf")
plt.close()

############# Phase 7 - Plots Scatter #############

import shap

df_xgb = df_xgb.drop(columns=['unique_id', 'period'])

# Prepare the data for training
X = df_xgb.drop(columns=['final_churn'])
y = df_xgb['final_churn']

df_xgb_2 = df_xgb.drop(columns = 'final_churn')
df_xgb['final_churn']

# Train the XGBoost model
smote = SMOTE(random_state=1)
X_res, y_res = smote.fit_resample(X, y)

# Sample a random subset of 1000 observations
sample_indices = np.random.choice(X_res.index, size=50000, replace=False)
X_sample = X_res.loc[sample_indices]
y_sample = y_res.loc[sample_indices]

# Continue with the XGBoost model training on the sampled data
model = XGBClassifier(**param_grid, use_label_encoder=False, eval_metric='logloss')
model.fit(X_sample, y_sample)

# Use TreeExplainer to explain the model's predictions on the sampled data
explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
shap_values_sample = explainer.shap_values(X_sample)

# Create a SHAP Explanation object for the sampled data
shap_explanation_sample = shap.Explanation(values=shap_values_sample, data=X_sample.values, feature_names=X_sample.columns)

binary_features = ['fixed_contract_electricity_ind',
                   'electricity_ind',
                   'fixed_contract_gas_ind',
                   'gas_ind',
                   'changed_policy_ind',
                   'no_fixed_contracts_offered_ind',
                   'spring_ind',
                   'autumn_ind',
                   'winter_ind',
                   'price_cap_ind',
                   'fine_for_churning_ind',
                   'saleschannel_auction_ind',
                   'saleschannel_ownwebsite_ind',
                   'saleschannel_pricecomparisonwebsite_ind',
                   'solar_panels_ind']

integer_features = ['remaining_months_contract']

for feature in binary_features:
    feature_index = X_sample.columns.get_loc(feature)
    non_binary_indices = (X_sample[feature] != 0) & (X_sample[feature] != 1)
    shap_explanation_sample.values[non_binary_indices, feature_index] = np.nan 
    
for feature in integer_features:
    # Get the column index for the feature in X_sample
    feature_index = X_sample.columns.get_loc(feature)
    
    non_integer_indices = X_sample[feature].apply(lambda x: x != np.floor(x))
    
    # Set SHAP values for non-integer indices to np.nan
    shap_explanation_sample.values[non_integer_indices, feature_index] = np.nan

df_xgb2 = df_xgb.drop(columns='final_churn')

# Get the current figure and axes objects.
for feature_name in df_xgb2.columns:
    # Generate SHAP scatter plot for the current feature
    print(feature_name)
    shap.plots.scatter(shap_explanation_sample[:, feature_name], show=False, hist=False)
    plt.ylabel(f'SHAP Value')
    plt.savefig(f'SHAP_Scatter_{feature_name}.pdf')
    
    plt.close()  # Close the plot to prevent display issues in the loop
    
import matplotlib.pyplot as plt
shap.plots.scatter(shap_explanation_sample[:, 'time_with_coolblue'], show=False, hist=False)
plt.ylabel(f'SHAP Value for {feature_name}')
plt.show()

shap.plots.scatter(ylabel=)
