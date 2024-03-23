%reset

import pandas as pd

# Import dataframes from R (the feature values and the shap values for RSF)
filepath_X_train_shap = "C:/Users/maaik/Documents/Business analytics and quantitative marketing/Seminar/X_subset_df_shap.csv"
X_train_shap =  pd.read_csv(filepath_X_train_shap)

filepath_df_shap_values = "C:/Users/maaik/Documents/Business analytics and quantitative marketing/Seminar/df_shap_values.csv"
df_shap_values =  pd.read_csv(filepath_df_shap_values)

# Extract the column names
colnames_shap = X_train_shap.columns

# We need to change the type of X_train_shap and df_shap_values from pandas array to numpy array. 
# This is in order for shap.summary_plot to work.
X_train_shap = X_train_shap.to_numpy()
df_shap_values = df_shap_values.to_numpy()

# Generate Shapley Values summary plot
import shap
shap.summary_plot(df_shap_values, X_train_shap, feature_names = colnames_shap)
