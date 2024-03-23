#CALCULATING STOCHASTIC SHAPLEY VALUES

install.packages("devtools")
library(devtools)
devtools::install_github("nredell/shapFlex")
library(shapFlex)

# The Stochastic Shapley Values can be computed using the trained RSF model with the optimal parameter setting. A subset of 0.05% the data is used since the method can't handle large datasets. 

# Predict function used to calculate shap values
# A prediction function that takes a trained model and a datframe of model features, and returns a 1-column data.frame of predictions for each instance to be explained.
predict_function <- function(model, data) {
  prediction <- predict(model, newdata = data)
  if (inherits(prediction, "rfsrc")) {
    # Extract relevant information from the rfsrc prediction and convert it into a data frame.
    data_pred <- data.frame("y_pred" = prediction$predicted)
  } else {
    # Handle other types of predictions
    data_pred <- data.frame("y_pred" = prediction)
  }
  return(data_pred)
}

# Create a subset of 0.05% of the data frame, because of limited computational power. 
df_rsf_subset_shap <- f_subset(df_rsf, percentage_churn, 0.0005 * number_of_unique_values)

# Remove outcome variables, and variables that are usually removed in the train and test split.
cols_to_remove <- c('final_churn', 'time_to_event', 'period', 'unique_id')

# Create a dataframe that only contains features
X_subset_df_shap <- select(df_rsf_subset_shap, -all_of(cols_to_remove))
X_total_df <-  select(df_rsf, -all_of(cols_to_remove))

explain <- X_subset_df_shap # Compute Shapley feature-level predictions for sample_size instances

reference <- X_total_df  # A reference population to compute the baseline prediction.

sample_size <- 60  # Number of Monte Carlo samples

target_features <- colnames(X_subset_df_shap) # The set of target features

# Compute the stochastic shapley values for each observation using the shapFlex package. The model used is the best RSF model with the optimal hyperparameter settings.  
explained_non_causal <- shapFlex::shapFlex(explain = explain,
                                           reference = reference,
                                           model = best_model,
                                           predict_function = predict_function,
                                           target_features = target_features,
                                           sample_size = sample_size)


# Compute the mean absolute shapley values for each feature in the model
explained_non_causal_sum <- explained_non_causal %>%
  dplyr::group_by(feature_name) %>%
  dplyr::summarize("shap_effect" = mean(abs(shap_effect), na.rm = TRUE))


#Save all the stochastic shapley values for each observation, for each feature in a csv file. This file can be used to create plots in python that are similar to the plots of the other models. 

# All feature values:
X_subset_df_shap <- as.data.frame(X_subset_df_shap)
all_colnames <- colnames(X_subset_df_shap)

# All shap values: 
individual_shap_values <- explained_non_causal$shap_effect
shap_matrix <- matrix(individual_shap_values, nrow=length(X_subset_df_shap), byrow = FALSE)
df_shap_values <- as.data.frame(shap_matrix) 
colnames(df_shap_values) <- all_colnames

# Export these two dataframes so that we can import it into Python and generate summary plots, and impact plots there
write.csv(X_subset_df_shap, "C:/Users/476366sg/Downloads/df_rsf_final_20240308/X_subset_df_shap.csv", row.names = FALSE)
write.csv(df_shap_values, "C:/Users/476366sg/Downloads/df_rsf_final_20240308/df_shap_values.csv", row.names = FALSE)
