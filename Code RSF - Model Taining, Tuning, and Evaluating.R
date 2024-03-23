#RSF model tuning, training, and evaluating 

############### Phase 1 - Start Up ############### 

# Make working directory empty
rm(list=ls())

# Install packages
install.packages("randomForestSRC")
install.packages("readr")
install.packages("doParallel")

# Load packages
library(readr)
library(randomForestSRC)
library(dplyr)
library(purrr)
library(survival)
library(caret)
library(pROC) 
library(doParallel)
library(foreach)
library(dplyr)
library(reshape2)

# Load the DataFrame
df_rsf <- read_csv("/Users/476366sg/Downloads/df_rsf_final_20240308.csv") #Your own file path

############### Phase 2 - Create a Representative subset used for hyperparameter tuning ############### 

# Define a function to make a subset of the data neccesary for hyperparameter tuning.
f_subset <- function(df, percentage, number_unique_ids) {
  df <- as.data.frame(df)
  
  # Identify unique_ids with at least one churn event and those without
  churners_ids <- unique(df[df$final_churn == 1, ]$unique_id)
  non_churners_ids <- setdiff(unique(df$unique_id), churners_ids)
  
  # Calculate the number of churner unique_ids based on the percentage
  number_churners = round(number_unique_ids * percentage / 100)
  
  # Adjust in case the desired number of churners exceeds available churners
  number_churners = min(number_churners, length(churners_ids))
  
  # Determine the number of non-churners needed to meet the total unique_id count
  number_non_churners = number_unique_ids - number_churners
  
  # Sample unique_ids from churners and non-churners
  sampled_churners_ids <- sample(churners_ids, number_churners)
  sampled_non_churners_ids <- sample(non_churners_ids, number_non_churners)
  
  # Combine sampled IDs and filter the dataframe
  sampled_ids <- c(sampled_churners_ids, sampled_non_churners_ids)
  subset_df <- df[df$unique_id %in% sampled_ids, ]
  
  return(subset_df)
}

# Count unique values in the unique_id column of df_rsf
number_of_unique_values <- length(unique(df_rsf$unique_id))
number_churners_ids <- length(unique(df_rsf[df_rsf$final_churn == 1, ]$unique_id))
percentage_churn <- 100 * (number_churners_ids/number_of_unique_values)

# Define a representative subset
df_rsf_subset <- f_subset(df_rsf, percentage_churn, 0.3 * number_of_unique_values)

############### Phase 3 - Train/Test Sets ############### 

# Define a function to devide the DataFrame into train, test, and validation
f_train_test_kfold <- function(df, k = 3) {
  # Ensure df is a data frame
  df <- as.data.frame(df)
  
  # Shuffle groups
  set.seed(123) # For reproducibility
  groups <- df %>% group_by(unique_id) %>% group_split()
  groups <- sample(groups)
  
  # Calculate the target number of rows for the train and test splits
  total_rows <- nrow(df)
  train_target <- total_rows * 0.8
  
  # Initialize counters and containers for the splits
  train_rows <- 0
  train_groups <- list()
  test_groups <- list()
  
  # Assign groups to train or test splits
  for (group in groups) {
    if (train_rows + nrow(group) <= train_target) {
      train_groups <- append(train_groups, list(group))
      train_rows <- train_rows + nrow(group)
    } else {
      test_groups <- append(test_groups, list(group))
    }
  }
  
  # Concatenate groups into data frames for train and test sets
  train_df <- bind_rows(train_groups) 
  test_df <- bind_rows(test_groups) 
  
  #Store relevant data that is later used for creating survival curves survival curves
  time_to_event_data <- test_df[, c('time_to_event', 'unique_id')]
  unique_id_mapping <- data.frame(row_number = row.names(test_df), unique_id = test_df$unique_id)
  
  # Remove specified columns from train and test sets, which can lead to data leakage, or is can not be handled by the model. 
  cols_to_remove <- c('unique_id', 'period') 
  train_df <- select(train_df, -all_of(cols_to_remove))
  test_df <- select(test_df, -all_of(cols_to_remove))  
  
  # Split the train_df into k-fold sets
  set.seed(27) # Shuffle again to randomize for k-fold
  train_groups <- sample(train_groups)
  fold_size <- floor(nrow(train_df) / k)
  
  # Initialize counters and containers for k-folds
  current_fold_rows <- 0
  current_fold_groups <- list()
  kfolds <- list()
  
  for (group in train_groups) {
    if (current_fold_rows + nrow(group) <= fold_size) {
      current_fold_groups <- append(current_fold_groups, list(group))
      current_fold_rows <- current_fold_rows + nrow(group)
    } else {
      kfolds <- append(kfolds, list(bind_rows(current_fold_groups)))
      current_fold_groups <- list(group)
      current_fold_rows <- nrow(group)
    }
  }
  
  # Ensure the last fold is added
  if (length(current_fold_groups) > 0) {
    kfolds <- append(kfolds, list(bind_rows(current_fold_groups)))
  }
  
  # Adjust if the last fold is significantly smaller than the rest
  if (length(kfolds) > k) {
    # Merge the last two folds if more than k folds are created
    kfolds[[length(kfolds) - 1]] <- bind_rows(kfolds[[length(kfolds) - 1]], kfolds[[length(kfolds)]])
    kfolds <- kfolds[-length(kfolds)]
  }
  
  # Remove specified columns from each k-fold
  kfolds <- lapply(kfolds, function(x) select(x, -all_of(cols_to_remove)))
  
  return(list(train_df = train_df, test_df = test_df, kfolds = kfolds, unique_id_mapping = unique_id_mapping, time_to_event_data = time_to_event_data))
}

# Use the function to create a train and test test, and split the train set into 3 folds. 
result <- f_train_test_kfold(df = df_rsf_subset, k = 3)
train_set <- result$train_df
test_set <- result$test_df
kfolds <- result$kfolds

# Check the percentage of churners for each made DataFrame
sum(df_rsf_subset$final_churn)/nrow(df_rsf_subset)
sum(train_set$final_churn)/nrow(train_set)
sum(test_set$final_churn)/nrow(test_set)
sum(df_rsf$final_churn)/nrow(df_rsf)
sum(kfolds[[1]]$final_churn)/nrow(kfolds[[1]])
sum(kfolds[[2]]$final_churn)/nrow(kfolds[[2]])
sum(kfolds[[3]]$final_churn)/nrow(kfolds[[3]])

############### Phase 4 - Hyper parameter tuning using k-Fold Cross Validation ############### 

# Define the parameter grid
param_combinations <- expand.grid(
  ntree = c(200,400,600,1200),  
  mtry = c(round(sqrt(ncol(train_set)))  # Square root of variables
           ,round(ncol(train_set)/3)  # A third of variables
           ,round(ncol(train_set)/2)),  # Half of the variables 
  nodesize = c(sqrt(nrow(train_set))/10  # Smaller nodesize
               ,sqrt(nrow(train_set))/5,  # Larger nodesize
               sqrt(nrow(train_set))/20),  # Even smaller, for finer granularity
  nsplit = c(2 ,5 ,10) 
)

# Take a subset of the parameter grid for faster computation (run this on 10 computers all with 11 different settings)
param_combinations <- param_combinations[36,]

# Define functions to compute the evaluation metrics
# Define function to calculate F1 score
calculate_f1_score <- function(predicted, actual) {
  tp <- sum((predicted == 1) & (actual == 1))
  fp <- sum((predicted == 1) & (actual == 0))
  fn <- sum((predicted == 0) & (actual == 1))
  precision <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  recall <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  f1_score <- ifelse((precision + recall) > 0, 2 * ((precision * recall) / (precision + recall)), 0)
  return(f1_score)
}

# Define function to calculate accuracy 
calculate_accuracy <- function(predicted, actual) {
  TP <- sum(predicted == 1 & actual == 1)
  TN <- sum(predicted == 0 & actual == 0)
  FP <- sum(predicted == 1 & actual == 0)
  FN <- sum(predicted == 0 & actual == 1)
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  return(accuracy)
}

# Define function to calculate precision 
calculate_precision <- function(predicted, actual) {
  TP <- sum(predicted == 1 & actual == 1)
  FP <- sum(predicted == 1 & actual == 0)
  precision <- ifelse((TP + FP) > 0, TP / (TP + FP), 0) # Avoid division by zero
  return(precision)
}

# Define function to calculate recall 
calculate_recall <- function(predicted, actual) {
  TP <- sum(predicted == 1 & actual == 1)
  FN <- sum(predicted == 0 & actual == 1)
  recall <- ifelse((TP + FN) > 0, TP / (TP + FN), 0) # Avoid division by zero
  return(recall)
}

# Register parallel backend to use multiple cores (
no_cores <- detectCores() - 1
registerDoParallel(cores=no_cores)

# Store all the performances of each parameter setting in all_performances 
all_performances <- foreach(i = 1:nrow(param_combinations), .combine = 'rbind', .packages = c("randomForestSRC")) %dopar% {
  params <- param_combinations[i, ]
  
  # Initialize structures to store churn probabilities and F1 scores
  churn_probabilities_list <- vector("list", length = length(kfolds))
  f1_scores_by_threshold <- list()
  
  # Perform k-fold cross-validation
  for (fold in 1:length(kfolds)) {
    validation_df <- kfolds[[fold]]
    training_dfs <- kfolds[-fold]
    training_df <- do.call(rbind, training_dfs)
    
    # Generate case weights to handle the data imbalance
    case_weights <- randomForestSRC:::make.wt(training_df$final_churn)
    
    # Fit the RSF model
    fit <- rfsrc.fast(Surv(time_to_event, final_churn) ~ ., 
                      data = training_df, 
                      ntree = params$ntree, 
                      mtry = params$mtry, 
                      nodesize = params$nodesize, 
                      nsplit = params$nsplit, 
                      importance = FALSE, 
                      save.memory = TRUE, 
                      forest = TRUE, 
                      perf.type = "none", 
                      case.wt = case_weights)
    
    # Predict and extract churn probabilities
    predictions <- predict(fit, newdata = validation_df)
    churn_prob <- predictions$predicted
    churn_prob <- as.vector(churn_prob)
    
    # Calculate F1 scores for this fold for a range of thresholds
    actual_classes <- validation_df$final_churn 
    for (threshold in seq(0, 1, by=0.01)) {
      predicted_classes <- ifelse(churn_prob > threshold, 1, 0)
      f1_score <- calculate_f1_score(predicted_classes, actual_classes) 
      
      if (!is.list(f1_scores_by_threshold[[as.character(threshold)]])) {
        f1_scores_by_threshold[[as.character(threshold)]] <- c()
      }
      f1_scores_by_threshold[[as.character(threshold)]] <- c(f1_scores_by_threshold[[as.character(threshold)]], f1_score)
    }
  }
  
  # Calculate average F1 score for each threshold
  average_f1_by_threshold <- sapply(f1_scores_by_threshold, mean)
  
  # Find the best threshold
  best_threshold <- which.max(average_f1_by_threshold)
  best_f1_score <- max(average_f1_by_threshold)
  
  # Return the predictions, best threshold, and F1 score for this parameter set
  list(params = params, 
       best_threshold = names(best_threshold),
       best_f1_score = best_f1_score)
}

############### Phase 5 - Find the best hyperparameter set and optimal threshold by evluating F1-Scores ############### 

# Initialize variables to track the best performance
best_index <- NULL
highest_f1_score <- -Inf

# Loop over all performances to find the best F1 score and its index
for (i in 1:nrow(all_performances)) {
  current_f1_score <- as.numeric(all_performances[i,]$best_f1_score)
  
  if (current_f1_score > highest_f1_score) {
    highest_f1_score <- current_f1_score
    best_index <- i
  }
}

# Extract the best performance details
best_performance <- all_performances[best_index, ]
best_params <- best_performance$params
best_threshold <- best_performance$best_threshold
best_f1_score <- best_performance$best_f1_score

# Print the best hyperparameters, threshold, and F1 score
cat("Best Hyperparameters:\n")
print(best_params)
cat(sprintf("Best Threshold: %s\n", best_threshold))
cat(sprintf("Best F1 Score: %f\n", best_f1_score))

############### Phase 7 - Testing Evaluation ############### 

# Extract the best parameter setting
best_ntree <- best_params$ntree
best_mtry <- best_params$mtry
best_nodesize <- best_params$nodesize
best_nsplit <- best_params$nsplit 

# Define case weights
case_weights <- randomForestSRC:::make.wt(train_set$final_churn)

# Fit the model on the training set using the best hyperparameter settings
best_model <- rfsrc.fast(Surv(time_to_event, final_churn) ~ ., 
                         data = train_set, 
                         ntree = best_ntree, 
                         mtry = best_mtry, 
                         nodesize = as.numeric(best_nodesize), 
                         nsplit = best_nsplit, 
                         importance = FALSE,
                         save.memory=TRUE, 
                         forest = TRUE,
                         perf.type = "none",
                         case.wt = case_weights
)

# Make predictions on the test set
predictions_final <- predict(best_model, newdata = test_set)

# Compute churn probabilities for the test set
churn_prob_normal <- as.vector(predictions_final$predicted)

# Function to convert survival outcomes to binary classification based on a threshold such that it can be used for performance metrics
binary_classification_threshold <- function(predicted_risk, threshold) {
  as.numeric(predicted_risk > threshold)
}

# Apply the best threshold to get binary predictions
predicted_classes_final_normal <- binary_classification_threshold(churn_prob_normal, best_threshold)

# Actual binary churn outcomes the test data
actual_classes_final <- test_set$final_churn

# Compute the test F1 score using the predicted classes and the actual classes
f1_score_final_normal <- calculate_f1_score(predicted_classes_final_normal, actual_classes_final)
f1_score_final_normal

##################### Phase 8 - Compute all performance metrics ##########################

# Compute accuracy score
accuracy_final <- calculate_accuracy(predicted_classes_final_normal,actual_classes_final)

# Compute AUC-ROC value
roc_result <- roc(response = actual_classes_final, predictor = churn_prob_normal)
auc_roc <- auc(roc_result)

# Compute AUC-PR value
#install.packages("PRROC")
library(PRROC)

# Compute the Precision-Recall curve
pr_result <- pr.curve(scores.class0 = churn_prob_normal, weights.class0 = actual_classes_final, curve = TRUE)

# Extract the AUC-PR
auc_pr <- pr_result$auc.integral

# Print the AUC-PR
print(paste("AUC-PR:", auc_pr))

# Create Confusion Matrix
# Ensure both predicted and actual vectors are factors
predicted_classes_final_normal <- as.factor(predicted_classes_final_normal)
actual_classes_final <- as.factor(actual_classes_final)

# Create the confusion matrix
conf_matrix <- confusionMatrix(predicted_classes_final_normal, actual_classes_final)

# Print the confusion matrix
print(conf_matrix)
