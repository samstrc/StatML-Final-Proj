# MACHINE LEARNING FINAL PROJECT: Customer Churn Data
# Ava, Sam, Jordan, Taylor

# EDA + PREPROCESSING ONLY===============================================================================
# NOTE: No CreateCV() here. No modeling yet.
# We only explore, clean, and define recipes for later use.

# Load Libraries
packages <- c("tidyverse", "caret", "recipes", "janitor", "pROC", "themis") # caret for cross validation, recipes for pipelines, janitor for cleaning, pROC for metrics, and themis for 
to_install <- packages[!packages %in% installed.packages()[, "Package"]]
if (length(to_install)) install.packages(to_install, repos = "https://cloud.r-project.org")

lapply(packages, library, character.only = TRUE)

# Import Dataset + Clean Names
data <- read_csv(
  "https://raw.githubusercontent.com/samstrc/StatML-Final-Proj/refs/heads/main/customer_churn.csv",
  show_col_types = FALSE
) %>%
  janitor::clean_names()

glimpse(data)
summary(data)
names(data)

# Drop ID Column
df <- data %>%
  select(-customer_id)

# Convert categoricals to factors, rename churn to resp, and move resp to first column
df <- df %>%
  mutate(
    resp          = factor(churn, levels = c(0, 1), labels = c("No", "Yes")),
    country       = factor(country),
    gender        = factor(gender),
    credit_card   = factor(credit_card),    
    active_member = factor(active_member)    
  ) %>%
  select(resp, everything(), -churn)

table(df$resp)

# BASIC EDA
# Missing values
colSums(is.na(df))

# Check class imbalance ------------
ggplot(df, aes(x = resp)) +
  geom_bar(aes(y = after_stat(count/sum(count))), fill = "steelblue") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(y = "Proportion", x = "Churn Status") +
  theme_minimal()

table(df$resp)
prop.table(table(df$resp))

# Numeric Histograms ------------------
df %>%
  select(where(is.numeric)) %>%
  pivot_longer(cols = everything(),
               names_to = "variable",
               values_to = "value") %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 30, fill = "#6699CC") +
  facet_wrap(~ variable, scales = "free") +
  theme_minimal()


# Barplots for categorical predictors -----------------------
df %>%
  select(country, gender, credit_card, active_member, resp) %>%
  mutate(
    credit_card   = factor(credit_card),
    active_member = factor(active_member)
  ) %>%
  pivot_longer(
    cols = -resp,
    names_to = "variable",
    values_to = "value"
  ) %>%
  ggplot(aes(x = value, fill = resp)) +
  geom_bar(position = "fill") +
  facet_wrap(~ variable, scales = "free_x") +
  labs(
    y = "Proportion",
    x = "Category",
    title = "Categorical Predictors by Churn Proportion"
  ) +
  theme_minimal()

# Correlation Matrix for Numeric Predictors ----------------------
numeric_df <- df %>% select(where(is.numeric))
round(cor(numeric_df), 2)

# ** DEFINE PREPROCESSING RECIPES **

# SVM / Logistic Regression recipe
rec_svm <- recipe(resp ~ ., data = df) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>% # Encodes categorical variables
  step_center(all_numeric_predictors()) %>% # Center variables at mean
  step_scale(all_numeric_predictors()) %>% # Scale variables
  step_smote(resp) # Oversample minority class

# Decision tree recipe
rec_tree <- recipe(resp ~ ., data = df) %>%
  step_smote(resp)                                   # Oversample minority class


# Set up cross validation without reversing types...tried CreateCV, but it changed data types and overcomplicated the process
# This method is from caret
ctrl <- trainControl(
  method = "cv", # Cross validation method
  number = 5, # Num of folds
  classProbs = TRUE,
  savePredictions = "final",
  summaryFunction = twoClassSummary
)

# Sam: SVM Model ================================================================

svm_model <- train(
  resp ~ ., 
  data = df,
  method = "svmRadial",
  trControl = ctrl,
  tuneLength = 7,     # tunes C + sigma (gamma)
  preProcess = NULL,
  metric = "Accuracy"
)

svm_model
best_params <- svm_model$bestTune
print(best_params)

# Extract predictions from all CV folds
pred_df <- svm_model$pred %>%
  filter(C == best_params$C, sigma == best_params$sigma) %>%
  mutate(obs = factor(obs, levels = c("No", "Yes")))

# PERFORMANCE AT DEFAULT THRESHOLD 0.5

default_preds <- factor(ifelse(pred_df$Yes >= 0.5, "Yes", "No"), 
                        levels = c("No", "Yes"))

cm_default <- confusionMatrix(default_preds, pred_df$obs)
cm_default

# ROC CURVE + AUC

roc_obj <- roc(response = pred_df$obs, predictor = pred_df$Yes)
auc_value <- auc(roc_obj)
cat("AUC =", round(auc_value, 3), "\n")

plot(
  roc_obj,
  col="#3366CC",
  lwd=3,
  main=sprintf("ROC Curve (AUC = %.3f)", auc_value)
)
abline(a = 0, b = 1, col="gray50", lty=2)

# AUTOMATIC THRESHOLD TUNING (MAX F1)

thresholds <- seq(0.01, 0.99, 0.01)

metric_table <- data.frame(
  threshold = thresholds,
  Accuracy = NA,
  Sensitivity = NA,
  Specificity = NA,
  F1 = NA
)

for (i in seq_along(thresholds)) {
  
  t <- thresholds[i]
  
  preds_t <- factor(ifelse(pred_df$Yes >= t, "Yes", "No"),
                    levels = c("No", "Yes"))
  
  cm <- confusionMatrix(preds_t, pred_df$obs)
  
  metric_table$Accuracy[i]    <- cm$overall["Accuracy"]
  metric_table$Sensitivity[i] <- cm$byClass["Sensitivity"]
  metric_table$Specificity[i] <- cm$byClass["Specificity"]
  metric_table$F1[i]          <- cm$byClass["F1"]
}

# ---- Best threshold ----
best_idx <- which.max(metric_table$F1)
best_threshold <- metric_table$threshold[best_idx]
best_stats <- metric_table[best_idx, ]

cat("\nBEST THRESHOLD =", best_threshold, "\n")
print(best_stats)

# F1 VS THRESHOLD PLOT

ggplot(metric_table, aes(x = threshold, y = F1)) +
  geom_line(color="#3366CC", linewidth=1.2) +
  geom_vline(xintercept = best_threshold, color="red", linetype="dashed") +
  labs(
    title = "F1 Score Across Thresholds",
    x = "Threshold",
    y = "F1"
  ) +
  theme_minimal()

# FINAL CONFUSION MATRIX AT OPTIMAL THRESHOLD

final_preds <- factor(ifelse(pred_df$Yes >= best_threshold, "Yes", "No"),
                      levels = c("No", "Yes"))

cm_final <- confusionMatrix(final_preds, pred_df$obs)
cm_final

# END OF SVM CODE =============================================================
# DECISION TREE & RANDOM FOREST
# 2. Target + Categorical Predictors
data_raw$churn <- ifelse(data_raw$churn == 1, "Yes",
                         ifelse(data_raw$churn == 0, "No", NA))

data_raw$churn <- factor(data_raw$churn, levels = c("No", "Yes"))


# Identify categorical columns
cat_cols <- c("country", "gender", "credit_card", "active_member")
data_raw[cat_cols] <- lapply(data_raw[cat_cols], factor)


# 3. Train/Test Split + Drop ID Columns
set.seed(123)

idx <- caret::createDataPartition(data_raw$churn, p = 0.7, list = FALSE)
train_data <- data_raw[idx, ]
test_data  <- data_raw[-idx, ]

# Drop ID / high-cardinality identifier columns if present
id_cols <- c("RowNumber", "CustomerId", "Surname",
             "row_number", "customer_id", "surname", "ID", "Name")
id_cols <- intersect(id_cols, names(train_data))

if (length(id_cols) > 0) {
  train_data <- train_data[, !(names(train_data) %in% id_cols)]
  test_data  <- test_data[,  !(names(test_data)  %in% id_cols)]
}

train_data$churn <- droplevels(train_data$churn)
test_data$churn  <- droplevels(test_data$churn)


# 4. Recipe: Dummy Encode + SMOTE
#   - step_dummy: make factors numeric (0/1)
#   - step_smote: oversample minority churn class (Yes)
rec_tree <- recipe(churn ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors(), -all_outcomes()) %>%
  step_smote(churn)   # SMOTE only on training set (skip=TRUE for bake)

prep_tree <- prep(rec_tree, training = train_data)

# SMOTE-balance + dummy-encode training data
train_balanced <- bake(prep_tree, new_data = NULL)

# Apply same encoding (no SMOTE) to test data
test_processed <- bake(prep_tree, new_data = test_data)

# Sanity check class balance after SMOTE
table(train_data$churn)
table(train_balanced$churn)


# 5. Decision Tree on SMOTE Data
tree_fit <- rpart(
  churn ~ .,
  data   = train_balanced,
  method = "class",
  control = rpart.control(cp = 0.01)
)

# Plot tree 
rpart.plot(tree_fit, cex = 0.7)

# Predictions on processed test set
tree_pred_class <- predict(tree_fit, newdata = test_processed, type = "class")

cm_tree <- confusionMatrix(tree_pred_class, test_processed$churn)
cm_tree

# Variable importance
tree_varimp <- varImp(tree_fit)
tree_varimp


# 6. Random Forest on SMOTE Data
set.seed(123)

rf_fit <- randomForest(
  churn ~ .,
  data      = train_balanced,
  ntree     = 500,
  mtry      = floor(sqrt(ncol(train_balanced) - 1)),
  importance = TRUE
)

rf_fit   # shows OOB error + OOB confusion matrix

# Predictions on processed test set
rf_pred_class <- predict(rf_fit, newdata = test_processed, type = "class")

cm_rf <- confusionMatrix(rf_pred_class, test_processed$churn)
cm_rf

# Variable importance (random forest)
importance(rf_fit)
varImpPlot(rf_fit)
# END OF DECISION TREE & RANDOM FOREST ===============================

