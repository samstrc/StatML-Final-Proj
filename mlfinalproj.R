

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

# Decision tree/RF recipe
rec_tree <- recipe(resp ~ ., data = df) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_smote(resp) # Oversample minority class


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
# RANDOM FOREST ===========================================================
set.seed(123)

rf_model <- train(
  x = rec_tree,
  data = df,
  method = "rf",
  trControl = ctrl,
  metric = "ROC",
  tuneLength = 10
)


rf_model
plot(rf_model)
varImp(rf_model)

# Extract predictions from best tuning mtry

best_mtry <- rf_model$bestTune$mtry

rf_pred_df <- rf_model$pred %>%
  filter(mtry == best_mtry) %>%
  mutate(obs = factor(obs, levels = c("No", "Yes")))

# Random Forest ROC Curve

rf_roc <- roc(
  response = rf_pred_df$obs,
  predictor = rf_pred_df$Yes
)

plot(
  rf_roc,
  col = "#33AA55",
  lwd = 3,
  main = sprintf("Random Forest ROC Curve (AUC = %.3f)", auc(rf_roc))
)
abline(a = 0, b = 1, lty = 2, col = "gray50")

# Threshold tuning: compute Accuracy, Sens, Spec, F1

thresholds <- seq(0.01, 0.99, 0.01)

rf_metric_table <- data.frame(
  threshold = thresholds,
  Accuracy = NA,
  Sensitivity = NA,
  Specificity = NA,
  F1 = NA
)

for (i in seq_along(thresholds)) {
  
  t <- thresholds[i]
  
  preds_t <- factor(
    ifelse(rf_pred_df$Yes >= t, "Yes", "No"),
    levels = c("No", "Yes")
  )
  
  cm <- confusionMatrix(preds_t, rf_pred_df$obs)
  
  rf_metric_table$Accuracy[i]    <- cm$overall["Accuracy"]
  rf_metric_table$Sensitivity[i] <- cm$byClass["Sensitivity"]
  rf_metric_table$Specificity[i] <- cm$byClass["Specificity"]
  rf_metric_table$F1[i]          <- cm$byClass["F1"]
}

# Identify the best threshold (max F1)

rf_best_idx <- which.max(rf_metric_table$F1)
rf_best_threshold <- rf_metric_table$threshold[rf_best_idx]
rf_best_stats <- rf_metric_table[rf_best_idx, ]

cat("\nRF BEST THRESHOLD =", rf_best_threshold, "\n")
print(rf_best_stats)

# F1 vs Threshold plot

ggplot(rf_metric_table, aes(x = threshold, y = F1)) +
  geom_line(color = "#33AA55", linewidth = 1.2) +
  geom_vline(xintercept = rf_best_threshold, color = "red", linetype = "dashed") +
  labs(
    title = "Random Forest F1 Score Across Thresholds",
    x = "Threshold",
    y = "F1 Score"
  ) +
  theme_minimal()

# Final confusion matrix using best threshold

rf_final_preds <- factor(
  ifelse(rf_pred_df$Yes >= rf_best_threshold, "Yes", "No"),
  levels = c("No", "Yes")
)

rf_cm_final <- confusionMatrix(rf_final_preds, rf_pred_df$obs)
rf_cm_final

# END OF DECISION TREE & RANDOM FOREST ===============================

# Taylor: Logistic Regression
# ============================================================
# LOGISTIC REGRESSION (Clean + caret + matching pipeline)
# ============================================================

# -----------------------------
# 1. Fit logistic regression
# -----------------------------

lr_model <- train(
  recipe = rec_svm,
  data = df,
  method = "glm",
  family = binomial,
  trControl = ctrl,
  metric = "ROC"
)

lr_model
best_lr <- lr_model$bestTune
print(best_lr)

# -----------------------------
# 2. Extract CV predictions
# -----------------------------

lr_pred_df <- lr_model$pred %>%
  mutate(obs = factor(obs, levels = c("No", "Yes")))

# -----------------------------
# 3. ROC Curve + AUC
# -----------------------------

lr_roc <- roc(
  response = lr_pred_df$obs,
  predictor = lr_pred_df$Yes
)

plot(
  lr_roc,
  col = "#AA3355",
  lwd = 3,
  main = sprintf("Logistic Regression ROC Curve (AUC = %.3f)", auc(lr_roc))
)
abline(a = 0, b = 1, col = "gray50", lty = 2)

lr_auc <- auc(lr_roc)
cat("LR AUC =", round(lr_auc, 3), "\n")

# -----------------------------
# 4. Threshold tuning (maximize F1)
# -----------------------------

thresholds <- seq(0.01, 0.99, 0.01)

lr_model <- train(
  rec_svm,     # <- NOTICE: no argument name, just the recipe object
  data = df,
  method = "glm",
  family = binomial,
  trControl = ctrl,
  metric = "ROC"
)


for (i in seq_along(thresholds)) {
  
  t <- thresholds[i]
  
  preds_t <- factor(
    ifelse(lr_pred_df$Yes >= t, "Yes", "No"),
    levels = c("No", "Yes")
  )
  
  cm <- confusionMatrix(preds_t, lr_pred_df$obs)
  
  lr_metric_table$Accuracy[i]    <- cm$overall["Accuracy"]
  lr_metric_table$Sensitivity[i] <- cm$byClass["Sensitivity"]
  lr_metric_table$Specificity[i] <- cm$byClass["Specificity"]
  lr_metric_table$F1[i]          <- cm$byClass["F1"]
}

lr_best_idx <- which.max(lr_metric_table$F1)
lr_best_threshold <- lr_metric_table$threshold[lr_best_idx]
lr_best_stats <- lr_metric_table[lr_best_idx, ]

cat("\nLR BEST THRESHOLD =", lr_best_threshold, "\n")
print(lr_best_stats)

# -----------------------------
# 5. F1 vs Threshold Plot
# -----------------------------

ggplot(lr_metric_table, aes(x = threshold, y = F1)) +
  geom_line(color = "#AA3355", linewidth = 1.2) +
  geom_vline(xintercept = lr_best_threshold, color = "red", linetype = "dashed") +
  labs(
    title = "Logistic Regression F1 Score Across Thresholds",
    x = "Threshold",
    y = "F1 Score"
  ) +
  theme_minimal()

# -----------------------------
# 6. Final Confusion Matrix at Optimal Threshold
# -----------------------------

lr_final_preds <- factor(
  ifelse(lr_pred_df$Yes >= lr_best_threshold, "Yes", "No"),
  levels = c("No", "Yes")
)

lr_cm_final <- confusionMatrix(lr_final_preds, lr_pred_df$obs)
lr_cm_final
