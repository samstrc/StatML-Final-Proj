# MACHINE LEARNING FINAL PROJECT: Customer Churn Data
# Ava, Sam, Jordan, Taylor

# -------------------------
# LOAD LIBRARIES
# -------------------------
packages <- c(
  "tidyverse", "tidymodels", "janitor", "pROC",
  "themis", "vip", "ranger"
)

to_install <- packages[!packages %in% installed.packages()[,1]]
if (length(to_install)) install.packages(to_install)

lapply(packages, library, character.only = TRUE)
tidymodels_prefer()

# -------------------------
# LOAD & CLEAN DATA
# -------------------------
data <- read_csv(
  "https://raw.githubusercontent.com/samstrc/StatML-Final-Proj/refs/heads/main/customer_churn.csv",
  show_col_types = FALSE
) %>% clean_names()

df <- data %>%
  select(-customer_id) %>%
  mutate(
    resp = factor(churn, levels = c(0,1), labels = c("No","Yes")),
    resp = forcats::fct_relevel(resp, "Yes", "No"),   # <-- FIX HERE
    across(c(country, gender, credit_card, active_member), as.factor)
  ) %>%
  select(resp, everything(), -churn)

# -------------------------
# BASIC CHECKS
# -------------------------
print(table(df$resp))
print(prop.table(table(df$resp)))
print(colSums(is.na(df)))

# -------------------------
# EDA - CLASS IMBALANCE!!!!!
# -------------------------
ggplot(df, aes(x = resp)) +
  geom_bar(aes(y = after_stat(count/sum(count))), fill = "steelblue") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Class Proportions", y = "Percent")

# -------------------------
# EDA - NUMERIC DISTRIBUTIONS
# -------------------------
df %>%
  select(where(is.numeric)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 30, fill = "#6699CC") +
  facet_wrap(~variable, scales = "free") +
  theme_minimal()

# -------------------------
# EDA - CATEGORICAL
# -------------------------
df %>%
  select(country, gender, credit_card, active_member, resp) %>%
  pivot_longer(cols = -resp, names_to = "variable", values_to = "value") %>%
  ggplot(aes(x = value, fill = resp)) +
  geom_bar(position = "fill") +
  facet_wrap(~variable, scales = "free_x") +
  labs(y = "Proportion") +
  theme_minimal()

# -------------------------
# EDA - CORRELATION MATRIX...look for colinearity 
# -------------------------
df %>% select(where(is.numeric)) %>% cor() %>% round(3)

# -------------------------
# Chi-square test of association (resp vs categorical predictors)
# -------------------------

# Pick only categorical predictors (exclude resp itself)
cat_vars <- df %>%
  select(where(is.factor)) %>%   # all factors
  select(-resp)                  # drop response

# Run chi-square test for each categorical variable vs resp
chi_results <- purrr::map_df(names(cat_vars), \(v) {
  tbl  <- table(df[[v]], df$resp)   # contingency table
  test <- stats::chisq.test(tbl)

  tibble(
    variable = v,
    p_value  = test$p.value
  )
})

chi_results # Note that the only categorical variable that isn't associated with the resp is credit_card

# Simple barplot of p-values
pvals <- chi_results$p_value
names(pvals) <- chi_results$variable

barplot(
  pvals,
  las = 2,
  cex.names = 0.9,
  main = "Chi-square p-values: Association with Churn",
  ylab = "p-value"
)
abline(h = 0.05, col = "red", lty = 2)

# ====================================================================
#  PREPROCESSING RECIPE (WITH SMOTE)
# ====================================================================
rec <- recipe(resp ~ ., data = df) %>%
  step_dummy(all_nominal_predictors()) %>%      # convert factors
  step_normalize(all_numeric_predictors()) %>%  # scale numeric
  step_smote(resp)                              # balance classes (only applied to training)

# -------------------------
# DATA SPLIT & CV
# -------------------------
set.seed(123)
split <- initial_split(df, prop = 0.8, strata = resp)
train_data <- training(split)
test_data  <- testing(split)

folds <- vfold_cv(train_data, v = 5, strata = resp)

# ====================================================================
#  MODEL SPECIFICATIONS
# ====================================================================

# Logistic Regression (no tuning, just baseline wo regularization)
lr_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# SVM (RBF) tuned
svm_spec <- svm_rbf(
  cost = tune(),
  rbf_sigma = tune()
) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

# Random Forest tuned
rf_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

# ====================================================================
#  WORKFLOWS (puts together models & recipes)
# ====================================================================
lr_wf  <- workflow() %>% add_model(lr_spec)  %>% add_recipe(rec)
svm_wf <- workflow() %>% add_model(svm_spec) %>% add_recipe(rec)
rf_wf  <- workflow() %>% add_model(rf_spec)  %>% add_recipe(rec)

# ====================================================================
#  HYPERPARAMETER GRIDS (tests parameters for optimal performance)
# ====================================================================

# Prep once so we know number of predictors
rec_prep <- prep(rec)
p <- length(bake(rec_prep, new_data = train_data)) - 1  # number predictors minus outcome

# SVM grid
grid_svm <- grid_regular(
  cost(range = c(-3, 3)),        # log2 scale, narrower than [-5, 5]
  rbf_sigma(range = c(-3, 3)),
  levels = 4                     # 4 values per dimension (16 total)
)

# RF grid
rf_params <- parameters(
  finalize(mtry(), train_data),
  min_n()
)

grid_rf <- grid_random(
  rf_params,
  size = 10   # 10 random combos instead of 25
)

# ====================================================================
#  TUNING
# ====================================================================

# Logistic regression (NO TUNING - BASELINE MODEL)
lr_fit <- lr_wf %>%
  fit_resamples(
    folds,
    metrics = metric_set(roc_auc, accuracy, f_meas),
    control = control_resamples(save_pred = TRUE)
  )

# SVM tuning
svm_tuned <- svm_wf %>%
  tune_grid(
    folds,
    grid = grid_svm,
    metrics = metric_set(roc_auc, accuracy, f_meas),
    control = control_resamples(save_pred = TRUE)
  )

# Random Forest tuning
rf_tuned <- rf_wf %>%
  tune_grid(
    folds,
    grid = grid_rf,
    metrics = metric_set(roc_auc, accuracy, f_meas),
    control = control_resamples(save_pred = TRUE)
  )

# ====================================================================
#  SELECT BEST MODELS
# ====================================================================

# Select best hyperparameters
best_svm <- svm_tuned %>% select_best(metric = "roc_auc")
best_rf  <- rf_tuned  %>% select_best(metric = "roc_auc")

# Finalize workflows using best params
svm_final_wf <- svm_wf %>% finalize_workflow(best_svm)
rf_final_wf  <- rf_wf  %>% finalize_workflow(best_rf)

# Fit models on full training data
svm_final <- svm_final_wf %>% fit(data = train_data)
rf_final  <- rf_final_wf  %>% fit(data = train_data)

# Logistic Regression had no tuning, so just fit
lr_final  <- lr_wf %>% fit(data = train_data)

# ====================================================================
#  TEST SET PREDICTIONS
# ====================================================================

# LOGISTIC REGRESSION PREDICTIONS
lr_pred <- predict(lr_final, test_data, type = "prob") %>%
  bind_cols(predict(lr_final, test_data, type = "class")) %>%
  bind_cols(test_data %>% select(resp))

# SVM (RBF) PREDICTIONS
svm_pred <- predict(svm_final, test_data, type = "prob") %>%
  bind_cols(predict(svm_final, test_data, type = "class")) %>%
  bind_cols(test_data %>% select(resp))

# RANDOM FOREST PREDICTIONS
rf_pred <- predict(rf_final, test_data, type = "prob") %>%
  bind_cols(predict(rf_final, test_data, type = "class")) %>%
  bind_cols(test_data %>% select(resp))

# Colors for all ROC plots
model_colors <- c(
  "Logistic Regression" = "steelblue",
  "SVM (RBF)"            = "firebrick2",
  "Random Forest"        = "forestgreen"
)

# ====================================================================
#  TEST-SET ROC CURVES
# ====================================================================

# ROC curve data frames (test set)
lr_test_roc <- lr_pred %>%
  roc_curve(truth = resp, .pred_Yes, event_level = "second") %>%
  mutate(Model = "Logistic Regression")

svm_test_roc <- svm_pred %>%
  roc_curve(truth = resp, .pred_Yes, event_level = "second") %>%
  mutate(Model = "SVM (RBF)")

rf_test_roc <- rf_pred %>%
  roc_curve(truth = resp, .pred_Yes, event_level = "second") %>%
  mutate(Model = "Random Forest")

test_roc_df <- bind_rows(lr_test_roc, svm_test_roc, rf_test_roc)

ggplot(test_roc_df,
       aes(x = 1 - specificity, y = sensitivity, colour = Model)) +
  geom_line(linewidth = 1.1) +
  geom_abline(linetype = "dashed", colour = "gray50") +
  coord_equal() +
  scale_color_manual(values = model_colors) +
  labs(
    title  = "ROC Curves - Test Set",
    x      = "1 - Specificity (False Positive Rate)",
    y      = "Sensitivity (True Positive Rate)",
    colour = "Model"
  ) +
  theme_minimal(base_size = 13)

# ====================================================================
#  MODEL PERFORMANCE TABLE - TEST SET
# ====================================================================

# Helper function to find all metrics for a prediction tibble
compute_metrics <- function(pred_df) {
  # Make "Yes" the first level so event_level = 'first' = churn
  pred_df <- pred_df %>%
    mutate(
      resp        = factor(resp, levels = c("Yes", "No")),
      .pred_class = factor(.pred_class, levels = c("Yes", "No"))
    )

  acc  <- accuracy(pred_df, truth = resp, estimate = .pred_class)$.estimate
  prec <- precision(pred_df, truth = resp, estimate = .pred_class,
                    event_level = "first")$.estimate
  rec  <- recall(pred_df, truth = resp, estimate = .pred_class,
                 event_level = "first")$.estimate
  spec <- specificity(pred_df, truth = resp, estimate = .pred_class,
                      event_level = "first")$.estimate
  f1   <- f_meas(pred_df, truth = resp, estimate = .pred_class,
                 event_level = "first")$.estimate

  auc  <- roc_auc(
    pred_df,
    truth       = resp,
    .pred_Yes,
    event_level = "first"
  )$.estimate

  tibble(
    Accuracy    = acc,
    Precision   = prec,
    Recall      = rec,
    Specificity = spec,
    F1          = f1,
    AUC         = auc
  )
}

# Compute metrics for each model
lr_metrics_test  <- compute_metrics(lr_pred)  %>% mutate(Model = "Logistic Regression")
svm_metrics_test <- compute_metrics(svm_pred) %>% mutate(Model = "SVM (RBF)")
rf_metrics_test  <- compute_metrics(rf_pred)  %>% mutate(Model = "Random Forest")

# Combine into one table
model_comparison <- bind_rows(
  lr_metrics_test,
  svm_metrics_test,
  rf_metrics_test
) %>%
  select(Model, everything())

cat("\n===== TEST-SET PERFORMANCE (churn = 'Yes') =====\n")
print(model_comparison)

# ====================================================================
#  CONFUSION MATRICES - TEST SET
# ====================================================================

# For these we also want Yes/No ordering
cm_relevel <- function(pred_df) {
  pred_df %>%
    mutate(
      resp        = factor(resp, levels = c("Yes", "No")),
      .pred_class = factor(.pred_class, levels = c("Yes", "No"))
    )
}

lr_cm  <- conf_mat(cm_relevel(lr_pred),  truth = resp, estimate = .pred_class)
svm_cm <- conf_mat(cm_relevel(svm_pred), truth = resp, estimate = .pred_class)
rf_cm  <- conf_mat(cm_relevel(rf_pred),  truth = resp, estimate = .pred_class)

cat("\n--- Logistic Regression Confusion Matrix ---\n")
print(lr_cm)

cat("\n--- SVM (RBF) Confusion Matrix ---\n")
print(svm_cm)

cat("\n--- Random Forest Confusion Matrix ---\n")
print(rf_cm)

# ====================================================================
#  TRAINING (CROSS-VALIDATION) METRICS
#  using lr_fit, svm_tuned, rf_tuned
# ====================================================================

# Logistic regression: only one spec, so just keep its metrics
lr_train_metrics <- lr_fit %>%
  collect_metrics() %>%
  filter(.metric %in% c("accuracy", "roc_auc", "f_meas")) %>%
  transmute(
    Model  = "Logistic Regression",
    .metric,
    mean
  )

# SVM: restrict to the best (cost, rbf_sigma) combo
svm_train_metrics <- svm_tuned %>%
  collect_metrics() %>%
  inner_join(best_svm, by = c("cost", "rbf_sigma")) %>%  # keep best row only
  filter(.metric %in% c("accuracy", "roc_auc", "f_meas")) %>%
  transmute(
    Model  = "SVM (RBF)",
    .metric,
    mean
  )

# Random Forest: restrict to best (mtry, min_n) combo
rf_train_metrics <- rf_tuned %>%
  collect_metrics() %>%
  inner_join(best_rf, by = c("mtry", "min_n")) %>%       # keep best row only
  filter(.metric %in% c("accuracy", "roc_auc", "f_meas")) %>%
  transmute(
    Model  = "Random Forest",
    .metric,
    mean
  )

# Combine and pivot wide so we get Train_accuracy, Train_roc_auc, Train_f_meas
train_metrics <- bind_rows(
  lr_train_metrics,
  svm_train_metrics,
  rf_train_metrics
) %>%
  pivot_wider(
    names_from  = .metric,
    values_from = mean,
    names_prefix = "Train_"
  )

# ====================================================================
#  TRAIN vs TEST METRICS TABLE
# ====================================================================

test_metrics <- model_comparison %>%
  rename_with(~ paste0("Test_", .), -Model)

train_test_metrics <- train_metrics %>%
  left_join(test_metrics, by = "Model")

cat("\n===== TRAIN vs TEST METRICS =====\n")
print(train_test_metrics)

# ====================================================================
#  TRAINING ROC CURVES
# ====================================================================

lr_train_pred <- lr_fit %>% collect_predictions()

svm_train_pred <- svm_tuned %>%
  collect_predictions(parameters = best_svm)

rf_train_pred <- rf_tuned %>%
  collect_predictions(parameters = best_rf)

lr_train_roc <- lr_train_pred %>%
  roc_curve(truth = resp, .pred_Yes, event_level = "second") %>%
  mutate(Model = "Logistic Regression")

svm_train_roc <- svm_train_pred %>%
  roc_curve(truth = resp, .pred_Yes, event_level = "second") %>%
  mutate(Model = "SVM (RBF)")

rf_train_roc <- rf_train_pred %>%
  roc_curve(truth = resp, .pred_Yes, event_level = "second") %>%
  mutate(Model = "Random Forest")

train_roc_df <- bind_rows(lr_train_roc, svm_train_roc, rf_train_roc)

ggplot(train_roc_df,
       aes(x = 1 - specificity, y = sensitivity, colour = Model)) +
  geom_line(linewidth = 1.1) +
  geom_abline(linetype = "dashed", colour = "gray50") +
  coord_equal() +
  scale_color_manual(values = model_colors) +
  labs(
    title = "Training ROC Curves (5-fold Cross-Validation)",
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)",
    colour = "Model"
  ) +
  theme_minimal(base_size = 13)

# ====================================================================
#  MODEL COMPLEXITY / PARAMETERS
# ====================================================================

# Logistic Regression: number of coefficients
lr_fit_obj   <- lr_final %>% extract_fit_parsnip()
lr_coef_tbl  <- tidy(lr_fit_obj)
lr_num_params <- nrow(lr_coef_tbl)

cat("\n===== MODEL COMPLEXITY =====\n")
cat("Logistic Regression: # of coefficients (including intercept):",
    lr_num_params, "\n")

# SVM (RBF): support vectors & tuned hyperparameters
svm_fit_obj <- svm_final %>% extract_fit_parsnip()
svm_raw     <- svm_fit_obj$fit   # kernlab ksvm object
num_sv      <- nrow(svm_raw@xmatrix[[1]])

cat("\nSVM (RBF):\n")
cat("  Cost (C):         ", best_svm$cost, "\n")
cat("  Sigma (rbf_sigma):", best_svm$rbf_sigma, "\n")
cat("  # Support vectors:", num_sv, "\n")

# Random Forest: trees & hyperparameters
rf_fit_obj <- rf_final %>% extract_fit_parsnip()
rf_raw     <- rf_fit_obj$fit   # ranger object

cat("\nRandom Forest:\n")
cat("  # Trees: ", rf_raw$num.trees, "\n")
cat("  mtry:    ", best_rf$mtry, "\n")
cat("  min_n:   ", best_rf$min_n, "\n")