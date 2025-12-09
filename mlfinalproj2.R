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
# LOAD + CLEAN DATA
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
# EDA — CLASS IMBALANCE!!!!!
# -------------------------
ggplot(df, aes(x = resp)) +
  geom_bar(aes(y = after_stat(count/sum(count))), fill = "steelblue") +
  scale_y_continuous(labels = scales::percent_format()) +
  labs(title = "Class Proportions", y = "Percent")

# -------------------------
# EDA — NUMERIC DISTRIBUTIONS
# -------------------------
df %>%
  select(where(is.numeric)) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "value") %>%
  ggplot(aes(value)) +
  geom_histogram(bins = 30, fill = "#6699CC") +
  facet_wrap(~variable, scales = "free") +
  theme_minimal()

# -------------------------
# EDA — CATEGORICAL
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
# EDA — CORRELATION MATRIX...look for colinearity 
# -------------------------
df %>% select(where(is.numeric)) %>% cor() %>% round(3)

# ====================================================================
#  PREPROCESSING RECIPE (WITH SMOTE)
# ====================================================================
rec <- recipe(resp ~ ., data = df) %>%
  step_dummy(all_nominal_predictors()) %>%      # convert factors
  step_normalize(all_numeric_predictors()) %>%  # scale numeric
  step_smote(resp)                              # balance classes (only applied to training)

# -------------------------
# DATA SPLIT + CV
# -------------------------
set.seed(123)
split <- initial_split(df, prop = 0.8, strata = resp)
train_data <- training(split)
test_data  <- testing(split)

folds <- vfold_cv(train_data, v = 5, strata = resp)

# ====================================================================
#  MODEL SPECIFICATIONS
# ====================================================================

# Logistic Regression (no tuning)
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
# mtry MUST be finalized later after recipe prep
rf_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("classification")

# ====================================================================
#  WORKFLOWS
# ====================================================================
lr_wf  <- workflow() %>% add_model(lr_spec)  %>% add_recipe(rec)
svm_wf <- workflow() %>% add_model(svm_spec) %>% add_recipe(rec)
rf_wf  <- workflow() %>% add_model(rf_spec)  %>% add_recipe(rec)

# ====================================================================
#  HYPERPARAMETER GRIDS
# ====================================================================

# Prep once so we know number of predictors
rec_prep <- prep(rec)
p <- length(bake(rec_prep, new_data = train_data)) - 1  # number predictors minus outcome

# SVM grid
grid_svm <- grid_regular(
  cost(range = c(-3, 3)),        # log2 scale, narrower than [-5, 5]
  rbf_sigma(range = c(-3, 3)),
  levels = 4                     # 4 values per dimension → 16 total
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

# Logistic regression — NO TUNING NEEDED
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

# --- Select best hyperparameters ---
best_svm <- svm_tuned %>% select_best(metric = "roc_auc")
best_rf  <- rf_tuned  %>% select_best(metric = "roc_auc")

# --- Finalize workflows using best params ---
svm_final_wf <- svm_wf %>% finalize_workflow(best_svm)
rf_final_wf  <- rf_wf  %>% finalize_workflow(best_rf)

# --- Fit models on full training data ---
svm_final <- svm_final_wf %>% fit(data = train_data)
rf_final  <- rf_final_wf  %>% fit(data = train_data)

# Logistic Regression had no tuning, so just fit as-is
lr_final  <- lr_wf %>% fit(data = train_data)


# ====================================================================
#  TEST SET PREDICTIONS
# ====================================================================
# LOGISTIC REGRESSION PREDICTIONS
lr_pred <- predict(lr_final, test_data, type = "prob") %>%
  bind_cols(predict(lr_final, test_data, type = "class")) %>%
  bind_cols(test_data %>% select(resp))

# SVM PREDICTIONS
svm_pred <- predict(svm_final, test_data, type = "prob") %>%
  bind_cols(predict(svm_final, test_data, type = "class")) %>%
  bind_cols(test_data %>% select(resp))

# RANDOM FOREST PREDICTIONS
rf_pred <- predict(rf_final, test_data, type = "prob") %>%
  bind_cols(predict(rf_final, test_data, type = "class")) %>%
  bind_cols(test_data %>% select(resp))


# ====================================================================
#  METRICS (TEST SET)
# ====================================================================
lr_metrics  <- lr_pred  %>% roc_auc(resp, .pred_Yes)
svm_metrics <- svm_pred %>% roc_auc(resp, .pred_Yes)
rf_metrics  <- rf_pred  %>% roc_auc(resp, .pred_Yes)

print(lr_metrics)
print(svm_metrics)
print(rf_metrics)

# ====================================================================
#  COMBINED ROC CURVE — TEST SET ONLY
# ====================================================================
lr_roc  <- roc(lr_pred$resp,  lr_pred$.pred_Yes)
svm_roc <- roc(svm_pred$resp, svm_pred$.pred_Yes)
rf_roc  <- roc(rf_pred$resp,  rf_pred$.pred_Yes)

plot(lr_roc, col = "blue",  lwd = 3, main = "ROC Curves — Test Set")
plot(svm_roc, col = "red",  lwd = 3, add = TRUE)
plot(rf_roc, col = "green", lwd = 3, add = TRUE)

legend(
  "bottomright",
  legend = c(
    paste("Logistic Regression  (AUC =", round(lr_roc$auc, 3), ")"),
    paste("SVM-RBF              (AUC =", round(svm_roc$auc, 3), ")"),
    paste("Random Forest        (AUC =", round(rf_roc$auc, 3), ")")
  ),
  col = c("blue", "red", "green"),
  lwd = 3
)

# ====================================================================
#  MODEL PERFORMANCE TABLE — TEST SET
# ====================================================================

# Helper function to compute all metrics for a prediction tibble
compute_metrics <- function(pred_df) {
  pred_df <- pred_df %>%
    mutate(
      resp        = factor(resp, levels = c("Yes", "No")),
      .pred_class = factor(.pred_class, levels = c("Yes", "No"))
    )

  acc  <- accuracy(pred_df, truth = resp, estimate = .pred_class)$.estimate
  prec <- precision(pred_df, truth = resp, estimate = .pred_class, event_level = "first")$.estimate
  rec  <- recall(pred_df, truth = resp, estimate = .pred_class, event_level = "first")$.estimate
  spec <- specificity(pred_df, truth = resp, estimate = .pred_class, event_level = "first")$.estimate
  f1   <- f_meas(pred_df, truth = resp, estimate = .pred_class, event_level = "first")$.estimate

  auc  <- roc_auc(
    pred_df,
    truth      = resp,
    .pred_Yes,
    event_level = "first"   # "Yes" is the first level now
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
lr_metrics  <- compute_metrics(lr_pred)  %>% mutate(Model = "Logistic Regression")
svm_metrics <- compute_metrics(svm_pred) %>% mutate(Model = "SVM (RBF)")
rf_metrics  <- compute_metrics(rf_pred)  %>% mutate(Model = "Random Forest")

# Combine into one table
model_comparison <- bind_rows(lr_metrics, svm_metrics, rf_metrics) %>%
  select(Model, everything())

print(model_comparison)
