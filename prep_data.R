# Libraries
packages <- c("tidyverse", "caret", "recipes", "janitor")
to_install <- packages[!packages %in% installed.packages()[, "Package"]]
if (length(to_install)) install.packages(to_install, repos = "https://cloud.r-project.org")

lapply(packages, library, character.only = TRUE)


# 1. Import + Clean Column Names
library(readr)
library(janitor)
library(dplyr)
data <- read_csv("customer_churn.csv", skip = 1, show_col_types = FALSE) |>
  janitor::clean_names()
glimpse(data)
names(data)


# 2. Drop ID Column
data <- data %>%
  select(-customer_id)


# 3. Make churn a factor
data <- data %>%
  mutate(churn = factor(churn, levels = c(0, 1), labels = c("No", "Yes")))


# 4. Train / Test Split (Stratified)
set.seed(123)

idx <- caret::createDataPartition(data$churn, p = 0.7, list = FALSE)

train <- data[idx, ]
test  <- data[-idx, ]

table(train$churn)


# 5. RECIPE FOR SVM / LINEAR MODELS
rec_svm <- recipe(churn ~ ., data = train) %>%
  step_zv(all_predictors()) %>%                     # remove zero-variance
  step_dummy(all_nominal_predictors()) %>%          # convert factors to dummy vars
  step_center(all_numeric_predictors()) %>%         # scale
  step_scale(all_numeric_predictors())

prep_svm <- prep(rec_svm)
train_svm <- bake(prep_svm, train)
test_svm  <- bake(prep_svm, test)


# 6. RECIPE FOR DECISION TREES
rec_tree <- recipe(churn ~ ., data = train) %>%
  step_zv(all_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors())

prep_tree <- prep(rec_tree)
train_tree <- bake(prep_tree, train)
test_tree  <- bake(prep_tree, test)


# 7. SAVE OUTPUTS
dir.create("prepared", showWarnings = FALSE)

write_csv(train_svm,  "prepared/train_svm.csv")
write_csv(test_svm,   "prepared/test_svm.csv")

write_csv(train_tree, "prepared/train_tree.csv")
write_csv(test_tree,  "prepared/test_tree.csv")

saveRDS(prep_svm,  "prepared/recipe_svm.rds")
saveRDS(prep_tree, "prepared/recipe_tree.rds")

cat("DONE! Files saved in /prepared folder\n")
