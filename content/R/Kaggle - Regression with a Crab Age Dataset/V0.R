# 현재 script 저장되어 있는 경로 지정
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

# Introduce the requred packages
library(tidyverse)
library(mlr3verse)
library(data.table)
library(ggplot2)
library(patchwork)


# Reduce annoying logging
# lgr::get_logger("mlr3")$set_threshold("warn")
# lgr::get_logger("bbotk")$set_threshold("warn")

# Load data
train = fread("../data/playground-series-s3e16/train.csv")
test = fread("../data/playground-series-s3e16/test.csv")
original = fread("../data/playground-series-s3e16/CrabAgePrediction.csv")
submission = fread("../data/playground-series-s3e16/sample_submission.csv")

# View Information
str(train)

# Sex was found to be a factor variable
# XGBoost cannot process factorial data, into numeric data
train[Sex == "F", Sex := 0]
train[Sex == "M", Sex := 1]
train[Sex == "I", Sex := 2]
train$Sex = as.numeric(train$Sex)

# The test also needs to be processed
test[Sex == "F", Sex := 0]
test[Sex == "M", Sex := 1]
test[Sex == "I", Sex := 2]
test$Sex = as.numeric(test$Sex)

# Rename a variable
colnames(train) = c("id", "sex", paste0("V", 1:7), "Age")
colnames(test) = c("id", "sex", paste0("V", 1:7))

# Set the task
# Do not use the id column
task = as_task_regr(train[, 2:10], target = "Age")

# Observe missing values
task$missings()

# There are no missing values
# Select the learner
lrn_xgb = lrn("regr.xgboost") # need xgboost package

# Split the train data
set.seed(2)
split = partition(task, ratio = 0.7, stratify = T)

# Set the search space
search_space = ps(
  eta = p_dbl(lower = 0, upper = 1),
  min_child_weight = p_dbl(lower = 0, upper = 20),
  subsample = p_dbl(lower = 0.3, upper = 1),
  colsample_bytree = p_dbl(lower = 0.1, upper = 1),
  colsample_bylevel = p_dbl(lower = 0.1, upper = 1),
  nrounds = p_int(lower = 1, upper = 40))

# lts("regr.xgboost.default")
# lts("regr.xgboost.rbv1")
# lts("regr.xgboost.rbv2")

# Set the auto tuner
at = auto_tuner(
  tuner = tnr("grid_search", batch_size = 40),
  learner = lrn_xgb,
  resampling = rsmp("cv", folds = 5),
  measure = msr("regr.mae"),
  search_space = search_space,
  term_evals = 400
)

# To adjust the parameters 
# Start parallelization
future::plan("multisession", workers = 8)
set.seed(3)
at$train(task, row_ids = split$train)

# View the results of your training
at$tuning_result

# Update the model parameters
lrn_xgb$param_set$values = at$tuning_result$learner_param_vals[[1]]

# Start training and testing
lrn_xgb$train(task, row_ids = split$train)
predictions = lrn_xgb$predict(task, row_ids = split$test)

# View the results
predictions$score(msr("regr.mae"))

# Train the model with all the data
lrn_xgb$train(task, row_ids = seq_len(task$nrow))

# Make the forecast data
submission = lrn_xgb$predict_newdata(test[, 2:9])

# Process data for submission
submission = as.data.table(submission)
submission = submission[, c(1, 3)]
colnames(submission) = c("id", "Age")
submission$Age = round(submission$Age)
submission$id = test$id

head(submission)
write.csv(submission, "../data/playground-series-s3e16/submission.csv", row.names = F)
