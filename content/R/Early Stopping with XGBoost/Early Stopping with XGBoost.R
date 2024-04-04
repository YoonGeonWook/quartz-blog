# 현재 script 저장되어 있는 경로 지정
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

set.seed(7832)

library(mlr3verse)
library(ggplot2)
library(data.table)

task = tsk("spam")
split = partition(task, ratio = 0.8, stratify = T)
task$set_row_roles(rows = split$test, roles = "test")

learner = lrn("classif.xgboost",
  nrounds = 1000,
  early_stopping_rounds = 100,
  early_stopping_set = "test",
  eval_metric = "error")
learner$train(task)

data = melt(learner$model$evaluation_log, id.vars = "iter", variable.name = "set", value.name = "error")
ggplot(data, aes(x = iter, y = error, group = set)) +
  geom_line(aes(color = set)) +
  geom_vline(aes(xintercept = learner$model$best_iteration), color = "grey") +
  scale_x_continuous(breaks = seq(0, 150, 25)) +
  scale_color_manual(values = c("#f8766d", "#00b0f6"), labels = c("Train", "Test")) +
  labs(x = "Rounds", y = "Classification Error", color = "Set") +
  theme_minimal()

learner$model$best_iteration


learner = lrn("classif.xgboost",
  nrounds = 1000,
  early_stopping_rounds = 100,
  early_stopping_set = "test")

tuning_space = lts("classif.xgboost.default")
as.data.table(tuning_space)
learner = lts(learner)

learner$param_set$set_values(nrounds = 1000)

instance = tune(
  tuner = tnr("random_search", batch_size = 2),
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 3),
  measure = msr("classif.ce"),
  term_evals = 4,
  callbacks = clbk("mlr3tuning.early_stopping")
)

as.data.table(instance$archive)[, .(batch_nr, max_nrounds, eta, max_depth, colsample_bylevel, lambda, alpha, subsample)]
instance$result_learner_param_vals

learner = lrn("classif.xgboost")
learner$param_set$values = instance$result_learner_param_vals
learner$train(task)
