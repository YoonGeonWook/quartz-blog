library(mlr3verse)
set.seed(123)
options(datatable.print.nrows = 10)
as.data.table(mlr_learners)[sapply(properties, function(x) "importance" %in% x),
                            .(key, label)]

optimizer = fs("rfe",
  n_features = 1,
  feature_number = 1,
  aggregation = "rank"
)

task = tsk("sonar")
task$head()

library(ggplot2)
library(data.table)
library(tidyverse)
data = as.data.table(task) %>% 
  melt(id.vars = task$target_names, measure.vars = task$feature_names)
data = data[c("V1", "V10", "V11", "V12", "V13", "V14"), ,on = "variable"]

ggplot(data, aes(x = value, fill = Class)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~variable, ncol = 6, scales = "free") +
  scale_fill_viridis_d(end = 0.8) +
  theme_minimal() +
  theme(axis.title.x = element_blank())

learner = lrn("classif.gbm",
  distribution = "bernoulli",
  predict_type = "prob")

instance = fsi(
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 6),
  measures = msr("classif.auc"),
  terminator = trm("none")
)

optimizer$optimize(instance)

library(viridisLite)
library(mlr3misc)
data = as.data.table(instance$archive)
data[, n := map_int(importance, length)]

ggplot(data, aes(x = n, y = classif.auc)) +
  geom_line(color = viridis(n = 1, begin = 0.5), linewidth = 1) +
  geom_point(fill = viridis(n = 1, begin = 0.5), shape = 21, size = 3, stroke = 0.5, alpha = 0.8) +
  xlab("Number of Features") +
  scale_x_reverse() + 
  theme_minimal()

as.data.table(instance$archive)[, .(features, classif.auc, importance)]

lrn("classif.svm")

learner = lrn("classif.svm",
  type = "C-classification",
  kernel = "linear",
  predict_type = "prob")

learner$properties

instance = fsi(
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 6),
  measures = msr("classif.auc"),
  terminator = trm("none"),
  callback = clbk("mlr3fselect.svm_rfe")
)
optimizer$optimize(instance)

data = as.data.table(instance$archive)
data[, n := map_int(importance, length)]

ggplot(data, aes(x = n, y = classif.auc)) +
  geom_line(color = viridis(n = 1, begin = 0.5), linewidth = 1) +
  geom_point(fill = viridis(n = 1, begin = 0.5), shape = 21, size = 3, stroke = 0.5, alpha = 0.8) +
  xlab("Number of Features") +
  scale_x_reverse() + 
  theme_minimal()


optimizer = fs("rfe",
               n_features = 1,
               feature_fraction = 0.75,
               aggregation = "rank"
)
instance = fsi(
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 6),
  measures = msr("classif.auc"),
  terminator = trm("none"),
  callback = clbk("mlr3fselect.svm_rfe")
)
optimizer$optimize(instance)
data = as.data.table(instance$archive)
data[, n := map_int(importance, length)]

ggplot(data, aes(x = n, y = classif.auc)) +
  geom_line(color = viridis(n = 1, begin = 0.5), linewidth = 1) +
  geom_point(fill = viridis(n = 1, begin = 0.5), shape = 21, size = 3, stroke = 0.5, alpha = 0.8) +
  xlab("Number of Features") +
  scale_x_reverse() + 
  theme_minimal()

optimizer = fs("rfecv",
  n_features = 1,
  feature_number = 1)

learner = lrn("classif.svm",
  type = "C-classification",
  kernel = "linear",
  predict_type = "prob")
instance = fsi(
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 6),
  measures = msr("classif.auc"),
  terminator = trm("none"),
  callback = clbk("mlr3fselect.svm_rfe")
)
optimizer$optimize(instance)

data = as.data.table(instance$archive)[!is.na(iteration), ]
aggr = data[, list("y" = mean(unlist(.SD))), by = "batch_nr", .SDcols = "classif.auc"]
aggr[, batch_nr := 61 - batch_nr]

data[, n := map_int(importance, length)]
ggplot(aggr, aes(x = batch_nr, y = y)) +
  geom_line(color = viridis(1, begin = 0.5), linewidth = 1) +
  geom_point(fill = viridis(1, begin = 0.5), shape = 21, size = 3, stroke = 0.5, alpha = 0.8) +
  geom_vline(xintercept = aggr[y == max(y)]$batch_nr,
             colour = viridis(1, begin = 0.33), linetype = 3, linewidth = 1) +
  xlab("Number of Features") +
  scale_x_reverse() +
  theme_minimal()

as.data.table(instance$archive)[, .(features, classif.auc, iteration, importance)]

task$select(instance$result_feature_set)
learner$train(task)
