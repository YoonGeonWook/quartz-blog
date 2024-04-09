# 현재 script 저장되어 있는 경로 지정
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

set.seed(123)

# Packages Load
suppressWarnings(
  suppressMessages({
    library(tidyverse)
    library(mlr3verse)
    library(mlr3mbo)
    library(mlr3tuningspaces)
    library(mlr3tuning)
    library(bbotk)
    library(data.table)
    library(formattable)
    library(patchwork)
    library(ggpubr)
    library(cowplot)
    library(rcompanion)
    library(cvms)
    library(tictoc)
    library(DiagrammeR)
    library(iml)
    library(DALEX)
    library(DALEXtra)
  })
)

train = fread("../data/playground-series-s4e1/train.csv")
test = fread("../data/playground-series-s4e1/test.csv")
submission = fread("../data/playground-series-s4e1/sample_submission.csv")
original = fread("../data/playground-series-s4e1/Churn_Modelling.csv")

sprintf("Dimension of the train dataset: (%s, %s)", nrow(train), ncol(train))
sprintf("Dimension of the test dataset: (%s, %s)", nrow(test), ncol(test))
sprintf("Dimension of the original dataset: (%s, %s)", nrow(original), ncol(original))
original = original %>% 
  mutate(RowNumber = RowNumber - 1) %>% 
  rename(id = RowNumber)

train = train %>% 
  mutate(across(c(id, CustomerId), as.character)) %>% 
  mutate(across(c(Geography, Gender, NumOfProducts, HasCrCard, IsActiveMember, Exited), as.factor)) %>% 
  mutate(CreditScore = as.numeric(CreditScore))
test = test %>% 
  mutate(across(c(id, CustomerId), as.character)) %>% 
  mutate(across(c(Geography, Gender, NumOfProducts, HasCrCard, IsActiveMember), as.factor)) %>% 
  mutate(CreditScore = as.numeric(CreditScore))
original = original %>% 
  mutate(across(c(id, CustomerId), as.character)) %>% 
  mutate(across(c(Geography, Gender, NumOfProducts, HasCrCard, IsActiveMember, Exited), as.factor)) %>% 
  mutate(CreditScore = as.numeric(CreditScore))


# tmp = lapply(names(train), function(.x) n_distinct(train[[.x]]))
# names(tmp) = names(train)
# tmp %>% unlist()
# 
# tmp = lapply(names(train), function(.x) class(train[[.x]]))
# names(tmp) = names(train)
# tmp %>% unlist()


# train %>% head() %>% formattable()
# 
# train %>% select_if(is.numeric) %>% 
#   pivot_longer(everything()) %>% 
#   group_by(name) %>% 
#   reframe(
#     count = n(),
#     Mean = mean(value),
#     SD = sd(value),
#     Min = min(value),
#     `25%` = quantile(value, 0.25),
#     `50%` = quantile(value, 0.50),
#     `75%` = quantile(value, 0.75),
#     Max = max(value)
#   ) %>% 
#   formattable()
# 
# train %>% 
#   select(where(is.character), where(is.factor)) %>% 
#   pivot_longer(everything()) %>% 
#   group_by(name) %>% 
#   reframe(
#     n_unique = n_distinct(value)
#   ) %>% 
#   formattable()
# 
# test %>% head() %>% formattable()
# 
# test %>% select_if(is.numeric) %>% 
#   pivot_longer(everything()) %>% 
#   group_by(name) %>% 
#   reframe(
#     count = n(),
#     Mean = mean(value),
#     SD = sd(value),
#     Min = min(value),
#     `25%` = quantile(value, 0.25),
#     `50%` = quantile(value, 0.50),
#     `75%` = quantile(value, 0.75),
#     Max = max(value)
#   ) %>% 
#   formattable()
# 
# test %>% 
#   select(where(is.character), where(is.factor)) %>% 
#   pivot_longer(everything()) %>% 
#   group_by(name) %>% 
#   reframe(
#     n_unique = n_distinct(value)
#   ) %>% 
#   formattable()
# 
# original %>% head() %>% formattable()
# 
# original %>% select_if(is.numeric) %>% 
#   pivot_longer(everything()) %>% 
#   group_by(name) %>% 
#   reframe(
#     count = n(),
#     Mean = mean(value, na.rm = T),
#     SD = sd(value, na.rm = T),
#     Min = min(value, na.rm = T),
#     `25%` = quantile(value, 0.25, na.rm = T),
#     `50%` = quantile(value, 0.50, na.rm = T),
#     `75%` = quantile(value, 0.75, na.rm = T),
#     Max = max(value, na.rm = T)
#   ) %>% 
#   formattable()
# 
# original %>% 
#   select(where(is.character), where(is.factor)) %>% 
#   pivot_longer(everything()) %>% 
#   group_by(name) %>% 
#   reframe(
#     n_unique = n_distinct(value)
#   ) %>% 
#   formattable()

# train %>% is.na() %>% colSums()
# test %>% is.na() %>% colSums()

# p1 = train %>%
#   count(Exited) %>%
#   mutate(prop = n/sum(n)) %>%
#   ggplot(aes(x = '', y = prop, fill = Exited)) +
#   geom_col(color = 'black') +
#   geom_text(aes(label = scales::percent(prop, suffix = '%', accuracy = 0.1)),
#             position = position_stack(vjust = 0.5)) +
#   coord_polar(theta = 'y') +
#   theme_void() +
#   theme(legend.position = 'none')
# 
# p2 = train %>%
#   count(Exited) %>%
#   mutate(prop = n/sum(n)) %>%
#   ggplot(aes(x = Exited, y = n, fill = Exited)) +
#   geom_col(color = 'black') +
#   scale_y_continuous(name = "count",
#                      breaks = seq(0, 120000, 20000)) +
#   theme_minimal()
# (p1 + p2) +
#   plot_annotation(
#     title = "Target Value Analysis - Competition Data",
#     theme = theme(plot.title = element_text(hjust = 0.5))
#   )

# p1 = original %>% 
#   count(Exited) %>% 
#   mutate(prop = n/sum(n)) %>% 
#   ggplot(aes(x = '', y = prop, fill = Exited)) +
#   geom_col(color = 'black') +
#   geom_text(aes(label = scales::percent(prop, suffix = '%', accuracy = 0.1)),
#             position = position_stack(vjust = 0.5)) +
#   coord_polar(theta = 'y') +
#   theme_void() +
#   theme(legend.position = 'none')
# 
# p2 = original %>% 
#   count(Exited) %>% 
#   mutate(prop = n/sum(n)) %>% 
#   ggplot(aes(x = Exited, y = n, fill = Exited)) +
#   geom_col(color = 'black') +
#   scale_y_continuous(name = "count", 
#                      breaks = seq(0, 8000, 2000)) +
#   theme_minimal()
# (p1 + p2) +
#   plot_annotation(
#     title = "Target Value Analysis - Original Data",
#     theme = theme(plot.title = element_text(hjust = 0.5))
#   )

# p1 = train %>% 
#   count(Geography) %>% 
#   mutate(prop = n / sum(n)) %>% 
#   ggplot(aes(x = '', y = prop, fill = Geography)) +
#   geom_col(color = 'black') +
#   geom_text(aes(label = scales::percent(prop, suffix = '%', accuracy = 0.1)),
#                 position = position_stack(vjust = 0.5)) +
#   coord_polar(theta = 'y') +
#   theme_void() +
#   theme(legend.position = "none")
# p2 = train %>% 
#   count(Geography) %>% 
#   mutate(prop = n / sum(n)) %>% 
#   ggplot(aes(x = Geography, y = n, fill = Geography)) +
#   geom_col(color = 'black') +
#   scale_y_continuous(name = "count", 
#                      breaks = seq(0, 80000, 20000)) +
#   theme_minimal() +
#   theme(legend.position = "left")
# 
# (p1 + p2) +
#   plot_annotation(
#     title = "Geography",
#     theme = theme(plot.title = element_text(hjust = 0.45))
#   )
# 
# 
# p1 = train %>%
#   count(Gender) %>%
#   mutate(prop = n / sum(n)) %>%
#   ggplot(aes(x = '', y = prop, fill = Gender)) +
#   geom_col(color = 'black') +
#   geom_text(aes(label = scales::percent(prop, suffix = '%', accuracy = 0.1)),
#             position = position_stack(vjust = 0.5)) +
#   coord_polar(theta = 'y', start = pi/2) +
#   theme_void() +
#   theme(legend.position = "none")
# p2 = train %>%
#   count(Gender) %>%
#   mutate(prop = n / sum(n)) %>%
#   ggplot(aes(x = Gender, y = n, fill = Gender)) +
#   geom_col(color = 'black') +
#   scale_y_continuous(name = "count",
#                      breaks = seq(0, 80000, 20000)) +
#   theme_minimal() +
#   theme(legend.position = "left")
# (p1 + p2) +
#   plot_annotation(
#     title = "Gender",
#     theme = theme(plot.title = element_text(hjust = 0.45))
#   )
# 
# p1 = train %>% 
#   count(Tenure) %>% 
#   mutate(prop = n / sum(n)) %>% 
#   ggplot(aes(x = '', y = prop, fill = Tenure)) +
#   geom_col(color = 'black') +
#   geom_text(aes(label = scales::percent(prop, suffix = '%', accuracy = 0.1)),
#             position = position_stack(vjust = 0.5),
#             size = 3.5) +
#   coord_polar(theta = 'y', start = pi/2) +
#   theme_void() +
#   theme(legend.position = "none")
# p2 = train %>% 
#   count(Tenure) %>% 
#   mutate(prop = n / sum(n)) %>% 
#   ggplot(aes(x = Tenure, y = n, fill = Tenure)) +
#   geom_col(color = 'black') +
#   scale_y_continuous(name = "count", 
#                      breaks = seq(0, 17500, 2500)) +
#   theme_minimal() +
#   theme(legend.position = "left")
# (p1 + p2) +
#   plot_annotation(
#     title = "Tenure",
#     theme = theme(plot.title = element_text(hjust = 0.45))
#   )
# 
# p1 = train %>% 
#   count(NumOfProducts) %>% 
#   mutate(prop = n / sum(n)) %>% 
#   ggplot(aes(x = '', y = prop, fill = NumOfProducts)) +
#   geom_col(color = 'black') +
#   geom_text(aes(label = scales::percent(prop, suffix = '%', accuracy = 0.1)),
#             position = position_stack(vjust = 0.5),
#             size = 3.5) +
#   coord_polar(theta = 'y', start = pi/2) +
#   theme_void() +
#   theme(legend.position = "none")
# p2 = train %>% 
#   count(NumOfProducts) %>% 
#   mutate(prop = n / sum(n)) %>% 
#   ggplot(aes(x = NumOfProducts, y = n, fill = NumOfProducts)) +
#   geom_col(color = 'black') +
#   scale_y_continuous(name = "count", 
#                      breaks = seq(0, 80000, 10000)) +
#   theme_minimal() +
#   theme(legend.position = "left")
# (p1 + p2) +
#   plot_annotation(
#     title = "NumOfProducts",
#     theme = theme(plot.title = element_text(hjust = 0.45))
#   )
# 
# p1 = train %>% 
#   count(HasCrCard) %>% 
#   mutate(prop = n / sum(n)) %>% 
#   ggplot(aes(x = '', y = prop, fill = HasCrCard)) +
#   geom_col(color = 'black') +
#   geom_text(aes(label = scales::percent(prop, suffix = '%', accuracy = 0.1)),
#             position = position_stack(vjust = 0.5),
#             size = 3.5) +
#   coord_polar(theta = 'y', start = pi/2) +
#   theme_void() +
#   theme(legend.position = "none")
# p2 = train %>% 
#   count(HasCrCard) %>% 
#   mutate(prop = n / sum(n)) %>% 
#   ggplot(aes(x = HasCrCard, y = n, fill = HasCrCard)) +
#   geom_col(color = 'black') +
#   scale_y_continuous(name = "count", 
#                      breaks = seq(0, 120000, 20000)) +
#   theme_minimal() +
#   theme(legend.position = "left")
# (p1 + p2) +
#   plot_annotation(
#     title = "HasCrCard",
#     theme = theme(plot.title = element_text(hjust = 0.45))
#   )
# 
# p1 = train %>% 
#   count(IsActiveMember) %>% 
#   mutate(prop = n / sum(n)) %>% 
#   ggplot(aes(x = '', y = prop, fill = IsActiveMember)) +
#   geom_col(color = 'black') +
#   geom_text(aes(label = scales::percent(prop, suffix = '%', accuracy = 0.1)),
#             position = position_stack(vjust = 0.5),
#             size = 3.5) +
#   coord_polar(theta = 'y', start = pi/2) +
#   theme_void() +
#   theme(legend.position = "none")
# p2 = train %>% 
#   count(IsActiveMember) %>% 
#   mutate(prop = n / sum(n)) %>% 
#   ggplot(aes(x = IsActiveMember, y = n, fill = IsActiveMember)) +
#   geom_col(color = 'black') +
#   scale_y_continuous(name = "count", 
#                      breaks = seq(0, 80000, 10000)) +
#   theme_minimal() +
#   theme(legend.position = "left")
# (p1 + p2) +
#   plot_annotation(
#     title = "IsActiveMember",
#     theme = theme(plot.title = element_text(hjust = 0.45))
#   )

# train %>%
#   ggplot(aes(x = CreditScore, y = ..count.., fill = Exited)) +
#   geom_histogram(bins = 50, alpha = 0.8, color = 'white', position = 'identity') +
#   scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
#   scale_y_continuous(breaks = seq(0, 8000, 1000)) +
#   theme_minimal()
# train %>%
#   ggplot(aes(x = Age, y = ..count.., fill = Exited)) +
#   geom_histogram(bins = 50, alpha = 0.8, color = 'white', position = 'identity') +
#   scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
#   scale_y_continuous(breaks = seq(0, 16000, 2000)) +
#   scale_x_continuous(breaks = seq(20, 90, 10)) +
#   theme_minimal()
# train %>%
#   ggplot(aes(x = Balance, y = ..count.., fill = Exited)) +
#   geom_histogram(bins = 50, alpha = 0.8, color = 'white', position = 'identity') +
#   scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
#   scale_y_continuous(breaks = seq(0, 70000, 10000)) +
#   scale_x_continuous(breaks = seq(0, 250000, 50000)) +
#   theme_minimal()
# train %>%
#   ggplot(aes(x = EstimatedSalary, y = ..count.., fill = Exited)) +
#   geom_histogram(bins = 50, alpha = 0.8, color = 'white', position = 'identity') +
#   scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
#   scale_y_continuous(breaks = seq(0, 5000, 1000)) +
#   scale_x_continuous(breaks = seq(0, 200000, 25000)) +
#   theme_minimal()

# numeric_vars = train %>% select_if(is.numeric) %>% names()
# category_vars = train %>% select_if(is.factor) %>% names()
# var_names = c("Exited", numeric_vars, category_vars[-7])
# combs = expand.grid(y = var_names, x = var_names)
# combs = combs %>% 
#   mutate(mode = case_when(y %in% numeric_vars & x %in% numeric_vars   ~ '1',  # 1: 수치형 vs 수치형
#                           y %in% numeric_vars & x %in% category_vars  ~ '2',  # 2: 수치형 vs 범주형
#                           y %in% category_vars & x %in% numeric_vars  ~ '3',  # 3: 수치형 vs 범주형 >> 순서 바꿔야 함
#                           y %in% category_vars & x %in% category_vars ~ '4')) # 4: 범주형 vs 범주형
# my_cor = function(cnames, data) {
#   y = data %>% pull(cnames[1])
#   x = data %>% pull(cnames[2])
#   
#   if (cnames[3] %in% c('1', '2', '3')) {
#     if (cnames[3] == '3') {
#       y = data %>% pull(cnames[2])
#       x = data %>% pull(cnames[1])
#     }
#     suppressWarnings({
#       av = anova(lm(y ~ x))
#     })
#     return(av[[2]][1] / sum(av[[2]]))
#   } else {
#     return(cramerV(x = x, y = y))
#   }
# }
# combs$cor = apply(combs, 1, my_cor, data = train)
# combs$lab = sprintf("%.4f", combs$cor)
# combs = combs %>% 
#   mutate(x = factor(x, levels = var_names),
#          y = factor(y, levels = rev(var_names)))
# combs %>% 
#   ggplot(aes(x = x, y = y, fill = cor, label = lab)) +
#   geom_tile(color = 'white', width = 0.95, height = 0.95) +
#   geom_label(fill = 'white', size = 3) +
#   viridis::scale_fill_viridis(name = "Relationship", begin = 0.25) +
#   theme_minimal() +
#   theme(axis.text.x = element_text(angle = 45, hjust = 1),
#         axis.title = element_blank())

# 3. Model
## 3.1 Data Preparation
train = train %>% select(-c(id, CustomerId, Surname))
test = test %>% select(-c(id, CustomerId, Surname))
original = original %>% select(-c(id, CustomerId, Surname))

task = as_task_classif(train, target = "Exited", id = "train_data")
task$positive = "1"
task

task_test = as_task_classif(
  test %>% mutate(Exited = factor(NA, levels = 0:1)),
  target = "Exited", id = "test_data"
)
task_test$positive = "1"
task_test

graph = po("removeconstants", id = "removeconstants_preencoding") %>>%
  po("collapsefactors", no_collapse_above_prevalence = 0.01) %>>%
  po("encode", method = "one-hot", id = 'low_cardinality_encode') %>>%
  po("removeconstants", id = "removeconstants_postencoding")
# graph$plot()

# graph = po("encode", method = "one-hot")

# fig = magick::image_graph(width = 2500, height = 1500, res = 100, pointsize = 24)
# graph$plot()
# invisible(dev.off())
# magick::image_trim(fig)



task_encoded = graph$train(task)[[1]]
task_test_encoded = graph$predict(task_test)[[1]]
split = partition(task_encoded, ratio = 0.80, stratify = T)
task_encoded$set_row_roles(rows = split$test, roles = "test")


# task_encoded$head() %>% formattable()
# task_test_encoded$head() %>% formattable()

xgb_clf = lrn("classif.xgboost", predict_type = "prob",
              objective = "binary:logistic",
              verbose = 1,
              eval_metric = "auc",
              nrounds = 1000,
              early_stopping_rounds = 10,
              early_stopping_set = "test")
# xgb_clf$train(task_encoded)
# xgb_clf$model$best_iteration

# melt(xgb_clf$model$evaluation_log, id.vars = "iter", variable.name = "set", value.name = "auc") %>%
# ggplot(aes(x = iter, y = auc, group = set)) +
#   geom_line(aes(color = set), lwd = 2) +
#   # ylim(c(0.7, 1)) +
#   geom_vline(aes(xintercept = xgb_clf$model$best_iteration), color = "black", lwd = 1, lty = 'dashed') +
#   # scale_y_continuous(breaks = seq(0.5, 1, 0.05)) +
#   scale_color_manual(values = c("#f8766d", "#00b0f6"), labels = c("Train", "Test")) +
#   labs(x = "Rounds", y = "AUROC", color = "Set") +
#   theme_minimal()

# pred_base = xgb_clf$predict(task_encoded, row_ids = split$test)
# pred_base$confusion

# autoplot(pred_base, type = "roc") +
#   geom_line(lwd = 1.5) +
#   geom_abline(lwd = 1, lty = 'dotted')


# pred_base$confusion %>%
#   as.data.table() %>%
#   plot_confusion_matrix(target_col = "truth",
#                         prediction_col = "response",
#                         counts_col = "N",
#                         add_sums = T)

# 4. Hyperparameter Tuning & CV
instance_xgb = readRDS("./instance_xgb.rds")
# instance_xgb$result_learner_param_vals %>% unlist()

xgb_clf$param_set$values = instance_xgb$result_learner_param_vals

design = benchmark_grid(
  tasks = task_encoded,
  learners = xgb_clf,
  resamplings = rsmp("cv", folds = 10)
)
# tic()
# future::plan("multisession", workers = 10)
# bmr = benchmark(design)
# toc()
# bmr %>% write_rds("10foldCV_xgb.rds")
bmr = readRDS("./10foldCV_xgb.rds")

bmr$aggregate(msr("classif.auc"))
## 4.3 Final Model
tic()
xgb_clf$train(task_encoded)
toc()
pred_tuned = xgb_clf$predict(task_encoded, row_ids = task_encoded$row_roles$test)
# pred_tuned$confusion %>%
#   as.data.table() %>%
#   plot_confusion_matrix(target_col = "truth",
#                         prediction_col = "response",
#                         counts_col = "N",
#                         add_sums = T)


xgboost::xgb.plot.tree(model = xgb_clf$model, trees = 1)

## Feature Importance
xgb_exp = explain_mlr3(
  model = xgb_clf,
  data = task_encoded$data(rows = task_encoded$row_roles$test, cols = task_encoded$feature_names),
  y = as.numeric(task_encoded$data(rows = task_encoded$row_roles$test, cols = task_encoded$target_names)$Exited == 0),
  colorize = F,
  label = "XGBoost Explanation"
)
xgb_effect = model_parts(xgb_exp, B = 10, type = "raw")
plot(xgb_effect, show_boxplots = F)

# 6. Submission
submission$Exited = xgb_clf$predict(task_test_encoded)$prob[, 1]
submission %>% head() %>% formattable()
submission %>% fwrite("../data/playground-series-s4e1/submission_xgb.csv", row.names = F)
