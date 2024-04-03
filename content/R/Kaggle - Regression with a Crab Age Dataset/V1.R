# 현재 script 저장되어 있는 경로 지정
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

# Introduce the requred packages
library(tidyverse)
library(mlr3verse)
library(data.table)
library(mlr3mbo)
library(mlr3tuningspaces)
library(mlr3tuning)
library(bbotk)
library(ggplot2)
library(ggcorrplot)
library(patchwork)
library(skimr)
library(quantreg)
library(checkmate)
library(tictoc)
set.seed(123)

# Reduce annoying logging
# lgr::get_logger("mlr3")$set_threshold("warn")
# lgr::get_logger("bbotk")$set_threshold("warn")

# Load data
train = fread("../data/playground-series-s3e16/train.csv")
test = fread("../data/playground-series-s3e16/test.csv")
original = fread("../data/playground-series-s3e16/CrabAgePrediction.csv")
submission = fread("../data/playground-series-s3e16/sample_submission.csv")

sprintf("Dimension of the train synthetic dataset: (%s, %s)", nrow(train), ncol(train))
sprintf("Dimension of the test synthetic dataset: (%s, %s)", nrow(test), ncol(test))
sprintf("Dimension of the original dataset: (%s, %s)", nrow(original), ncol(original))
sprintf("Dimension of the submission dataset: (%s, %s)", nrow(submission), ncol(submission))

source("./my_skim.R")

# my_skim2(train)
# my_skim2(test)
# my_skim2(original)

# EDA
p1 = ggplot(train, aes(x = Age)) +
  geom_density(fill = "steelblue",  alpha = 0.4) +
  scale_x_continuous(breaks = seq(0, 30, 5)) +
  labs(title = "Competition Dataset")
p2 = ggplot(original, aes(x = Age)) +
  geom_density(fill = "orange",  alpha = 0.4) +
  scale_x_continuous(breaks = seq(0, 30, 5)) +
  labs(title = "Original Dataset")
# (p1 + p2) &
#   theme_minimal() &
#   plot_layout(nrow = 1)

p1 = ggcorrplot(cor(train %>% select(-c(id, Sex))), type = "lower", lab = T, colors = c("yellow", "orange", "red"), lab_col = "white", tl.srt = 45)
p2 = ggcorrplot(cor(original %>% select(-c(Sex))), type = "lower", lab = T, colors = c("yellow", "orange", "red"), lab_col = "white", tl.srt = 45)
# gridExtra::grid.arrange(p1, p2, nrow = 1)



# 중복 체크
cat(
  " There are", nrow(train), "observations in the train competition dataset\n", 
  "There are", train %>% select(-id) %>% n_distinct(), "unique observations in the train competition dataset\n",
  "There are", train %>% select(-c(id, Age)) %>% n_distinct(), "unique observations (only features) in the train competition dataset"
)

cat(
  " There are", nrow(test), "observations in the test competition dataset\n", 
  "There are", test %>% select(-id) %>% n_distinct(), "unique observations in the test competition dataset"
)

cat(
  " There are", nrow(original), "observations in the original dataset\n", 
  "There are", original %>% n_distinct(), "unique observations in the original dataset"
)

# Relationship b/w `Sex` and `Age`
p1 = ggplot(train %>% mutate(Sex = factor(Sex, levels = c("I", "M", "F"))), aes(x = Sex, y = Age, fill = Sex)) + 
  geom_boxplot(width = 0.5) +
  labs(title = "Competition Dataset")
p2 = ggplot(original %>% mutate(Sex = factor(Sex, levels = c("I", "M", "F"))), aes(x = Sex, y = Age, fill = Sex)) + 
  geom_boxplot(width = 0.5) +
  labs(title = "Original Dataset")
# (p1 + p2) *
#   scale_fill_viridis_d(end = 0.8) *
#   scale_y_continuous(breaks = seq(0, 30, 5)) *
#   theme_minimal() *
#   plot_layout(nrow = 1)

# Relationship b/w `Shell Weight` and `Age`
p1 = ggplot(train, aes(x = `Shell Weight`, y = Age)) +
  geom_point(shape = 21, color = 'white', fill = "steelblue", size = 2) +
  labs(title = "Competition Dataset")
p2 = ggplot(original, aes(x = `Shell Weight`, y = Age)) +
  geom_point(shape = 21, color = 'white', fill = "orange", size = 2) +
  labs(title = "Original Dataset")
# (p1 + p2) *
#   scale_y_continuous(breaks = seq(0, 30, 5)) *
#   theme_minimal() *
#   plot_layout(nrow = 1)

# Relationship b/w `Diameter` and `Age`
p1 = ggplot(train, aes(x = Diameter, y = Age)) +
  geom_point(shape = 21, color = 'white', fill = "steelblue", size = 2) +
  labs(title = "Competition Dataset")
p2 = ggplot(original, aes(x = Diameter, y = Age)) +
  geom_point(shape = 21, color = 'white', fill = "orange", size = 2) +
  labs(title = "Original Dataset")
# (p1 + p2) *
#   scale_y_continuous(breaks = seq(0, 30, 5)) *
#   theme_minimal() *
#   plot_layout(nrow = 1)

# Base modeling 1.0 -------------------------------------------------------

train = train %>% 
  mutate(generated = 1)
test = test %>% 
  mutate(generated = 1)
original = original %>% 
  mutate(generated = 0)
train = train %>% select(-id) %>% 
  rbind(original)

test_baseline= test %>% select(-id)

# Sex was found to be a factor variable
# XGBoost cannot process factorial data, into numeric data
train[Sex == "F", Sex := 0]
train[Sex == "M", Sex := 1]
train[Sex == "I", Sex := 2]
train$Sex = as.numeric(train$Sex)

# The test also needs to be processed
test_baseline[Sex == "F", Sex := 0]
test_baseline[Sex == "M", Sex := 1]
test_baseline[Sex == "I", Sex := 2]
test_baseline$Sex = as.numeric(test$Sex)

colnames(train) = c(paste0("V", 1:8), "Age", "generated")
colnames(test_baseline) = c(paste0("V", 1:8), "generated")



# Set the task
task = as_task_regr(train, target = "Age")

lrn_gbm = lrn("regr.gbm", distribution = 'laplace', n.trees = 1000)

instance_gbm = readRDS("./instance_gbm.rds")
instance_gbm$result
unlist(instance_gbm$result_learner_param_vals)


lrn_xgb = lrn("regr.xgboost",
              nrounds = 1000,
              objective = "reg:absoluteerror",
              eval_metric = "mae",
              verbose = 2)
instance_xgb = readRDS("./instance_xgb.rds")
instance_xgb$result
unlist(instance_xgb$result_learner_param_vals)

lrn_lgbm = lrn("regr.lightgbm",
               num_iterations = 1000,
               objective = "regression_l1",
               boosting = "gbdt")
instance_lgbm = readRDS("./instance_lgbm.rds")
instance_lgbm$result
unlist(instance_lgbm$result_learner_param_vals)

lrn_cb = lrn("regr.catboost",
             iterations = 1000,
             loss_function = "MAE",
             bootstrap_type = "Bayesian")
instance_cb = readRDS("./instance_cb.rds")
instance_cb$result
unlist(instance_cb$result_learner_param_vals)

lrn_gbm$param_set$values = instance_gbm$result_learner_param_vals
lrn_xgb$param_set$values = instance_xgb$result_learner_param_vals
lrn_xgb$param_set$values$device = "cuda"

lrn_lgbm$param_set$values = instance_lgbm$result_learner_param_vals
lrn_cb$param_set$values = instance_cb$result_learner_param_vals


source("./LearnerRegrRQ.R")
po_gbm_cv = po("learner_cv", learner = lrn_gbm, resampling.folds = 2, id = "gbm_cv")
po_xgb_cv = po("learner_cv", learner = lrn_xgb, resampling.folds = 2, id = "xgboost_cv")
po_lgbm_cv = po("learner_cv", learner = lrn_lgbm, resampling.folds = 2, id = "lightgbm_cv")
po_cb_cv = po("learner_cv", learner = lrn_cb, resampling.folds = 2, id = "catboost_cv")

gr_level_0 = gunion(list(po_gbm_cv, po_xgb_cv, po_lgbm_cv, po_cb_cv))
gr_combined = gr_level_0 %>>% po("featureunion")

# Super learner: LAD regressor
lrn_lad = LearnerRegrRQ$new()
lrn_lad$param_set$values = list(tau = 0.5)

stack_lad = gr_combined %>>% po("learner", lrn_lad)
# fig = magick::image_graph(width = 1500, height = 1000, res = 100, pointsize = 24)
stack_lad$plot(horizontal = T)
# invisible(dev.off())
# magick::image_trim(fig)


# Generalization Performance 
stack_lad = as_learner(stack_lad)
stack_lad$id = "stacking"

results = readRDS("./oof_mae_1.rds")

for (i in 1:10) {
  cat("---------------------------------------------------------------\n")
  cat("Fold", i, "==> GBM oof MAE is          ==>", results$gbm_cv_scores[[i]], "\n")
  cat("Fold", i, "==> XGBoost oof MAE is      ==>", results$xgb_cv_scores[[i]], "\n")
  cat("Fold", i, "==> LightGBM oof MAE is     ==>", results$lgbm_cv_scores[[i]], "\n")
  cat("Fold", i, "==> Catboost oof MAE is     ==>", results$cb_cv_scores[[i]], "\n")
  cat("Fold", i, "==> LAD ensemble oof MAE is ==>", results$ens_cv_scores[[i]], "\n")
  cat("---------------------------------------------------------------\n")
}

data.frame(GBM = mean(unlist(results$gbm_cv_scores)), XGBoost = mean(unlist(results$xgb_cv_scores)),
           LightGBM = mean(unlist(results$lgbm_cv_scores)), CatBoost = mean(unlist(results$cb_cv_scores)),
           `LAD Ensemble` = mean(unlist(results$ens_cv_scores))) %>% 
  pivot_longer(everything()) %>% 
  mutate(name = factor(name, levels = rev(c("GBM", "XGBoost", "LightGBM", "CatBoost", "LAD.Ensemble")))) %>% 
  ggplot(aes(x = name, y = value, fill = name)) +
  geom_col(color = "white") +
  geom_text(aes(label = round(value, 6)), hjust = -0.1) +
  scale_y_continuous(breaks = seq(0, 1.6, 0.2)) +
  ylim(c(0, 1.5)) +
  ggsci::scale_fill_nejm() + 
  coord_flip() +
  theme_minimal() +
  theme(legend.position = 'none', axis.title.y = element_blank()) +
  labs(y = "CV_score")

bmr = readRDS("./bmr_0.rds")
bmr$aggregate(msr("regr.mae"))[, .(learner_id, regr.mae)]

gbm_preds_test = data.frame(results$gbm_preds) %>% apply(1, mean)
xgb_preds_test = data.frame(results$xgb_preds) %>% apply(1, mean)
lgbm_preds_test = data.frame(results$lgbm_preds) %>% apply(1, mean)
cb_preds_test = data.frame(results$cb_preds) %>% apply(1, mean)
ens_preds_test = data.frame(results$ens_preds) %>% apply(1, mean)

submission$Age = as.integer(round(gbm_preds_test))
submission %>% write.csv("../data/playground-series-s3e16/GBM_V1_submission.csv", row.names = F)

submission$Age = as.integer(round(xgb_preds_test))
submission %>% write.csv("../data/playground-series-s3e16/XGBoost_V1_submission.csv", row.names = F)

submission$Age = as.integer(round(lgbm_preds_test))
submission %>% write.csv("../data/playground-series-s3e16/LightGBM_V1_submission.csv", row.names = F)

submission$Age = as.integer(round(cb_preds_test))
submission %>% write.csv("../data/playground-series-s3e16/Catboost_V1_submission.csv", row.names = F)

submission$Age = as.integer(round(ens_preds_test))
submission %>% write.csv("../data/playground-series-s3e16/LAD_Ensemble_V1_submission.csv", row.names = F)

