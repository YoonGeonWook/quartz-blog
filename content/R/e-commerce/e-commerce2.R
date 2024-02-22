source("./e-commerce.R")

# 04. Build the recipe & workflow
## 04-01. Pre-processing data with recipes
# Assign libraries
suppressPackageStartupMessages({
  library(tidyverse)  # contains dplyr, ggplot and our data set
  library(vip)        # variable importance plots
  library(finetune)   # package for more advanced hyperparameter tuning  
  library(doParallel) # parallelisation package   
  library(tictoc)     # measure how long processes take
  library(tidymodels) # the main tidymodels packages all in one place - loaded last to overwrite any conflicts
  library(ggsci)
  library(patchwork)
  library(moments)
  library(GGally)
  library(corrplot)
  library(bonsai)
  library(workflowsets)
  library(viridis)
  library(iml)
  library(DALEX)
  library(DALEXtra)
  library(shapviz)
  library(treeshap)
})
tidymodels_prefer()

df_mart %>% 
  ggplot(aes(x=Country, fill = factor(target))) +
  geom_bar(position = "dodge") +
  theme_minimal() +
  coord_flip() +
  labs(fill = "Target", x = "Country", y = "Count")

df_mart %>% 
  count(Country) %>% 
  arrange(desc(n)) %>% 
  mutate(ratio = n/sum(n))
df_mart <- df_mart %>% 
  mutate(Country = ifelse(Country=="United Kingdom", "UK", "ETC")) 

df_mart <- df_mart %>% 
  mutate(across(c(Country, peak_time, season), factor)) %>% 
  mutate(target = factor(target))

set.seed(123)

#### Create train-test set
split <- initial_split(df_mart, prop = 0.7, strata = target)
#### Create folds to for resampling on the Train set
train <- training(split)
test <- testing(split)
split
skimr::skim(train %>% select(-c(bsym, CustomerID)))


categorical_vars <- train %>% 
  select(-c(bsym, CustomerID, target)) %>% 
  select_if(is.factor) %>% 
  names()
## Categorical variables
plot_categorical <- function(cat){
  cat <- ensym(cat)
  train %>%
    mutate(target = as.numeric(target) - 1) %>% 
    group_by(!!cat) %>%
    reframe(prop = mean(target, na.rm = T)) %>% 
    mutate(across(!!cat, ~reorder(.x, prop))) %>% 
    ggplot(aes(x = !!cat, y = prop, fill = !!cat)) +
    geom_col(color = 'black', alpha = 0.6) +
    geom_text(aes(label = scales::percent(prop, accuracy = 0.01)), vjust  = 1.5) +
    scale_color_nejm() +
    scale_fill_nejm() + 
    ggtitle(paste0("Target rate by level of ", as.character(cat))) +
    guides(fill = "none") +
    theme_light()
}
cat_p <- map(categorical_vars, plot_categorical)
cat_p[[1]] + cat_p[[2]] + cat_p[[3]]


#### 수치형 변수에 대해 왜도 계산
numeric_vars <- train %>% 
  select(-c(bsym, CustomerID)) %>% 
  select_if(is.numeric) %>% 
  names()

num_skew <- map(numeric_vars, ~ skewness(train %>% pull(.x))) %>% 
  unlist()
names(num_skew) <- numeric_vars
num_skew

map_dfr(numeric_vars, ~summary(train %>% pull(.x))) %>% 
  mutate(var = numeric_vars, .before = Min.)
  

basic_rec <- recipe(formula = target ~ ., data = train) 
basic_rec <- basic_rec %>% 
  update_role(bsym, CustomerID, new_role = "ID")


before_trans <- function(num){
  num <- ensym(num)
  basic_rec %>% 
    prep(train) %>% 
    bake(train) %>% 
    ggplot(aes(x = !!num)) + 
    geom_histogram(color = 'black', fill = 'skyblue') +
    theme_light() +
    theme(axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())
}
before_trans_plot <- map(numeric_vars, before_trans)
before_trans_plot[[1]]+before_trans_plot[[2]]+before_trans_plot[[3]]+
  before_trans_plot[[4]]+before_trans_plot[[5]]+before_trans_plot[[6]]+
  before_trans_plot[[7]]+before_trans_plot[[8]]+before_trans_plot[[9]]+
  before_trans_plot[[10]]+before_trans_plot[[11]] +
  plot_layout(ncol = 3) +
  plot_annotation(title = "Before Transformation",
                  theme = theme(plot.title = element_text(hjust = 0.5))) 


after_trans <- function(num){
  num <- ensym(num)
  basic_rec %>% 
    step_YeoJohnson(!!num) %>% # Yeo-Johnson 변환
    prep(train) %>% 
    bake(train) %>% 
    ggplot(aes(x = !!num)) + 
    geom_histogram(color = 'black', fill = 'skyblue') +
    theme_light() +
    theme(axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())
}
after_trans_plot <- map(numeric_vars, after_trans)
after_trans_plot[[1]]+after_trans_plot[[2]]+after_trans_plot[[3]]+
  after_trans_plot[[4]]+after_trans_plot[[5]]+after_trans_plot[[6]]+
  after_trans_plot[[7]]+after_trans_plot[[8]]+after_trans_plot[[9]]+
  after_trans_plot[[10]]+after_trans_plot[[11]] +
  plot_layout(ncol = 3) +
  plot_annotation(title = "After Yeo-Johnson Transformation",
                  theme = theme(plot.title = element_text(hjust = 0.5)))


#### Yeo-Johnson 변환 후 target 변수와의 관계 확인 - boxplot with violin plot
plot_numerical <- function(num){
  num <- ensym(num)
  basic_rec %>% 
    step_YeoJohnson(all_numeric_predictors()) %>% 
    prep(train) %>% 
    bake(train) %>% 
    ggplot(aes(x = target, y = !!num, fill = target)) +
    geom_violin(width=1) +
    geom_boxplot(width = 0.3, alpha=0.7) +
    # scale_fill_viridis(discrete = TRUE) +
    coord_flip() + 
    theme_light() +
    guides(color = 'none') +
    theme(legend.position = "none")
}
num_p <- map(numeric_vars, plot_numerical)
num_p[[1]]+num_p[[2]]+num_p[[3]]+
  num_p[[4]]+num_p[[5]]+num_p[[6]]+
  num_p[[7]]+num_p[[8]]+num_p[[9]]+
  num_p[[10]]+num_p[[11]] + 
  plot_layout(ncol = 3)


#### 변수별 상관관계 - ANOVA 설명 분산량
mycor <- function(cnames, dat){
  x.num <- dat %>% pull(cnames[1])
  x.cat <- dat %>% pull(cnames[2])
  
  suppressWarnings({
    av <- anova(lm(x.num ~ x.cat))
  })
  sqrt(av$`Sum Sq`[1] / sum(av$`Sum Sq`))
}

cnames <- basic_rec %>% 
  step_YeoJohnson(all_numeric_predictors()) %>% 
  prep(train) %>% 
  bake(train) %>% 
  mutate(target = factor(target)) %>% 
  select_if(is.numeric) %>% names()
combs <- expand.grid(y = cnames, 
                     x = setdiff(names(train %>% select(-c(bsym, CustomerID))), "target"))
combs$cor <- apply(combs, 1, mycor, 
                   dat = basic_rec %>% 
                     step_YeoJohnson(all_numeric_predictors()) %>% 
                     prep(train) %>% 
                     bake(train) )
combs$lab <- sprintf("%.2f", combs$cor)
forder <- c(cnames, setdiff(unique(combs$x), cnames))

combs <- combs %>% 
  mutate(x = factor(x, levels = forder),
         y = factor(y, levels = rev(cnames)))

combs %>% 
  ggplot(aes(x = x, y = y, fill = cor, label = lab)) +
  geom_tile() +
  geom_label(fill = "white", size = 3) +
  theme_light() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_x_discrete("") +
  scale_y_discrete("") +
  scale_fill_viridis("Variance\nexplained", begin = 0.2)


#### total_cnt 제거
basic_rec <- basic_rec %>% 
  step_rm(total_cnt)

#### Dummy encoding

basic_rec %>% 
  prep(train) %>% 
  bake(train) %>% 
  select_if(is.factor) %>% str()

basic_rec %>% 
  step_dummy(all_nominal_predictors()) %>% 
  prep(train) %>% 
  bake(train) %>% 
  select(starts_with("Country"), starts_with("peak_time"), starts_with("season"))

#### step_nzv()를 통해 near zero variance 변수 제거
basic_rec %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  prep(train) 

## 04-02. 모델 성능 비교
### Defining resampling schemes with `rsample`
set.seed(123)
train_cv <- train %>% 
  vfold_cv(v = 10, repeats = 5, strata = target)
train_cv %>% 
  tidy() %>% 
  group_by(Repeat, Fold, Data) %>% 
  count() %>% 
  print(n = 20)

### Model specifications with parsnip    
glmnet_spec <- 
  logistic_reg(mode = 'classification',
               engine = 'glmnet',
               penalty = 0.00001,
               mixture = 1)
rf_spec <- 
  rand_forest(mode = 'classification',
              engine = 'ranger')
lgbm_spec <- 
  boost_tree(mode = 'classification',
             engine = "lightgbm")

### Putting it all together with workflows 

glm_rec <- 
  basic_rec %>% 
  step_YeoJohnson(all_numeric_predictors()) %>%  
  step_dummy(all_nominal_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_nzv(all_predictors())

tree_rec <- 
  basic_rec %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors())

basic_wflows <- 
  bind_rows(
    workflow_set(preproc = list(dummy_trans = glm_rec),
                 models = list(glmnet = glmnet_spec),
                 cross = T),
    workflow_set(preproc = list(dummy = tree_rec),
                 models = list(RF = rf_spec,
                               LGBM = lgbm_spec),
                 cross = T)
  )
basic_wflows

# cl <- makePSOCKcluster(8)
# registerDoParallel(cl)
# getDoParWorkers()
# tic()
# set.seed(123)
# basic_models <- 
#   basic_wflows %>% 
#   workflow_map("fit_resamples",
#                resamples = train_cv,
#                verbose = T,
#                metrics = metric_set(f_meas, roc_auc))
# basic_models %>% saveRDS("./basic_models.rds")
# toc()
# stopCluster(cl)
# registerDoSEQ()
basic_models <- readRDS("./basic_models.rds")

basic_models %>% 
  rank_results()

#### 각 모형을 test set에 적합
glmnet_model <- basic_models %>% 
  extract_workflow(c("dummy_trans_glmnet")) %>% 
  last_fit(split, metrics = metric_set(f_meas, roc_auc))
RF_model <- basic_models %>% 
  extract_workflow(c("dummy_RF")) %>% 
  last_fit(split, metrics = metric_set(f_meas, roc_auc))
LGBM_model <- basic_models %>% 
  extract_workflow(c("dummy_LGBM")) %>% 
  last_fit(split, metrics = metric_set(f_meas, roc_auc))

#### Compare Test vs Resamples(Training)
train_perf <- basic_models %>% 
  rank_results() %>% 
  select(mean) %>% 
  mutate(model = rep(c("glmnet", "RF", "LGBM"), each = 2),
         metric = rep(c("f1", "roc_auc"), 3)) %>% 
  select(model, metric, .estimate = mean)



test_perf <- bind_rows(
  glmnet_model %>% 
    collect_metrics() %>% 
    select(.estimate),
  RF_model %>% 
    collect_metrics() %>% 
    select(.estimate),
  LGBM_model %>% 
    collect_metrics() %>% 
    select(.estimate)
) %>%
  mutate(model = rep(c("glmnet", "RF", "LGBM"), each = 2),
         metric = rep(c("f1", "roc_auc"), 3)) %>% 
  select(model, metric, .estimate)


before_tuning <- train_perf %>% 
  pivot_wider(names_from = metric, values_from = .estimate) %>% 
  left_join(
    test_perf %>% 
      pivot_wider(names_from = metric, values_from = .estimate),
    by = "model", suffix = c("_Train", "_Test"), 
  )
before_tuning

## 04-03. Hyperparameter tuning
### Tuning
glmnet_tune_spec <- 
  logistic_reg(mode = 'classification',
               engine = 'glmnet',
               penalty = tune(),
               mixture = tune())
RF_tune_spec <- 
  rand_forest(mode = 'classification',
              engine = 'ranger',
              trees = 1000L,
              mtry = tune(),
              min_n = tune())
LGBM_tune_spec <- 
  boost_tree(mode = 'classification',
             engine = 'lightgbm',
             trees = tune(),
             mtry = tune(),
             min_n = tune(),
             tree_depth = tune(),
             learn_rate = tune(),
             loss_reduction = tune())
RF_tune_spec %>% 
   extract_parameter_set_dials()
LGBM_tune_spec %>% 
  extract_parameter_set_dials()

RF_tune_spec %>%
  extract_parameter_dials("mtry")


RF_params <- 
  RF_tune_spec %>% 
  extract_parameter_set_dials() %>% 
  update(mtry = finalize(mtry(),
                         tree_rec %>% prep() %>% bake(train)))
LGBM_params <- 
  LGBM_tune_spec %>% 
  extract_parameter_set_dials() %>% 
  update(mtry = finalize(mtry(),
                         tree_rec %>% prep() %>% bake(train)))
tune_wflows <-
  bind_rows(
    workflow_set(preproc = list(dummy_trans = glm_rec),
                 models = list(glmnet = glmnet_tune_spec),
                 cross = T),
    workflow_set(preproc = list(dummy = tree_rec),
                 models = list(RF = RF_tune_spec,
                               LGBM = LGBM_tune_spec),
                 cross = T)
  )

tune_wflows <- tune_wflows %>% 
  option_add(param_info = RF_params, id = "dummy_RF") %>% 
  option_add(param_info = LGBM_params, id = "dummy_LGBM")

#### Racing anova method를 통한 hyperparameter tuning
# cl <- makePSOCKcluster(8)
# registerDoParallel(cl)
# getDoParWorkers()
# tic()
# race_ctrl <- control_race(verbose_elim = T,
#                           save_pred = T,
#                           save_workflow = T,
#                           parallel_over = "everything")
# race_results <- 
#   tune_wflows %>% 
#   workflow_map("tune_race_anova",
#                seed = 1234,
#                resamples = train_cv,
#                grid = 50,
#                control = race_ctrl,
#                metrics = metric_set(f_meas, roc_auc),
#                verbose = T)
# race_results %>% saveRDS("./race_results.rds")
# toc()
# stopCluster(cl)
# registerDoSEQ()

race_results <- readRDS("./race_results.rds")

race_results %>% 
  rank_results(rank_metric = 'f_meas')

race_results %>% 
  rank_results(rank_metric = 'f_meas', select_best = T)
race_results %>% 
  rank_results(rank_metric = 'roc_auc', select_best = T)

  
race_results %>% 
  rank_results() %>%
  mutate(model_id = paste(wflow_id, str_sub(.config, -2), sep = "_")) %>% 
  select(wflow_id, model, .config, .metric, mean, std_err, rank, model_id) %>% 
  mutate(model_id = fct_reorder(model_id, -rank)) %>%
  {. ->> res} %>% 
  ggplot(aes(x = model_id, y = mean, color = wflow_id)) +
  geom_point(size = 4) +
  geom_errorbar(aes(x = model_id, color = wflow_id, 
                    ymin = mean - std_err,
                    ymax = mean + std_err),
                width = diff(range(res$rank))/25) +
  facet_wrap(~.metric, scales = "free", nrow = 2) +
  scale_color_nejm() +
  coord_flip() +
  guides(color = 'none') +
  labs(y = "performance", title = 'Racing method LGBM wins') +
  theme_light() +
  theme(plot.title = element_text(hjust = 0.5),
        axis.title = element_blank())


race_results %>% 
  extract_workflow_set_result(id = 'dummy_LGBM') %>% 
  select_best(metric = 'f_meas') 

best_results <- 
  race_results %>% 
  extract_workflow_set_result(id = 'dummy_LGBM') %>% 
  select_best(metric = 'f_meas')

LGBM_best_test_results <- race_results %>% 
  extract_workflow(id = 'dummy_LGBM') %>% 
  finalize_workflow(best_results) %>% 
  last_fit(split = split, metrics = metric_set(f_meas, roc_auc))
LGBM_best_test_results %>% 
  collect_metrics()

#### 각 모형의 최적 조합 저장
glmnet_best <- race_results %>% 
  extract_workflow_set_result(c("dummy_trans_glmnet")) %>% 
  select_best(metric = 'f_meas') 
RF_best <- race_results %>% 
  extract_workflow_set_result(c("dummy_RF")) %>% 
  select_best(metric = 'f_meas') 
LGBM_best <- race_results %>% 
  extract_workflow_set_result(c("dummy_LGBM")) %>% 
  select_best(metric = 'f_meas') 

#### 최적 조합으로 각 모형 train/test set에 적합
glmnet_test <- race_results %>% 
  extract_workflow("dummy_trans_glmnet") %>% 
  finalize_workflow(glmnet_best) %>% 
  last_fit(split = split, metrics = metric_set(f_meas, roc_auc))
RF_test <- race_results %>% 
  extract_workflow("dummy_RF") %>% 
  finalize_workflow(RF_best) %>% 
  last_fit(split = split, metrics = metric_set(f_meas, roc_auc))
LGBM_test <- race_results %>% 
  extract_workflow("dummy_LGBM") %>% 
  finalize_workflow(LGBM_best) %>% 
  last_fit(split = split, metrics = metric_set(f_meas, roc_auc))


#### Compare Test vs Resamples
train_perf_tuned <- race_results %>% 
  rank_results(rank_metric = "f_meas", select_best = T) %>% 
  select(mean) %>% 
  mutate(model = rep(c("LGBM_tuned", "RF_tuned", "glmnet_tuned"), each = 2),
         metric = rep(c("f1", "roc_auc"), 3)) %>% 
  select(model, metric, .estimate = mean)



test_perf_tuned <- bind_rows(
  glmnet_test %>% 
    collect_metrics() %>% 
    select(.estimate),
  RF_test %>% 
    collect_metrics() %>% 
    select(.estimate),
  LGBM_test %>% 
    collect_metrics() %>% 
    select(.estimate)
) %>%
  mutate(model = rep(c("LGBM_tuned", "RF_tuned", "glmnet_tuned"), each = 2),
         metric = rep(c("f1", "roc_auc"), 3)) %>% 
  select(model, metric, .estimate)


after_tuning <- train_perf_tuned %>% 
  pivot_wider(names_from = metric, values_from = .estimate) %>% 
  left_join(
    test_perf_tuned %>% 
      pivot_wider(names_from = metric, values_from = .estimate),
    by = "model", suffix = c("_Train", "_Test"), 
  ) 
options(pillar.sigfig = 4)
perf_table <- bind_rows(before_tuning, after_tuning)
perf_table[c(1,2,3,6,5,4), ] %>% 
  mutate(across(where(is.numeric), ~round(.x, 4))) %>% 
  write.table("clipboard", sep = "\t", row.names = F)
  
#### 최종 모형: Logistic model before tuning
glmnet_model %>% 
  collect_predictions() %>% 
  conf_mat(target, .pred_class)

test_metrics <- metric_set(accuracy, recall, precision)
glmnet_model %>% 
  collect_predictions() %>% 
  test_metrics(truth = target, estimate = .pred_class)
LGBM_test %>% 
  collect_predictions() %>% 
  test_metrics(truth = target, estimate = .pred_class)


## 04-04. Model Explanation
glmnet_model %>% 
  extract_fit_parsnip() %>% 
  vip::vi()
glmnet_model %>% 
  extract_fit_parsnip() %>% 
  vip::vip(num_features = 15, horizontal = T) + 
  theme_light()

X <- glm_rec %>% prep() %>% bake(train) %>% select(-c(bsym, CustomerID, target)) %>% as.matrix()
y <- glm_rec %>% prep() %>% bake(train) %>% pull(target)
model_tmp <- glmnet(X, y, family = 'binomial', alpha = 1, lambda = 1e-05)
pf <- function(m, X){
  predict(m, X, type = "response") %>% as.vector()
}

set.seed(123)
X_explain <- glm_rec %>% prep() %>% bake(train) %>% select(-c(bsym, CustomerID, target)) %>% 
  sample_n(500) %>% 
  as.matrix()
X_background <- glm_rec %>% prep() %>% bake(train) %>% select(-c(bsym, CustomerID, target)) %>% 
  sample_n(200) %>% 
  as.matrix()
system.time( # 4 minutes
  shap_values <- kernelshap::kernelshap(model_tmp, X = X_explain, bg_X = X_background, pred_fun = pf)
)

shp <- shapviz(shap_values)
sv_importance(shp, show_numbers = T, max_display = 14) +
  theme_light()
sv_importance(shp, kind = "bee") +
  theme_light()
  
sv_dependence(shp, v = colnames(shp$X)[1:6])
glmnet_model