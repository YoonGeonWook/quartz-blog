source("./V0.R")

# XGBoost tuning
measure = msr("classif.auc")
search_space = ps(
  booster = p_fct(c("gbtree", "dart")), 
  tree_method = p_fct("hist"),
  eta = p_dbl(lower = 1e-04, upper = 1, logscale = T),
  colsample_bytree = p_dbl(lower = 0, upper = 1),
  gamma = p_dbl(lower = 1e-05, upper = 7, logscale = T),
  max_depth = p_int(lower = 1, upper = 15),
  subsample = p_dbl(lower = 1e-01, upper = 1),
  lambda = p_dbl(lower = 1e-4, upper = 1000, logscale = T),
  alpha = p_dbl(lower = 1e-4, upper = 1000, logscale = T),
  scale_pos_weight = p_dbl(lower = 1, upper = 10)
)

### BO building blocks
# bayesopt_ego = mlr_loop_functions$get("bayesopt_ego")
# surrogate = srlrn(
#   lrn("regr.km", 
#       covtype = "matern5_2", 
#       optim.method = "BFGS",
#       control = list(trace = F))
# )
# acq_function = acqf("ei")
# acq_optimizer = acqo(
#   optimizer = opt("random_search", batch_size = 100),
#   terminator = trm("stagnation", iters = 100, threshold = 1e-5)
# )
tuner_bo = tnr(
  "mbo"
  # loop_function = bayesopt_ego,
  # surrogate = surrogate,
  # acq_function = acq_function,
  # acq_optimizer = acq_optimizer
)

future::plan("multisession", workers = 10)
instance = tune(
  tuner = tuner_bo,
  task = task_encoded,
  learner = xgb_clf,
  resampling = rsmp("holdout"),
  measures = measure,
  search_space = search_space,
  term_evals = 25,
  callbacks = clbk("mlr3tuning.early_stopping")
)

write_rds(instance, file = "instance_xgb.rds")