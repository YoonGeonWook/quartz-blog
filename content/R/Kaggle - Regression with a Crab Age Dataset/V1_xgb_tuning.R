source("./V1.R")


## XGB tuning
measures = msr("regr.mae")
search_space = ps(
  booster = p_fct(c("gbtree", "dart")), 
  tree_method = p_fct("hist", depends = (booster %in% c("gbtree", "dart"))),
  # nrounds = p_int(lower = 100, upper = 3000),
  eta = p_dbl(lower = 1e-04, upper = 1, logscale = T),
  colsample_bytree = p_dbl(lower = 0, upper = 1),
  gamma = p_dbl(lower = 1e-05, upper = 7, logscale = T),
  max_depth = p_int(lower = 1, upper = 15),
  subsample = p_dbl(lower = 1e-01, upper = 1)
)
tuner_bo = tnr("mbo")
future::plan("multisession", workers = 10)
instance = tune(
  tuner = tuner_bo,
  task = task,
  learner = lrn_xgb,
  resampling = rsmp("holdout"),
  measures = measures,
  search_space = search_space,
  term_evals = 25
)

write_rds(instance, file = "instance_xgb.rds")