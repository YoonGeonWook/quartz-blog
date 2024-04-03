source("./V1.R")


## Catboost tuning
measures = msr("regr.mae")
search_space = ps(
  depth = p_int(lower = 1, upper = 15),
  learning_rate = p_dbl(lower = 1e-03, upper = 0.1, logscale = T),
  l2_leaf_reg = p_dbl(lower = 1e-04, upper = 100, logscale = T),
  random_strength = p_dbl(lower = 0, upper = 100),
  grow_policy = p_fct(c("SymmetricTree", "Depthwise", "Lossguide")),
  min_data_in_leaf = p_int(lower = 1, upper = 30, depends = (grow_policy %in% c("Depthwise", "Lossguide")))
)

tuner_bo = tnr("mbo")
future::plan("multisession", workers = 10)
instance = tune(
  tuner = tuner_bo,
  task = task,
  learner = lrn_cb,
  resampling = rsmp("holdout"),
  measures = measures,
  search_space = search_space,
  term_evals = 25
)

write_rds(instance, file = "instance_cb.rds")