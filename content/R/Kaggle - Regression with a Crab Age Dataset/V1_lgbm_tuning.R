source("./V1.R")


## LGBM tuning
measures = msr("regr.mae")
search_space = ps(
  max_depth = p_int(lower = 1, upper = 15),
  num_leaves = p_int(lower = 2, upper = 1024),
  learning_rate = p_dbl(lower = 1e-04, upper = 0.1, logscale = T),
  bagging_fraction = p_dbl(lower = 0.4, upper = 1),
  feature_fraction = p_dbl(lower = 0.7, upper = 1),
  lambda_l1 = p_dbl(lower = 1e-08, upper = 10.0),
  lambda_l2 = p_dbl(lower = 1e-08, upper = 10.0)
)

### BO building blocks
bayesopt_ego = mlr_loop_functions$get("bayesopt_ego")
surrogate = srlrn(
  lrn("regr.km", 
      covtype = "matern5_2", 
      optim.method = "BFGS",
      control = list(trace = F))
)
acq_function = acqf("ei")
acq_optimizer = acqo(
  optimizer = opt("random_search", batch_size = 100),
  terminator = trm("stagnation", iters = 100, threshold = 1e-5)
)
tuner_bo = tnr(
  "mbo",
  loop_function = bayesopt_ego,
  surrogate = surrogate,
  acq_function = acq_function,
  acq_optimizer = acq_optimizer
)
future::plan("multisession", workers = 10)
instance = tune(
  tuner = tuner_bo,
  task = task,
  learner = lrn_lgbm,
  resampling = rsmp("holdout"),
  measures = measures,
  search_space = search_space,
  term_evals = 25
)

write_rds(instance, file = "instance_lgbm.rds")