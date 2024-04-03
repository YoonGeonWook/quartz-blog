library(mlr3verse)
library(mlr3tuningspaces)
library(mlr3tuning)
library(mlr3mbo)
library(bbotk)
set.seed(123)
tsk_moons = tgen("moons")
tsk_moons_train = tsk_moons$generate(100)
tsk_moons_test = tsk_moons$generate(1e6)

lrn_xgboost = lrn("classif.xgboost")

# inner_1: BO tuner
bayesopt_ego = mlr_loop_functions$get("bayesopt_ego")
surrogate = srlrn(
  lrn("regr.km", 
      covtype = "matern5_2", 
      optim.method = "BFGS",
      control = list(trace = F))
)
acq_function = acqf("ei")
# acq_optimizer = acqo(
#   optimizer = opt("nloptr", algorithm = "NLOPT_GN_ORIG_DIRECT"), 
#   terminator = trm("stagnation", iters = 100, threshold = 1e-5)
# )
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

# inner_2: Random Search 700회 평가
tuner_random = tnr("random_search")
trm_evals700 = trm("evals", n_evals = 700)

## inner resampling: holdout
rsmp_ho = rsmp("holdout")


at_bo = auto_tuner(
  tuner = tuner_bo,
  learner = lrn_xgboost,
  resampling = rsmp_ho,
  measure = msr("classif.ce"),
  search_space = lts("classif.xgboost.default")
)

at_rs = auto_tuner(
  tuner = tuner_random, 
  learner = lrn_xgboost,
  resampling = rsmp_ho,
  measure = msr("classif.ce"),
  terminator = trm_evals700,
  search_space = lts("classif.xgboost.default")
)

instance_no_nested = tune(
  tuner = tuner_random, 
  task = tsk_moons_train,
  learner = lrn_xgboost,
  resampling = rsmp_ho,
  measure = msr("classif.ce"),
  terminator = trm_evals700,
  search_space = lts("classif.xgboost.default")
)
insample = instance_no_nested$result_y

## outer resampling: 5-fold cv
rsmp_3cv = rsmp("cv", folds = 3)
## Nested resampling benchmark design
design = benchmark_grid(
  tasks = tsk_moons_train,
  learners = list(at_bo, at_rs),
  resamplings = rsmp_3cv
)
bmr = benchmark(design)
