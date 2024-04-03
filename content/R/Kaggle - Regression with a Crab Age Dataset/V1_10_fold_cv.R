source("./V1.R")

design = benchmark_grid(
  tasks = task,
  learners = list(lrn_gbm, lrn_xgb, lrn_lgbm, lrn_cb, stack_lad),
  resamplings = rsmp("cv", folds = 10)
)
bmr = benchmark(design)
write_rds(bmr, file = "bmr_0.rds")