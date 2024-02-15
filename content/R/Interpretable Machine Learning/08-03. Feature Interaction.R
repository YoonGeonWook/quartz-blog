source("./utils.R")
source("./ggplot-theme.R")
source("./coef-plot.R")
source("./effect-plot.R")
source("./code.R")

# Assign libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(mlr) # ML in r
  library(tidymodels)
  library(iml) # Interpretable Machine Learning in R
  library(tictoc)
})

# 3. Examples
## interaction-bike
### The interaction strenth (H-statistic) for each feature with all other features for a support vector machine predicting bicycle rentals.
### Overall, the interaction effects b/w the features are very weak (below 10% of variance explained per feature).
bike.task <- makeRegrTask(data = bike %>% select(-atemp), target = "cnt")
mod.bike <- train(learner = makeLearner(cl = 'regr.svm', id = 'bike-svm'), task = bike.task)

pred.bike <- Predictor$new(model = mod.bike, data = bike[setdiff(colnames(bike %>% select(-atemp)), "cnt")])
ia <- Interaction$new(predictor = pred.bike, grid.size = 50)
ia$plot() +
  scale_y_discrete("")

## interaction-cervical-prep
set.seed(42)
cervical.task <- makeClassifTask(data = cervical, target = "Biopsy")
mod <- mlr::train(learner = makeLearner(cl = "classif.randomForest", id = 'cervical-rf', predict.type = 'prob'), task = cervical.task)

## interaction-cervical
### Due to long running time and timeouts on TravisCI, this has to be run locally.
#### The interaction strength (H-stat) for each feature with all other features for a random forest predicting the probability of cervical cancer.
#### The years on hormonal contraceptives has the highest relative interaction effect with all other features, followed by the number of pregnancies.
pred.cervical <- Predictor$new(model = mod, data = cervical, class = "Cancer")
tic()
ia1 <- Interaction$new(predictor = pred.cervical, grid.size = 100)
toc()
ia1$plot() +
  scale_y_discrete("")
