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

bike <- bike %>% select(-atemp)
set.seed(42)

# 2. Example
## surrogate-bike
### The terminal nodes of a surrogate tree that approximates the predictions of svm trained on the bike rental dataset.
### The distribution in the nodes show that the surrogate tree predicts a higher number of rented bikes when temperature is above 13 degrees Celsius and when the day was later in the 2 year period (cut point at 435 days).
bike.task <- makeRegrTask(data = bike, target = 'cnt')
mod.bike <- mlr::train(learner = makeLearner(cl = 'regr.svm'), task = bike.task)

pred.bike <- Predictor$new(model = mod.bike, data = bike %>% select(-cnt))
tree <- TreeSurrogate$new(predictor = pred.bike)
tree$plot()

#### R-squared (variance explained)
tree$r.squared


## surrogate-cercical 
### The terminal nodes of a surrogate tree that approximates the prediction of a random forest trained on the cervical cancer dataset.
### The counts in the nodes show the frequency of the black box models classifications in the nodes.
cervical.task <- makeClassifTask(data = cervical, target = "Biopsy")
mod.cervical <- mlr::train(learner = makeLearner(cl = 'classif.randomForest', predict.type = 'prob'), task = cervical.task)

pred.cervical <- Predictor$new(model = mod.cervical, data = cervical %>% select(-Biopsy), type = 'prob')
tree.cervical <- TreeSurrogate$new(predictor = pred.cervical, maxdepth = 2)
tree.cervical$plot() +
  theme(strip.text.x = element_text(size = 8))
pred.tree.cervical <- predict(tree.cervical, newdata = cervical) %>% select(Cancer)
pred.cervical <- getPredictionProbabilities(pred = predict(mod.cervical, task = cervical.task))

tree.cervical$r.squared
