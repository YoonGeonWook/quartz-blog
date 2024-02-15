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

# 2. Should I Compute Importance on Training or Test Data?
## prepare-garbage-svm
set.seed(1)
n <- 200; p <- 50
X <- data.frame(matrix(rnorm(n*p), nrow = n))
y <- rnorm(n)
dat <- cbind(X, y)
tsk <- makeRegrTask(data = dat, target = "y")

X2 <- data.frame(matrix(rnorm(n*p), nrow = n))
y2 <- rnorm(n)
dat2 <- cbind(X2, y=y2)
tsk2 <- makeRegrTask(data = dat2, target = 'y')

mod <- mlr::train(learner = makeLearner(cl = "regr.svm"), task = tsk)
pred <- predict(mod, task = tsk)
perf1 <- performance(pred = pred, measures = list(mlr::mae))

pred2 <- predict(mod, task = tsk2)
perf2 <- performance(pred2, measures = list(mlr::mae))

## feature-imp-sim
### Distributions of feature importance values by data type.
### An SVM was trained on a regression dataset with 50 random features and 200 instances.
### The SVM overfits the data: Feature importance based on the training data shows many important features.
### Computed on unseen test data, the feature importances are close to a ratio of one (= unimportant).
pred <- Predictor$new(model = mod, data = dat, y = "y")
imp <- FeatureImp$new(predictor = pred, loss = "mae")

pred2 <- Predictor$new(model = mod, data = dat2, y = "y")
imp2 <- FeatureImp$new(predictor = pred2, loss = "mae")

imp$results$dat.type <- "Training data"
imp2$results$dat.type <- "Test data"

imp.data <- rbind(imp$results, imp2$results)

imp.data %>% 
  ggplot() +
  geom_boxplot(aes(x = dat.type, y = importance)) +
  scale_y_continuous("Feature importance of all features") +
  scale_x_discrete("")

## The case for training data
## garbage-svm-mvp
max.imp <- imp$results %>% 
  slice_max(importance)
## garbage-svm-pdp
### PDP of feature X42, which is the most important feature according to the feature importance based on the training data.
### The plot shows the SVM depends on this feature to make predictions.
pdp <- FeatureEffect$new(predictor = pred2, feature = max.imp$feature, method = "pdp")
FeatureEffect$new(predictor = pred, feature = max.imp$feature, method = "pdp")$plot()
pdp$plot()

# 3. Example and Interpretation
## Cervical cancer (classification)
task <- makeClassifTask(data = cervical, target = "Biopsy", positive = "Cancer")
learner <- makeLearner(cl = "classif.randomForest", predict.type = "prob")
mod <- mlr::train(learner = learner, task = task)
predictor <- Predictor$new(model = mod, data = cervical %>% dplyr::select(-Biopsy), y = (cervical$Biopsy == "Cancer"), class = "Cancer")
auc_error <- function(actual, predicted){
  1 - Metrics::auc(actual = actual, predicted = predicted)
}
importance <- FeatureImp$new(predictor = predictor, loss = auc_error)
imp.dat <- data.frame(importance$results %>% dplyr::select(feature, permutation.error, importance))
most_imp <- imp.dat %>% 
  slice_max(importance) %>% 
  pull(feature)

## importance-cervical
### The importance of each of the features for predicting cervical cancer with a random forest.
### The most important feature was Hormonal.Contraceptives..years. .
### Permuting Hormonal.Contraceptives..years. resulted in an increase 1-AUC by a factor of 6.33.
importance$plot() +
  scale_x_continuous("Feature importance (loss: 1-AUC)") +
  scale_y_discrete("")

## Bike sharing (regression)
bike2 <- bike %>% dplyr::select(-atemp)
task <- makeRegrTask(data = bike2, target = 'cnt')
learner <- makeLearner(cl = 'regr.svm')
mod <- mlr::train(learner = learner, task = task)
predictor <- Predictor$new(model = mod, data = bike2 %>% dplyr::select(-cnt), y = bike %>% pull(cnt))
importance <- FeatureImp$new(predictor = predictor, loss = 'mae')
imp.dat <- importance$results
best <- which(imp.dat$importance == max(imp.dat$importance))
worst <- which(imp.dat$importance == min(imp.dat$importance))

## importance-bike
### The importance for each of the features in predicting bike counts with a svm.
### The most important feature a feature was temp, the least important was holiday.
importance$plot() +
  scale_y_discrete("")
