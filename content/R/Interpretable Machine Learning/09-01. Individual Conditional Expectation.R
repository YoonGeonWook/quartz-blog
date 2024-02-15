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

# 1. Examples
## ice-cervical 
### ICE plot of cervical cancer probability by age.
### Each line represents one woman.
### For most women there is an increase in predicted cancer probability with increasing age.
### For some women with a predicted cancer probability above 0.4, the prediction does not change much at higher age.
set.seed(43)
cervical_subset_index <- sample(1:nrow(cervical), size = 300)
cervical_subset <- cervical %>% slice(cervical_subset_index)
cervical.task <- makeClassifTask(data = cervical, target = 'Biopsy')
mod <- mlr::train(learner = makeLearner(cl = 'classif.randomForest', id = 'cervical-rf', predict.type = 'prob'), task = cervical.task)
pred.cervical <- Predictor$new(model = mod, data = cervical_subset, class = 'Cancer')
ice <- FeatureEffect$new(predictor = pred.cervical, feature = "Age", method = 'ice')$plot() +
  scale_color_discrete(guide = 'none') +
  scale_y_continuous('Predicted cancer probability')

## ice-bike
### ICE plots of predicted bicycle rentals by weather conditions.
### The same effects can be observed as in the partial dependence plots.
set.seed(42)
bike.subset.index <- sample(1:nrow(bike), size = 300)
bike.subset <- bike %>% slice(bike.subset.index)
bike.task <- makeRegrTask(data = bike, target = 'cnt')
mod.bike <- mlr::train(learner = makeLearner(cl = 'regr.randomForest', id = 'bike-rf'), task = bike.task)
pred.bike <- Predictor$new(model = mod.bike, data = bike.subset)
p1 <- FeatureEffect$new(predictor = pred.bike, feature = 'temp', method = 'ice')$plot() +
  scale_x_continuous('Temperature') +
  scale_y_continuous('Predicted bicycle rentals')
p2 <- FeatureEffect$new(predictor = pred.bike, feature = 'hum', method = 'ice')$plot() +
  scale_x_continuous('Humidity') +
  scale_y_continuous('')
p3 <- FeatureEffect$new(predictor = pred.bike, feature = 'windspeed', method = 'ice')$plot() +
  scale_x_continuous('Windspeed') +
  scale_y_continuous('')
gridExtra::grid.arrange(p1, p2, p3, ncol = 3)

## Example
## ice-cervical-centered
### Centered ICE plot for predicted cancer probability by age.
### Lines are fixed to 0 at 14.
### Compared to age 14, the predictions for most women remain unchanged until the age of 45 where the predicted probability increases.
predictor <- Predictor$new(model = mod, data = cervical_subset, class = "Cancer")
ice <- FeatureEffect$new(predictor = predictor, feature = 'Age', center.at = min(cervical_subset$Age), method = 'pdp+ice')
ice$plot() +
  scale_color_discrete(guide = 'none') +
  scale_y_continuous(sprintf("Cancer probability difference to age %i", min(cervical_subset$Age)))

## ice-bike-centered
### Centered ICE plots of predicted number of bikes by weather condition.
### The lines show the differences in prediction compared to the prediction with the respective feature value at its observed minimum.
set.seed(43)
bike.subset.index <- sample(1:nrow(bike), 100)
bike.subset <- bike %>% slice(bike.subset.index)

predictor <- Predictor$new(model = mod.bike, data = bike.subset)
ytext1 <- sprintf("Different to prediction at temp = %.2f", min(bike$temp))
ice1 <- FeatureEffect$new(predictor = predictor, feature = 'temp', center.at = min(bike$temp), method = 'pdp+ice')$plot() +
  scale_y_continuous(ytext1)

ytext2 <- sprintf("Different to prediction at hum = %.2f", min(bike$hum))
ice2 <- FeatureEffect$new(predictor = predictor, feature = 'hum', center.at = min(bike$windspeed), method = 'pdp+ice')$plot() +
  scale_y_continuous(ytext2)

ytext3 <- sprintf("Different to prediction at windspeed = %.2f", min(bike$windspeed))
ice3 <- FeatureEffect$new(predictor = predictor, feature = 'hum', center.at = min(bike$windspeed), method = 'pdp+ice')$plot() +
  scale_y_continuous(ytext3)
gridExtra::grid.arrange(ice1, ice2, ice3, ncol = 3)
