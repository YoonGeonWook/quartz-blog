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
})
# 1. PDP-based Feature Importance
# 2. Examples
## pdp-bike
### PDPs for the bicycle count prediction model and temperature, humidity and wind speed. 
### The largest differences can be seen in the temperature.
### The hotter, the more bikes are rented.
### This trend goes up to 20 degrees Celsius, then flattens and drops slightly at 30.
### Marks on the x-axis indicate the data distribution.

bike.task <- makeRegrTask(data = bike, target = "cnt")
mod.bike <- mlr::train(mlr::makeLearner(cl = 'regr.randomForest', id = 'bike-rf'), bike.task)

pred.bike <- Predictor$new(mod.bike, data = bike)
pdp <- FeatureEffect$new(predictor = pred.bike,
                         feature = "temp", 
                         method = "pdp")
p1 <- pdp$plot() +
  scale_x_continuous("Temperature", limits = c(0, NA)) +
  scale_y_continuous('Predicted number of bikes', limits = c(0, 5500))

pdp$set.feature('hum')

p2 <- pdp$plot() +
  scale_x_continuous("Humidity", limits = c(0, NA)) +
  scale_y_continuous("", limits = c(0, 5500))

pdp$set.feature("windspeed")

p3 <- pdp$plot() +
  scale_x_continuous("Wind speed", limits = c(0, NA)) +
  scale_y_continuous("", limits = c(0, 5500))

gridExtra::grid.arrange(p1, p2, p3, ncol = 3)

## pdp-bike-cat
### PDPs for the bike count prediction model and the season.
### Unexpectedly all seasons show similar effect on the model predictions, only for winter the model predicts fewer bicycle rentals.
pdp <- FeatureEffect$new(predictor = pred.bike,
                         feature = "season", 
                         method = "pdp")

pdp$results %>% 
  ggplot() +
  geom_col(aes(x = season, y = .value), fill = default_color, width = 0.3) +
  scale_x_discrete("Season") +
  scale_y_continuous("Predicted number of bikes", limits = c(0, 5500))

## pdp-cervical
### PDPs of cancer probability based on age and years with hormonal contraceptives.
### For age, the PDP shows that probability is low until 40 and increases after.
### The more years on hormonal contraceptives the higher the predicted cancer risk, especially after 10 years.
### For both features not many data points with large values were available, so the PD estiimates are less reliable in those regions.
cervical.task <- makeClassifTask(data = cervical, target = "Biopsy")
mod <- mlr::train(learner = mlr::makeLearner(cl = "classif.randomForest", id = "cervical-rf", predict.type = "prob"), 
                  task = cervical.task)
pred.cervical <- Predictor$new(model = mod,
                               data = cervical,
                               class = "Cancer")
pdp <- FeatureEffect$new(predictor = pred.cervical, 
                         feature = "Age", 
                         method = "pdp")
p1 <- pdp$plot() +
  scale_x_continuous(limits = c(0, NA)) +
  scale_y_continuous("Predicted cancer probability", limits = c(0, 0.4))

pdp$set.feature("Hormonal.Contraceptives..years.")
p2 <- pdp$plot() +
  scale_x_continuous("Years on hormonal contraceptives", limits = c(0, NA)) +
  scale_y_continuous("", limits = c(0, 0.4))

gridExtra::grid.arrange(p1, p2, ncol = 2)

## pdp-cervical-2d
### PDP of cancer probability and the interaction of age and number of pregnancies.
### The plot shows the increase in cancer probability at 45.
### For ages below 25, women who had 1 or 2 pregnancies have a lower predicted cancer risk, compared with women who had 0 or more than 2 pregnancies.
### But be careful when drawing conclusions : This might just be a correlation and not causal!
pdp <- FeatureEffect$new(predictor = pred.cervical, 
                         feature = c("Age", "Num.of.pregnancies"), 
                         method = "pdp")
pdp$plot(rug = T, show.data = T) +
  scale_fill_viridis(option = "D")
