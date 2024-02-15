source("./utils.R")
source("./ggplot-theme.R")
source("./coef-plot.R")
source("./effect-plot.R")
source("./code.R")
source("./lime.R")
source("./get-text-classifier.R")

# Assign libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(mlr) # ML in r
  library(tidymodels)
  library(iml) # Interpretable Machine Learning in R
  library(tictoc)
  library(gridExtra)
  library(tm) # A framework for text mining applications within R
  library(anchors) # anchor in R
  library(jsonlite) # anchorOnR
  library(BBmisc)   # anchorOnR
  library(uuid)     # anchorOnR
  library(magrittr) # anchorOnR
  library(caret)
})

bike <- bike %>% select(-atemp)
set.seed(42)

# 2. Examples and Interpretation
## shapley-cervical-prepare
ntree <- 30
cervical.x <- cervical %>% select(-Biopsy)
model <- caret::train(x = cervical.x,
                      cervical$Biopsy,
                      method = 'rf', ntree = ntree, maximise = F)
predictor <- Predictor$new(model = model, class = "Cancer", data = cervical.x, type = 'prob')
instance_indicies <- 326 # 326번 인스턴스
x.interest <- cervical.x %>% slice(instance_indicies)

avg.prediction <- predict(model, type = "prob")[, 'Cancer'] %>% mean()
actual.prediction <- predict(model, newdata = x.interest, type = 'prob')['Cancer']
diff.prediction <- actual.prediction - avg.prediction


## shapley-cervical-plot
### Shapley values for a woman in the cervical cancer dataset.
### With a prediction of 0.57, this woman's cancer probability is 0.54 above the average prediction of 0.03.
### The number of diagnosed STDs increased the probability the most.
### The sum of contributions yields the difference b/w actual and average prediction (0.54).
shapley2 <- Shapley$new(predictor = predictor, x.interest = x.interest, sample.size = 100)
shapley2$plot() +
  scale_y_continuous("Feature value contribution") +
  ggtitle(sprintf("Actual prediction: %.2f\nAverage prediction: %.2f\nDifference: %.2f", actual.prediction, avg.prediction, diff.prediction)) +
  scale_x_discrete("")

## shapley-bike-prepare
ntree <- 30
bike.train.x <- bike %>% select(-cnt)
model <- caret::train(bike.train.x, bike$cnt,
                      method = 'rf', ntree=ntree, maximise = F)
predictor <- Predictor$new(model, data = bike.train.x)

# 관심 인스턴스 : 295번, 285번
instance_indices <- c(295, 285)

avg.prediction <- predict(model) %>% mean()
actual.prediction <- predict(model, newdata = bike.train.x %>% slice(instance_indices[2]))
diff.prediction <- actual.prediction - avg.prediction
x.interest <- bike.train.x %>% slice(instance_indices[2], )

## shapley-bike-plot
### Shapley values for day 285.
### With a predicted 2413 rental bikes, this day is -2092 below the average prediction of 4505.
### The weather situation and humidity had the largest negative contributions.
### The temperature on this day had a positive contribution.
### The sum of Shapley values yields the difference of actual and average prediction (-2092).
shapley2 <- Shapley$new(predictor, x.interest = x.interest)
shapley2$plot() +
  scale_y_continuous("Feature value contribution") +
  ggtitle(sprintf("Actual prediction: %.0f\nAverage prediction: %.0f\nDifference: %.0f", actual.prediction, avg.prediction, diff.prediction))  +
  scale_x_discrete("")
