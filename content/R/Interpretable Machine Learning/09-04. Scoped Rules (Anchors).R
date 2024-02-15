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
})

bike <- bike %>% select(-atemp)
set.seed(42)

# 3. Tabular Data Example
set.seed(1)
colPal = c("#555555","#DFAD47","#7EBCA9", "#E5344E", "#681885", "#d25d97", "#fd3c46", "#ff9a39", "#6893bf", "#42c3a8")

## 회귀 -> 분류 전환
bike$target <- factor(resid(lm(cnt ~ days_since_2011, data = bike)) > 0,
                      levels = c(F, T), labels = c("below", "above"))
bike$cnt <- NULL

# Make long factor levels shorter
levels(bike$weathersit)[3] <- "BAD"


bike.task <- makeClassifTask(data = bike, target = "target")

mod <- mlr::train(learner = makeLearner(cl = 'classif.randomForest',
                                        id = 'bike-rf'),
                  task = bike.task)
bikeDisc <- list(
  integer(),
  integer(),
  integer(),
  integer(),
  integer(),
  integer(),
  integer(),
  c(0, 7, 14, 21, 28),
  c(30, 60, 69, 92),
  c(5, 10, 15, 20, 25),
  integer()
)

bike.explainer <- anchors(bike, model = mod, target = "target",
                          bins = bikeDisc, tau = 0.9, batchSize = 1000,
                          beams = 1)
bike.explained.instances <- bike[c(40, 475, 610, 106, 200, 700), ]
bike.explanations <- anchors::explain(x = bike.explained.instances, explainer = bike.explainer)
saveRDS(bike.explanations, file = "./anchors/cached-anchors.RDS")

bike.explainer_edge <- anchors(x = bike, model = mod, target = "target", tau = 1,
                               batchSize = 1000, beams = 1, allowSuboptimalSteps = F)
bike.explained.instances_edge <- bike[c(452, 300), ]
bike.explanations_edge <- anchors::explain(x = bike.explained.instances_edge, explainer = bike.explainer_edge)
saveRDS(bike.explanations_edge, file = "./anchors/cached-anchors-edge.RDS")

## anchor1
### Anchors explaining six instances of the bike rental dataset.
### Each row represents one explanation or anchor, and each bar depicts the feature predicates contained by it.
### The x-axis displays a rule's precision, and a bar's thickness corresponds to its coverage.
### The "base" rule contains no predicates.
### These anchors show that the model considers the temperature for predictions.

bike.explanations <- readRDS("./anchors/cached-anchors.RDS")
plotExplanations(bike.explanations, colPal = colPal)

### cervical anchors
set.seed(1)
cervical.sampled.healthy <- cervical[sample(which(cervical$Biopsy == 'Healthy'), 20), ]
cervical.balanced <- rbind(cervical[cervical$Biopsy == 'Cancer', ], cervical.sampled.healthy)

set.seed(1)
cervical.task <- makeClassifTask(data = cervical, target = "Biopsy")
mod <- train(learner = makeLearner(cl = "classif.randomForest", id = "cervical-rf", 
                                   predict.type = "prob"),
             task = cervical.task)
cancer.explainer <- anchors(x = cervical, model = mod, beams = 1)
cancer.explanation <- anchors::explain(x = cervical[c(1,7), ], explainer = cancer.explainer)

set.seed(1)
cancer.explainer.balanced <- anchors(x = cervical.balanced, model = mod, tau = 1, beams = 2,
                                     delta = 0.05, epsilon = 0.05, batchSize = 1000,
                                     emptyRuleEvaluations = 1000)
cancer.explanation.balanced <- anchors::explain(x = cervical.sampled.healthy[2:5, ], explainer = cancer.explainer.balanced)

## anchor3
printExplanations(explainer = cancer.explainer, explanations = cancer.explanation)
plotExplanations(explanations = cancer.explanation, colPal = colPal)
