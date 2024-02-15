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

## Prediction surface of a function with two features X_1 and X_2
x1 <- seq(-1, 1, length.out = 30)
x2 <- seq(-1, 1, length.out = 30)

f <- function(x1, x2){
  2 + exp(x1) - x2 + x1 * x2
}

dat <- expand.grid(x1 = x1, x2 = x2)
dat$y <- f(dat$x1, dat$x2)
mean(dat$y)
dat %>% 
  ggplot(aes(x = x1, y = x2, fill = y, z = y)) +
  geom_tile() +
  geom_contour() +
  scale_x_continuous(latex2exp::TeX(r'(Feature $X_1$)')) +
  scale_y_continuous(latex2exp::TeX(r'(Feature $X_2$)')) +
  scale_fill_continuous("Prediction")

## Decomposition of a function.

pred.fun <- function(model = NULL,  newdata){
  f(newdata$x1, newdata$x2)
}
pred <- Predictor$new(predict.function = pred.fun, data = dat, y = 'y')
p1 <- FeatureEffect$new(predictor = pred, feature = "x1", method = "ale")$plot(rug = F) +
  ggtitle(expression(paste(f[1], " main effect of ", X[1]))) +
  scale_x_continuous(latex2exp::TeX(r'(Feature $X_1$)'))
p2 <- FeatureEffect$new(predictor = pred, feature = "x2", method = "ale")$plot(rug = F) +
  ggtitle(expression(paste(f[2], " main effect of ", X[2]))) +
  scale_x_continuous(latex2exp::TeX(r'(Feature $X_2$)'))
interact <- FeatureEffect$new(predictor = pred, feature = c("x1", "x2"), method = "ale")
p12 <- interact$plot(rug = F) +
  geom_contour(aes(z = .ale), color = "black") +
  scale_fill_continuous("value", low = "blue", high = "yellow") +
  ggtitle(expression(paste(f[12], " interaction between ", X[1], " and ", X[2]))) +
  scale_x_continuous(latex2exp::TeX(r'(Feature $X_1$)')) +
  scale_y_continuous(latex2exp::TeX(r'(Feature $X_2$)'))
(p1+p2)/p12 + 
  patchwork::plot_layout(heights = c(1,2.5))
