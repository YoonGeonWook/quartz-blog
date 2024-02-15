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

# 1. Motivation and Intuition
## aleplot-motivation1
### Strongly correlated feature x1 and x2.
### To calculate the feature effect of x1 at 0.75, the PDP replaces x1 of all instances with 0.75, falsely assuming that the distribution of x2 at x1=0.75 is the same as the marginal distribution of x2 (vertical line).
### This results in unlikely combinations of x1 and x2 (e.g. x2=0.2 and x2=0.75), which the PDP uses for the calculation of the average effect.

set.seed(1)
n <- 100
intercept <- 0.75
x1 <- runif(n)
x2 <- x1 + rnorm(n, sd = 0.1)
df <- data.frame(x1, x2)

p <- df %>% 
  ggplot() +
  geom_point(aes(x = x1, y = x2)) +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank())
p.int <- p + 
  geom_vline(xintercept = intercept)
x1.dens <- density(x1)
x1.dens.df <- data.frame(dens = x1.dens$y, 
                         x = x1.dens$x)
p1 <- p.int +
  geom_path(data = x1.dens.df, aes(x = intercept - dens/10, y = x)) +
  ggtitle("Marginal distribution P(x2)") +
  scale_y_continuous(limits = c(-0.2, 1.2))

## aleplot-motivation2
### Strongly correlated features x1 and x2.
### M-plots average over the conditional distribution.
### Here the conditional distribution of x2 at x1=0.75.
### Averaging the local predictions leads to mixing the effects of both features.
set.seed(1)
n <- 100
intercept <- 0.75
x1 <- runif(n)
x2 <- x1 + rnorm(n, sd = 0.1)
df <- data.frame(x1, x2)

p <- df %>% 
  ggplot() +
  geom_point(aes(x = x1, y = x2)) +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank())

p.int <- p + 
  geom_vline(xintercept = intercept)
x1.dens <- density(x1)
x1.dens.df <- data.frame(dens = x1.dens$y, 
                         x = x1.dens$x)
  
p1 <- p.int +
  geom_path(data = x1.dens.df, aes(x = intercept - dens/10, y = x)) +
  ggtitle("Marginal distribution P(x2)") +
  scale_y_continuous(limits = c(-0.2, 1.2))

x1.dens.ale <- density(x1[(x1 > (intercept - 0.1)) & (x1 < (intercept + 0.1))])
x1.dens.ale.df <- data.frame(dens = x1.dens.ale$y, x = x1.dens.ale$x)

p2 <- p.int +
  geom_path(data = x1.dens.ale.df, 
            aes(x = intercept - dens/20, y = x)) +
  ggtitle(sprintf("Conditional distribution P(x2|x1=%.2f)", intercept)) + 
  scale_y_continuous(limits = c(-0.2, 1.2))

## aleplot-computation
### Calculation of ALE for feature x1, which is correlated with x2.
### First, we divide the feature into intervals (vertical lines).
### For the data instances (points) in an interval, we calculate the difference in the prediction when we replace the feature with the upper and lower limit of the interval (horizontal lines).
### These differences are later accumulted and centered, resulting in the ALE curve.
set.seed(12)
n <- 25
x1 <- runif(n)
x2 <- x1 + rnorm(n, sd = 0.1)
df <- data.frame(x1, x2)


p <- df %>% 
  ggplot() +
  geom_point(aes(x = x1, y = x2)) +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank())
p

grid.df <- data.frame(x1 = seq(0, 1, length.out = 6)[1:6], x2 = NA)
label.df <- grid.df[1:5, ]
label.df$x1 <- label.df$x1 + 0.1
label.df$x2 <- 0.95
label.df$label <- sprintf("N1(%i)", 1:5)

break.labels <- c(expression(z[0~","~1]),  expression(z[1~","~1]), expression(z[2~","~1]), expression(z[3~","~1]),
                  expression(z[4~","~1]), expression(z[5~","~1]))
diff.df <- df %>% 
  filter(x1 <= 0.8 & x1 > 0.6)
p + 
  geom_vline(data = grid.df,
             aes(xintercept = x1),
             linetype = 3) +
  scale_x_continuous(breaks = seq(0, 1, length.out = 6),
                     limits = c(0, 1), 
                     labels = break.labels) +
  geom_label(data = label.df, 
             aes(x = x1, y = x2, label = label)) +
  geom_segment(data = diff.df,
               aes(x = 0.6, xend = 0.8, y = x2, yend = x2),
               arrow = arrow(ends = "both", angle = 90, length = unit(0.07, "inches")))

## aleplot-computation-2d
### Calculation of 2D-ALE.
### We place a grid over the two features.
### In each grid cell we calculate the 2nd-order differences for all instance within.
### We first replace values of x1 and x2 with the values from the cell corners.
### If a, b, c and d represent the "corner"-predictions of a manipulated instance (as labeled in the graphic), then the 2nd-order difference is (d-c) - (b-a).
### The mean 2nd-order difference in each cell is accumulated over the grid and centered.
p <- df %>% 
  ggplot() +
  geom_point(aes(x=x1, y=x2)) +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank())

grid.df1 <- data.frame(x1 = seq(min(df$x1), max(df$x1), length.out=6),
                       x2 = NA)
grid.df2 <- data.frame(x2 = seq(min(df$x1), max(df$x2), length.out=6),
                       x1 = NA)

chosen.tile <- expand.grid(x1 = grid.df1$x1[4:5], x2 = grid.df2$x2[4:5])
chosen.tile2 <- data.frame(x = grid.df1$x1[4], xend = grid.df1$x1[5],
                           y = grid.df2$x2[4], yend = grid.df2$x2[5])

points.df <- df %>% 
  filter(x1 < grid.df1$x1[5] & x1 > grid.df1$x1[4] & x2 < grid.df2$x2[5] & x2 > grid.df2$x2[4])

p + 
  geom_vline(data = grid.df1,
             aes(xintercept = x1), 
             linetype = 3) +
  geom_hline(data = grid.df2,
             aes(yintercept = x2),
             linetype = 3) +
  geom_rect(data = chosen.tile2,
            aes(xmin = x, xmax = xend, ymin = y, ymax = yend),
            alpha = 0, color = "black", linewidth = 1.1) +
  geom_label(data = chosen.tile, 
             aes(x = x1, y = x2),
             label = letters[1:4]) +
  geom_point(data = points.df,
             aes(x = x1, y = x2),
             size = 3)

# 4. Examples
## correlation-problem
### Two features and the predicted outcome.
### The model predicts the sum of the two features (shaded backgroud), with the exception that if x1 is greater than 0.7 and x2 less than 0.3, the model always predicts 2.
### This area is far from the distribution of data (point cloud) and does not affect the performance of the model and also should not affect its interpretation.

set.seed(1)
n <- 25
x1 <- runif(n)
x2 <- x1 + rnorm(n, sd = 0.1)
df <- data.frame(x1, x2)
df$y <- x1 + x2

mod <- lm(y ~ ., data = df)

y.fun <- function(X.model, newdata){
  pred <- predict(X.model, newdata)
  pred[newdata$x1 > 0.7 & newdata$x2 < 0.3] <- 2
  pred
}

grid.dat <- expand.grid(x1 = seq(0,1, length.out = 20),
                        x2 = seq(0,1, length.out = 20))
grid.dat$predicted <- y.fun(mod, grid.dat)

df %>% 
  ggplot() +
  geom_tile(data = grid.dat,
            aes(x = x1, y = x2, fill = predicted)) +
  geom_point(aes(x = x1, y = x2), size = 3) +
  scale_fill_viridis("Model\nprediction", option = "D")

## correlation-pdp-ale-plot
### Comparison of the feature effects computed with PDP (upper row) and ALE (lower row).
### The PDP estimates are influenced by the odd/weird behavior of the model outside the data distribution (steep jumps in the plots).
### The ALE plots correctly identify that the ML model has a linear relationship b/w features and prediction, ignoring areas w/o data.
pred <- Predictor$new(mod, data = df, predict.fun = y.fun)
pdp <- FeatureEffect$new(predictor = pred, feature = "x1", method = "pdp")
pdp1 <- pdp$plot() + ggtitle("PDP")
pdp$set.feature("x2")
pdp2 <- pdp$plot() + ggtitle("PDP")

ale <- FeatureEffect$new(predictor = pred, feature = "x1", method = "ale")
ale1 <- ale$plot() + ggtitle("ALE")
ale$set.feature("x2")
ale2 <- ale$plot() + ggtitle("ALE")

gridExtra::grid.arrange(pdp1, pdp2, ale1, ale2)


## ale-bike-train
set.seed(42)
bike.task <- makeRegrTask(data = bike %>% select(-atemp), target = 'cnt')
mod.bike <- mlr::train(learner = makeLearner(cl = "regr.ctree"), task = bike.task)$learner.model

pred.bike <- Predictor$new(model = mod.bike,
                           data = bike, 
                           y = "cnt")

## ale-bike
### ALE plots for the bike prediction model by temperature, humidity and wind speed.
### The temperature has a strong effect on the prediction.
### The average prediction rises with increasing temperature, but falls again above 25 degrees Celcius.
### Humidity has a negative effect: Wehn above 60%, the higher the relative humidity, the lower the prediction.
### The wind speed does not affect the predictions much.
limits <- c(-1500, 800)
ale1 <- FeatureEffect$new(predictor = pred.bike, feature = "temp", method = "ale")$plot() +
  scale_x_continuous("Temperature") +
  scale_y_continuous("ALE", limits = limits)
ale2 <- FeatureEffect$new(predictor = pred.bike, feature = "hum", method = "ale")$plot() +
  scale_x_continuous("Humidity") +
  scale_y_continuous("", limits = limits)
ale3 <- FeatureEffect$new(predictor = pred.bike, feature = "windspeed", method = "ale")$plot() +
  scale_x_continuous("Wind Speed") +
  scale_y_continuous("", limits = limits)
gridExtra::grid.arrange(ale1, ale2, ale3, ncol = 3)

## ale-bike-cor
### The strength of the correlation b/w temperature, humidity and wind speed with all features, measured as the amount of variance explained, when we train a linear model with e.g. temperature to predict and season as feature.
### For temperature we observe -- not surprisingly -- a high correlation with season and month.
### Humidity correlates with weather situation.
mycor <- function(cnames, dat){
  x.num <- dat %>% pull(cnames[1])
  x.cat <- dat %>% pull(cnames[2])
  
  suppressWarnings({
    av <- anova(lm(x.num ~ x.cat))
  })
  sqrt(av$`Sum Sq`[1] / sum(av$`Sum Sq`))
}
cnames <- c("temp", "hum", "windspeed")
combs <- expand.grid(y = cnames, x = setdiff(colnames(bike %>% select(-atemp)), "cnt"))
combs$cor <- apply(combs, 1, mycor, dat = bike)
combs$lab <- sprintf("%.2f", combs$cor)
forder <- c(cnames, setdiff(unique(combs$x), cnames))

combs <- combs %>% 
  mutate(x = factor(x, levels = forder),
         y = factor(y, levels = rev(cnames)))

combs %>% 
  ggplot(aes(x = x, y = y, fill = cor, label = lab)) +
  geom_tile() +
  geom_label(fill = "white", size = 3) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_x_discrete("") +
  scale_y_discrete("") +
  scale_fill_viridis("Variance\nexplained", begin = 0.2)

## pdp-bike-compare
### PDPs for temperature, humudity and wind speed. 
### Compared to the ALE plots, the PDPs show a smaller decrease in predicted number of bikes for high temperature or high humidity.
### The PDP uses all data instances to calculate the effect of high temperature, even if they are, for example, instances with the season "winter".
### The ALE plots are more reliable.
pdp <- FeatureEffect$new(predictor = pred.bike, feature = "temp", method = "pdp")
p1 <- pdp$plot() + 
  scale_x_continuous("Temperature") +
  scale_y_continuous("Predicted number of rental bikes", limits = c(3700, 5300)) 

pdp$set.feature("hum")
p2 <- pdp$plot() +
  scale_x_continuous("Humidity") +
  scale_y_continuous("", limits = c(3000, 5500))

pdp$set.feature("windspeed")
p3 <- pdp$plot() +
  scale_x_continuous("Wind speed") +
  scale_y_continuous("", limits = c(3000, 5500))

gridExtra::grid.arrange(p1, p2, p3, ncol = 3)

## ale-bike-cat
### ALE plot for the categorical feature month. 
### The months are ordered by their similarity to each other, based on the distibutions of the other features by month.
### We observe that Jan, Mar and Apr, but especially Dec and Nov, have a lower effect on the predicted number of rented bikes compared to other months.
alecat1 <- FeatureEffect$new(predictor = pred.bike, feature = "mnth", method = "ale")

alecat1$results %>% 
  ggplot() +
  geom_col(aes(x = mnth, y = .value),
           fill = default_color, width = 0.3) +
  scale_x_discrete("") +
  scale_y_continuous("ALE of predicted bike rentals")


## ale-bike-2d
### ALE plot for the 2nd-order effect of humidity and temperature on the predicted number of rented bikes.
### Lighter shade indicates an above average and darker shade a below average prediction when the main effects are already taken into account.
### The plot reveals an interaction b/w temperature and humidity : Hot and humid weathre increases the prediction.
### In cold and humid weather an additional negative effect on the number of predicted bikes in shown.
FeatureEffect$new(predictor = pred.bike, feature = c("hum", "temp"), method = "ale", grid.size = 40)$plot() +
  scale_fill_gradient("ALE", low = "red", high = "yellow") +
  scale_x_continuous("Relative Humudity") +
  scale_y_continuous("Temperature") +
  scale_fill_viridis(option = "D")

## pdp-bike-vs-ale-2D
### PDP of the total effect of temperature and humidity on the predicted number of bikes.
### The plot combines the main effect of each of the features and theri interaction effect, as opposed to the 2D-ALE plot which only shows the interaction.
FeatureEffect$new(predictor = pred.bike, feature = c("hum", "temp"), method = "pdp")$plot() +
  scale_fill_gradient("Prediction", low = "red", high = "yellow") +
  scale_x_continuous("Relative Humudity") +
  scale_y_continuous("Temperature") +
  scale_fill_viridis(option = "D")


## ale-cervical-1D
### ALE plots for the effect of age and years with hormonal contraceptive on the predicted probability of cervical cancer.
### For the age feature, the ALE plot shows that the predicted cancer probability is low on average up to age 40 and increases after that.
### The number of years with hormonal contraceptives is associated with a higher predicted cancer risk after 8 years.
cervical.task <- makeClassifTask(data = cervical, target = "Biopsy")
mod <- mlr::train(learner = makeLearner(cl = "classif.randomForest", id = "cervical-rf", predict.type = "prob"), 
                  task = cervical.task)
pred.cervical <- Predictor$new(mod, data = cervical, class = "Cancer")
ale1 <- FeatureEffect$new(predictor = pred.cervical, feature = "Age", method = "ale")$plot()
ale2 <- FeatureEffect$new(predictor = pred.cervical, feature = "Hormonal.Contraceptives..years.", method = "ale")$plot() +
  scale_x_continuous("Years with hormonal contraceptives") +
  scale_y_continuous("")
gridExtra::grid.arrange(ale1, ale2, ncol =2)


## ale-cervical-2d
### ALE plot of the 2nd-order effect of number of pregnancies and age.
### The interpretation of the plot is a bit inconclusive, showing what seems like overfitting.
### For example, the plot shows an odd model behavior at age of 18-20 and more than 3 pregnancies (up to 5 percentage point increase in cancer probability).
### There are not many women in the data with this constellation of age and number of pregnancies (actual data are displayed as points), so the model is not severely penalized during the training for making mistakes for those women.
FeatureEffect$new(predictor = pred.cervical, feature = c("Age", "Num.of.pregnancies"), grid.size = 30)$plot(show.data = T) +
  scale_fill_gradient("ALE", low = "red", high = "yellow") +
  scale_y_continuous("Number of pregnancies") +
  scale_x_continuous("Age") +
  scale_fill_viridis(option = "D")


