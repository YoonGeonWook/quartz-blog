source("./utils.R")
source("./ggplot-theme.R")
source("./coef-plot.R")
source("./effect-plot.R")

# Assign libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(mlr)
  library(tidymodels)
})


bike <- read.csv("./data/bike-sharing-daily.csv", stringsAsFactors = F)

bike <- bike %>% 
  mutate(weekday = factor(weekday, levels = 0:6, labels = c('SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT')),
         holiday = factor(holiday, levels = c(0,1), labels = c('NO HOLIDAY', 'HOLIDAY')),
         workingday = factor(workingday, levels = c(0,1), labels = c('NO WORKING DAY', 'WORKING DAY')),
         season = factor(season, levels = 1:4, labels = c('WINTER', 'SPRING', 'SUMMER', 'FALL')),
         weathersit = factor(weathersit, levels = 1:3, labels = c('GOOD', 'MISTY', 'RAIN/SNOW/STORM')),
         mnth = factor(mnth, levels = 1:12, labels = c('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC')),
         yr = ifelse(yr == 0, 2011, 2012),
         days_since_2011 = day_diff(dteday, min(as.Date(dteday)))) %>% 
  # Denormalize weather features:
  # temp : Normalized temperature in Celsius.
  # The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
  mutate(temp = temp * (39 - (-8)) + (-8)) %>% 
  # atemp: Normalized feeling temperature in Celsius.
  # The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
  mutate(atemp = atemp * (50-(16)) + (16)) %>% 
  # windspeed: Normalized wind speed. 
  # The values are divided to max 67.
  mutate(windspeed = windspeed * 67) %>% 
  # hum: Normalized humidity. 
  # The values are divided to 100 (max)
  mutate(hum = hum * 100) %>% 
  select(-c(instant, dteday, registered, casual))

# get.bike.task = function(data_dir){
#   mlr::makeRegrTask(id='bike', data=get.bike.data(data_dir), target = 'cnt')
# }

bike.features.of.interest = c('season','holiday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed', 'days_since_2011')


X <- bike %>% select(all_of(bike.features.of.interest))
y <- bike %>% pull(cnt)
dat <- cbind(X,y)

mod <- lm(y ~ ., data = dat, x = T)
lm_summary <- summary(mod)$coefficients
tidy(mod) %>% 
  select(` ` = term, Weight = estimate, SE = std.error, `|t|` = statistic) %>% 
  mutate(`|t|` = abs(`|t|`))

### Visualization interpretation
#### 1. Weight Plot
coef_plot(mod) +
  scale_y_discrete("")

#### 2. Effect Plot
effect_plot(mod, dat) +
  scale_x_discrete("")

lm_summary['days_since_2011', 'Estimate']
lm_summary


### Explain Individual Predictions
predictions <- predict(mod)
effects <- get_effects(mod, dat)

effects_6 <- gather(effects[6, ])
predictions_mean <- mean(predictions)

names(predictions) <- NULL
pred_6 <- predictions[6]

df <- data.frame(feature = colnames(bike), value = t(bike[6, ]))
colnames(df) <- c("feature", "value")
df %>% write.table("clipboard", sep = "\t", row.names = F)

lm_summary
round(lm_summary[paste(df["workingday", "feature"], df["workingday", "value"], sep = ""), "Estimate"], 1)
round(as.numeric(as.character(df["temp", "value"])) * lm_summary[as.character(df["temp", "feature"]), "Estimate"], 1)

effect_plot(mod, dat) +
  geom_point(aes(x = key, y = value), color = 'red', data = effects_6, shape = 4, size = 4) +
  scale_x_discrete("") +
  ggtitle(sprintf("Predicted value for instance: %.0f\nAverage predicted value: %.0f\nActual value: %.0f",
                  pred_6, predictions_mean, y[6]))

dat[6, ]


library("glmnet")
X.d = model.matrix(y ~ . -1, data = X)
l.mod = glmnet(X.d, y)
plot(l.mod,  xvar = "lambda", ylab="Weights")


### Example with Lasso
extract.glmnet.effects <- function(betas, best.index){
  data.frame(beta = betas[, best.index])
}

n.features <- apply(l.mod$beta, 2, function(x){sum(x!=0)})
extract.glmnet.effects(l.mod$beta, max(which(n.features==2))) %>% 
  rownames_to_column() %>% 
  write.table("clipboard", sep = "\t", row.names = F)


extract.glmnet.effects(l.mod$beta, max(which(n.features==5))) %>% 
  rownames_to_column() %>% 
  write.table("clipboard", sep = "\t", row.names = F)
