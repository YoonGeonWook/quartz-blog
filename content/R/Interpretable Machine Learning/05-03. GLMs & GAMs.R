source("./utils.R")
source("./ggplot-theme.R")
source("./coef-plot.R")
source("./effect-plot.R")
source("./code.R")

# Assign libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(mlr)
  library(tidymodels)
  library(mgcv)
})

#### three-lm-problems
##### Three assumptions of the linear model (left side): 
##### Gaussian distibution of the outcome given the features, additivity (= no interactions) and linear relationship.
##### Reality usually does not adhere to those assumptions (right side): 
##### Outcome might have non-Gaussian distributions, features might interact and the relationship might be nonlinear.
theme_blank <- theme(
  axis.line = element_blank(),
  axis.text.x = element_blank(),
  axis.text.y = element_blank(),
  axis.ticks = element_blank(),
  axis.title.x = element_blank(),
  axis.title.y = element_blank(),
  legend.position = 'none',
  panel.background = element_blank(),
  panel.border = element_blank(),
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),
  plot.background = element_blank()
)
###### For the GLM
n <- 10000
df <- data.frame(x = c(rnorm(n), rexp(n, rate = 0.5)),
                 dist = rep(c("Gaussian", "Definitely Not Gaussian"), each = n))
df <- df %>% 
  mutate(dist = relevel(factor(dist), ref = "Gaussian"))
p.glm <- df %>% 
  ggplot() +
  geom_density(aes(x = x)) +
  facet_grid(. ~ dist, scales = 'free') +
  theme_blank
###### For the interaction
df <- data.frame(x1 = seq(-3, 3, length.out = n),
                 x2 = sample(c(1,2), size = n, replace = T))
df <- df %>% 
  mutate(y = 3+5*x1+(2-8*x1)*(x2==2),
         interaction = "Interaction")
df2 <- df %>% 
  mutate(y = 3+5*x1+0.5*(-8*x1)+2*(x2==2),
         interaction = "No Interaction")

df <- rbind(df, df2)
df <- df %>% 
  mutate(interaction = relevel(factor(interaction), ref = "No Interaction"),
         x2 = factor(x2))

p.interaction <- df %>% ggplot() +
  geom_line(aes(x=x1, y=y, group=x2, lty=x2)) +
  facet_grid(.~interaction) +
  theme_blank

###### For the gam
df <- data.frame(x = seq(0, 10, length.out=200))
df <- df %>% 
  mutate(y = 5+2*x,
         type = "Linear")
df2 <- df %>% 
  mutate(y = 3+2*x+3*sin(x),
         type = "Nonlinear")
df <- rbind(df, df2)

p.gam <- df %>% ggplot() +
  geom_line(aes(x=x, y=y)) +
  facet_grid(.~type) +
  theme_blank

gridExtra::grid.arrange(p.glm, p.interaction, p.gam)

### 1. Non-Gaussian Outcome: GLMs
#### Example
#### poisson-data
##### Simulated distribution of number fo daily coffees for 200 days.
###### simulate data where the normal linear model fails
n <- 200
df <- data.frame(stress = runif(n=n, min = 1, max = 10),
                 sleep = runif(n = n, min = 1, max = 10),
                 work = sample(c("YES", "NO"), size = n, replace = T))
df <- df %>% 
  mutate(lambda = exp(1*stress/10 - 2 * (sleep-5)/10 - 1*(work=="NO")))

df$y <- rpois(lambda = df$lambda, n = n)
df %>% str()
df %>% 
  count(y) %>% 
  ggplot() +
  geom_col(aes(x = y, y = n), fill = default_color, width = 0.3) +
  scale_x_discrete("Number of coffees on a given day") +
  scale_y_continuous("Number of days")

#### failing-linear-model
##### Predicted number of coffees dependent on stress, sleep and work.
##### The linear model predicts negative values.
df <- df %>% select(-lambda)
mod.gaus <- glm(y ~ ., data = df, x = T)
pred.gauss <- data.frame(pred = predict(mod.gaus), actual = df$y)
pred.gauss %>% 
  ggplot() +
  geom_histogram(aes(x = pred), fill = default_color, color = 'black') +
  scale_x_continuous("Predicted number of coffees") + 
  scale_y_continuous("Frequency")


#### linear-model-positive
##### Predicted number of coffees dependent on stress, sleep and work.
##### The GLM with Poisson assumption and log link is an appropriate model for this dataset.
mod.pois <- glm(y ~ ., data = df, x = T, family = poisson(link = 'log'))
pred.pois <- data.frame(pred = predict(mod.pois, type = 'response'),
                        actual = df$y)
pred.pois %>% 
  ggplot() +
  geom_histogram(aes(x = pred), fill = default_color, color = 'black') + 
  scale_x_continuous("Predicted number of coffees") + 
  scale_y_continuous("Frequency")

#### poisson-model-params
cc <- tidy(mod.pois) %>% select(term, beta = estimate, var.beta = std.error) %>% 
  mutate(exp.beta = exp(beta)) %>% 
  select(beta, exp.beta)
cc
cc <- cc %>% cbind(exp(confint(mod.pois)))
cc %>% 
  rename(weight = beta) %>% 
  mutate(across(where(is.numeric), ~round(.x, 2))) %>% 
  mutate(`exp(weight) [2.5%, 97.5%]` = paste0(exp.beta, " [", `2.5 %`,", ", `97.5 %`, "]")) %>% 
  select(c(1, 5)) %>% 
  write.table("clipboard", sep = "\t", row.names = T)


### 2. Interactions
#### data-frame
x = data.frame(work = c("Y", "N", "N", "Y"), temp = c(25, 12, 30, 5))
x %>% 
  write.table("clipboard", sep = "\t", row.names = F)

#### data-frame-lm-no-interaction
mod <- lm(1:4 ~ ., data=x)
model.tab <- model.matrix(mod) %>% data.frame()
colnames(model.tab)[1] <- "Intercept"
model.tab %>% 
  write.table("clipboard", sep = "\t", row.names = F)

#### data-frame-lm
mod <- lm(1:4 ~ work * temp, data = x)
model.tab <- data.frame(model.matrix(mod))
colnames(model.tab)[1] <- "Intercept"
model.tab %>% 
  write.table("clipboard", sep = "\t", row.names = F)


#### data-frame-lm-cat
x = data.frame(work = c("Y", "N", "N", "Y"), wthr = c("2", "0", "1", "2"))
x %>% write.table("clipboard", sep = "\t", row.names = F)

#### data-frame-lm-cat2
mod = lm(1:4 ~ work * wthr, data = x)
model.tab = data.frame(model.matrix(mod))
colnames(model.tab)[1] = c("Intercept")
model.tab %>% 
  write.table("clipboard", sep = "\t", row.names = F)

#### example-lm-interaction
X <- bike %>% select(all_of(bike.features.of.interest))
y <- bike %>% pull(cnt)
dat <- cbind(X, y)
mod <- lm(y ~ . + temp * workingday, data = dat, x = T)
lm_summary <- tidy(mod) %>% 
  rename(Estimate = estimate, `Std. Error`=std.error, `t value` = statistic, `Pr(>|t|)` = p.value) %>% 
  column_to_rownames(var = 'term')
lm_summary_print <- lm_summary
rownames(lm_summary_print) <- pretty_rownames(rownames(lm_summary_print))

rownames(lm_summary_print)[rownames(lm_summary_print) == "weathersitRAIN/SNOW/STORM"] = "weathersitRAIN/..."
lm_summary_print %>% 
  select(Weight = Estimate, `Std. Error`) %>% 
  bind_cols(confint(mod)) %>% 
  mutate(across(where(is.numeric), ~round(.x, 1))) %>% 
  write.table("clipboard", sep = "\t", row.names = T)

#### interaction-plot
##### The effect (including interaction) of temporature and working day on the predicted number of bikes for a linear model.
##### Effectively, we get two slopes for the temperature, one for each category of the working day feature.
interactions::interact_plot(mod, pred = "temp", modx = "workingday")


### 3. Nonlinear Effects - GAMs
#### nonliear-effects
#### Predicting the number of rented bicycles using only the temperature feature.
#### A linear model (top left) does not fit the data well.
#### One solution is to transform the feature with e.g. the logarithm (top right), 
#### categorize it (bottm left), which is usually a bad decistion, or use GAMs that can automatically fit a smooth curve for temperature (bottm right).
mod.simple <- lm(cnt ~ temp, data = bike)
bike.plt <- bike %>% 
  mutate(pred.lm = predict(mod.simple),
         log.temp = log(temp + 10))
mod.log = lm(cnt ~ log.temp, data = bike.plt)
bike.plt <- bike.plt %>% 
  mutate(pred.log = predict(mod.log),
         cat.temp = cut(temp, breaks = seq(min(temp), max(temp), length.out = 10), include.lowest = T))
mod.cat <- lm(cnt ~ cat.temp, data = bike.plt)
bike.plt <- bike.plt %>% 
  mutate(pred.cat = predict(mod.cat))

mod.gam <- gam(cnt ~ s(temp), data = bike)
bike.plt <- bike.plt %>% 
  mutate(pred.gam = predict(mod.gam))
bike.plt <- bike.plt %>% 
  select(starts_with("pred."), temp, cnt) %>% 
  pivot_longer(cols = starts_with("pred."), names_to = "variable", values_to = "value", cols_vary = "slowest")

model.type <- c(pred.lm = "Linear model",
                pred.log = "Linear model with log(temp + 10)",
                pred.cat = "Linear model with categorized temp",
                pred.gam = "GAM")

bike.plt %>% 
  mutate(variable = factor(variable, levels = c("pred.lm", "pred.log", "pred.cat", "pred.gam"), 
                           labels = model.type)) %>% 
  ggplot() +
  geom_point(aes(x = temp, y = cnt), size = 1, alpha = 0.3) +
  geom_line(aes(x = temp, y = value), lwd = 1.2, color = 'blue') +
  facet_wrap(~variable) +
  scale_x_continuous("Temperature (temp)") + 
  scale_y_continuous("(Predicted) Number of rented bikes")
  
#### splines-df
###### fit GAM with less splines
mod.gam <- gam(cnt ~ s(temp, k = 5), data = bike)
model.matrix(mod.gam) %>% head() %>% 
  round(2)

#### splines
##### To smoothly model the temperature effect, we used 4 splines basis functions.
##### Each temperature value is mapped to (here) 4 spline basis values.
##### If an instance has a temperature of 30 ℃, the value for the 1st spline basis feature is -1, for the 2nd 0.7, for the 3rd -0.8 and for the 4th 1.7.
mm <- model.matrix(mod.gam) %>% data.frame()
colnames(mm) <- dimnames(model.matrix(mod.gam))[[2]]
mm <- mm %>% 
  mutate(temp = bike$temp)
mm2 <- mm %>% 
  pivot_longer(cols = c(everything(), -temp), names_to = "variable", values_to = "value") %>% 
  mutate(variable = factor(variable))

mm2 %>% 
  filter(variable != "(Intercept)") %>% 
  ggplot() +
  geom_line(aes(x = temp, y = value)) +
  facet_wrap(~variable) +
  scale_x_continuous("Temperature") +
  scale_y_continuous("Value of spline basis feature")
mm2 %>% arrange(temp) %>% View()

data.frame(weight = coef(mod.gam) %>% round(2)) %>% 
  rownames_to_column() %>% 
  write.table("clipboard", sep = "\t", row.names = F)

#### spline-curve
##### GAM feature effect of the temperature for predicting the number of rented bikes (temperature used as the only feature).
plot(mod.gam)
