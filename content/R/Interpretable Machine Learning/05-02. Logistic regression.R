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
})

### What is Wrong with Linear Regression for Classification?

#### linear-class-threshold
#### A linear model classifies tumors as malignant (1) or begign (0) given their size. 
#### The lines show the prediction of the linear model.
#### For the data on the left, we can use 0.5 as classification threshold.
#### After introducting a few more maligant tumor cases, the regression line shifts and a threshold of 0.5 no longer separates the classes. 
#### Points are slightly jittered to reduce over-plotting.
df <- data.frame(x = c(1,2,3,8,9,10,11,9),
                 y = c(0,0,0,1,1,1,1,0),
                 case = '1) 0.5 threshold ok')
df_extra <- data.frame(x = c(df$x, 7,7,7,20,19,5,5,4,4.5),
                       y = c(df$y, 1,1,1,1,1,1,1,1,1),
                       case = '2) 0.5 threshold not ok')

df.lin.log <- rbind(df, df_extra)
df.lin.log %>% 
  ggplot(aes(x=x, y=y)) +
  # position_jitter(): 겹치는 데이터를 시각적으로 구분하기 위해 사용
  geom_point(position = position_jitter(width = 0, height = 0.02)) + 
  geom_smooth(method = 'lm', se=F) +
  my_theme() +
  scale_y_continuous('', breaks = c(0, 0.5, 1), labels = c('benign tumor', '0.5', 'malignant tumor'), limits = c(-0.1, 1.3)) +
  scale_x_continuous('Tumor size') +
  facet_grid(. ~ case) +
  geom_hline(yintercept = 0.5, linetype = 3)

### 2. Theory
#### logistic-function
##### The logistic function. 
##### It ouputs numbers b/w 0 and 1. 
##### At input 0, it outputs 0.5.
logistic <- function(x){1 / (1+exp(-x))}
x <- seq(from = -6, to = 6, length.out = 100)
df <- data.frame(x = x,
                 y = logistic(x))
df %>% 
  ggplot(aes(x=x, y=y)) +
  geom_line() + 
  my_theme()

#### logistic-class-threshold
##### The logistic regression model finds the correct decision boundary b/w maligant & benign depending on tumor size. 
##### The line is the logistic function shifted and squeezed to fit the data.
logistic1 <- glm(y ~ x, family = binomial(), data = df.lin.log %>% filter(case == '1) 0.5 threshold ok'))
logistic2 <- glm(y ~ x, family = binomial(), data = df.lin.log %>% filter(case != '1) 0.5 threshold ok'))

lgrid <- data.frame(x = seq(from = 0, to = 20, length.out = 100))
lgrid$y1_pred <- predict(logistic1, newdata = lgrid, type = 'response')
lgrid$y2_pred <- predict(logistic2, newdata = lgrid, type = 'response')

lgrid.m <- data.frame(lgrid %>% pivot_longer(cols = c(y1_pred, y2_pred), names_to = 'variable', cols_vary = "slowest"))
colnames(lgrid.m) <- c("x", "case", "value")

lgrid.m <- lgrid.m %>% 
  mutate(case = ifelse(case == 'y1_pred', '1) 0.5 threshold ok', '2) 0.5 threshold ok as well'))
df.lin.log <- df.lin.log %>% 
  mutate(case = ifelse(case == '2) 0.5 threshold not ok', '2) 0.5 threshold ok as well', case))

df.lin.log %>% 
  ggplot(aes(x=x, y=y)) +
  geom_line(aes(x=x, y=value), data = lgrid.m, color = 'blue', lwd = 1) +
  geom_point(position = position_jitter(width = 0, height = 0.02)) +
  my_theme() +
  scale_y_continuous('Tumor class', breaks = c(0, 0.5, 1), labels = c('benign tumor', '0.5',  'malignant tumor'), limits = c(-0.1,1.3)) +
  scale_x_continuous('Tumor size') +
  facet_grid(. ~ case) +
  geom_hline(yintercept=0.5, linetype = 3)

### 4. Example
#### logistic-example

neat_curvical_names <- c('Intercept', 'Hormonal contraceptives y/n',
                         'Smokes y/n', 'Num. of pregnancies',
                         'Num. of diagnosed STDs',
                         'Intrauterine device y/n')
mod <- glm(Biopsy ~ Hormonal.Contraceptives + Smokes + Num.of.pregnancies +STDs..Number.of.diagnosis + IUD,
           data = cervical, family = binomial())
coef.table <- tidy(mod) %>% 
  select(` ` = term, Estimate = estimate, `Std. Error` = std.error) %>% 
  mutate(`Odds ratio` = exp(round(Estimate, 2)), 
         ` ` = neat_curvical_names)


coef.table %>% 
  mutate(across(where(is.numeric), ~round(.x, 2))) %>% 
  relocate(`Odds ratio`, .after = Estimate) %>% 
  write.table("clipboard", sep = "\t", row.names = F)
