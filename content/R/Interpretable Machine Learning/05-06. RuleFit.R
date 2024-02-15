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
  library(mgcv) # GAMs
  library(partykit) # DT basic
  library(rpart) # DT efficient
  library(OneR)  # OneR
  library(RWeka) # Sequential covering
  library(rJava) # rules 의 처리를 위해
  library(sbrl)  # SBRL 사용
  library(arules) # discretize() 사용
  library(pre)   # RuleFit 함수인 pre()를 이용
})

# 1. Interpretation & Example
## prepare-rulefit
X <- bike %>% select(all_of(bike.features.of.interest))

### round features so that table is better
X <- X %>% 
  mutate(across(c(temp, hum, windspeed), ~round(.x, 0)))

y <- bike %>% pull(cnt)
dat <- cbind(X,y)

mod <- pre(y ~ ., data = dat, maxdepth = 2, ntrees = 100)
coeffs <- coef(mod)
coeffs$description[is.na(coeffs$description)] <- coeffs$rule[is.na(coeffs$description)]
coeffs <- left_join(coef(mod), pre::importance(mod, plot = F)$baseimp)
coeffs <- coeffs[!is.na(coeffs$coefficient), ]
coeffs <- coeffs %>% 
  mutate(across(c(imp, coefficient), ~ round(.x, 1))) %>% 
  mutate(sd = round(sd, 2))
coeffs$rule <- NULL # rule 칼럼 삭제

coeffs <- coeffs %>% 
  filter(!is.na(imp)) %>% 
  arrange(desc(imp))

coeffs <- coeffs %>% 
  mutate(description = gsub("\\%", "", description),
         description = gsub("c\\(", "(", description))

coeffs %>% head(5) %>% select(-sd) %>% 
  select(Description = description, Weight = coefficient, Importance = imp) %>% 
  write.table("clipboard", sep = "\t", row.names = F)

nrow(coef(mod))
ncol(X)
coeffs %>% 
  filter(coefficient!=0) %>% 
  nrow()

## rulefit-importance
### Feature importance measures for a RuleFit model predicting bike counts.
### The most important features for the predictions were temperature and time trend.
pre::importance(mod)
