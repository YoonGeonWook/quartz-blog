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
  library(arules) # discretize() 사용용
})

# 05-05. Decision Rules
## 1. Learn Rules from a Single Feature; OneR
#### OneR-freq-table1
value <-  factor(c("high", "high", "high", "medium", "medium", "medium", "medium", "low", "low", "low"), levels = c("low", "medium", "high"))

df <- data.frame(
  location = c("good", "good", "good", "bad", "good", "good", "bad", "bad", "bad", "bad"),
  size = c("small", "big", "big", "medium", "medium", "small", "medium", "small", "medium", "small"), 
  pets = c("yes", "no", "no", "no", "only cats", "only cats", "yes", "yes", "yes", "no"),
  value = value
)
df %>% 
  write.table("clipboard", sep = "\t", row.names = F)

value.f <- factor(paste("value=", value, sep = ""), levels = c("value=low", "value=medium", "value=high"))

#### OneR-freq-table2
table(paste0("location=", df[,"location"]), value.f) %>% 
  write.table("clipboard", sep = "\t")

table(paste0("size=", df[,"size"]), value.f) %>% 
  write.table("clipboard", sep = "\t")

table(paste0("pets=", df[,"pets"]), value.f) %>% 
  write.table("clipboard", sep = "\t")

#### Example
#### oner-cervical
rule <- OneR(Biopsy ~ ., data = cervical)

rule.to.table <- function(rule){
  dt <- data.frame(x = names(rule$rules), 
                   prediction = unlist(rule$rules))
  colnames(dt) <- c(rule$feature, "prediction")
  return(dt)
}

rule.to.table(rule) %>% 
  write.table("clipboard", sep = "\t", row.names = F)

#### oner-cervical-confusion
tt <- table(paste0("Age=", bin(cervical$Age)), cervical$Biopsy)
cn <- colnames(tt)
tt <- data.frame(matrix(tt, ncol = 2), row.names = rownames(tt))
colnames(tt)
tt %>% as.data.frame.table() %>% 
  pivot_wider(names_from = Var2, values_from = Freq) %>% 
  relocate(Cancer, .before = Healthy) %>% 
  mutate(`P(Cancer)` = round(Cancer / (Cancer+Healthy), 2)) %>% 
  column_to_rownames(var = 'Var1') %>% 
  rename(`# Cancer` = Cancer, `# Healthy` = Healthy) %>% 
  write.table("clipboard", sep = "\t", row.names = T)

#### oner-bike
bike2 <- bike
bike2 <- bike2 %>% 
  mutate(days_since_2011 = max(0, days_since_2011),
         cnt = cut(cnt, breaks = quantile(cnt), dig.lab = 10, include.lowest = T))
rule <- OneR(cnt ~ ., data = bike2)
rule
rule.to.table(rule) %>% 
  write.table("clipboard", sep = "\t", row.names = F)


## 2. Sequential Covering
#### covering-algo
##### The covering algorithm works by sequentially covering the feature space with single rules and removing the data points that are already covered by those rules.
##### For visualization purposes, the features x1 and x2 are continuous, but most rule learning algorithms require categorical features.
set.seed(42)
n <- 100
dat <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
dat$class <- rbinom(n=100, size=1, p = exp(dat$x1 + dat$x2)/(1+exp(dat$x1+dat$x2)))
dat$class <- factor(dat$class)

min.x1 <- min(dat$x1)
min.x2 <- min(dat$x2)

p1 <- dat %>% 
  ggplot() +
  geom_point(aes(x=x1, y=x2, color = class, shape = class)) +
  scale_color_viridis(guide = "none", discrete = T, option = "D", end = 0.9) +
  scale_shape_discrete(guide = "none") +
  ggtitle("Data")
p2 <- dat %>% 
  ggplot() +
  geom_rect(xmin = -3, xmax = 0, ymin = -2, ymax = -0.5, color = "black", fill = NA) +
  geom_point(aes(x = x1, y = x2, color = class, shape = class)) +
  scale_color_viridis(guide = "none", discrete = T, option = "D", end = 0.9) +
  scale_shape_discrete(guide = "none") +
  ggtitle("Step 1 : Find rule")

dat.reduced <- dat %>% 
  filter(!(x1 <=0 & x2 <= -0.5))

p3 <- dat.reduced %>% 
  ggplot() +
  geom_point(aes(x = x1, y = x2, color = class, shape = class)) +
  geom_rect(xmin = -3, xmax = 0, ymin = -2, ymax = -0.5, color = "black", fill = NA) +
  scale_x_continuous(limits = c(min.x1, NA)) +
  scale_y_continuous(limits = c(min.x2, NA)) +
  scale_color_viridis(guide = "none", discrete = T, option = "D", end = 0.9) +
  scale_shape_discrete(guide = "none") +
  ggtitle("Step 2 : Remove covered instances")

p4 <- p3 +
  geom_rect(xmin = 0.8, xmax = 2.5, ymin = -1.5, ymax = 1.5, color = 'black', fill = NA) +
  ggtitle("Step 3 : Find next rule")

gridExtra::grid.arrange(p1, p2, p3, p4, ncol = 2)  


#### learn-one-rule
##### Learning a rule by searching a path through a decision tree.
##### A decision tree is grown to predict the target of of interest.
##### We start at the root node, greedily and iteratively follow the path which locally produces the purest subset (e.g. highest accuracy) and add all the split values to the rule condition.
##### We end up with: If `location = good` and `size = big`, then `value = high`.


### Examples
#### jrip-cervical

extract.rules.jrip <- function(rule){
  rules <- scan(text = .jcall(rule$classifier, "S", "toString"), sep = "\n", what = "")
  # removes text
  rules <- rules[-c(1, 2, length(rules))]
  rules <- gsub("\\([0-9]*\\.[0-9]\\/[0-9]*\\.[0-9]\\)", "", rules)
  rules <- as.matrix(rules)[-c(1:2, 6), , drop = F]
  rules <- data.frame(rules)
  if(nrow(rules) == 0){
    return(NULL)
  } else {
    return(rules)
  }
}
rule <- JRip(Biopsy ~ ., data = cervical)
extract.rules.jrip(rule)

#### jrip-bike
bike2 <- bike %>% 
  mutate(cnt = round(cnt)) %>% 
  mutate(cnt = cut(cnt, breaks = quantile(cnt), dig.lab = 10, include.lowest = T),
         temp = round(temp),
         windspeed = round(windspeed), 
         hum = round(hum))

rule <- JRip(cnt ~ ., data = bike2)
tab <- extract.rules.jrip(rule)
tab %>% 
  write.table("clipboard", sep = "\t", row.names = F)
  

## 3. Bayesian Rule Lists
#### sbrl-cervical
cervical2 <- as.data.frame(lapply(cervical, function(x){
  if(is.factor(x) | n_distinct(x) < 3){
    as.factor(x)
  } else {
    arules::discretize(x, method = "interval", 3)
  }
}))

get.sbrl.rules <- function(x) {
  res = lapply(1:nrow(x$rs), function(i) {
    if (i == 1) 
      sprintf("If      %s (rule[%d]) then positive probability = %.8f\n", 
              x$rulenames[x$rs$V1[i]], x$rs$V1[i], x$rs$V2[i])
    else if (i == nrow(x$rs)) 
      sprintf("else  (default rule)  then positive probability = %.8f\n", 
              x$rs$V2[nrow(x$rs)])
    else sprintf("else if %s (rule[%d]) then positive probability = %.8f\n", 
                 x$rulenames[x$rs$V1[i]], x$rs$V1[i], x$rs$V2[i])
  })
  data.frame(rules = unlist(res))
}
rules$rulenames[rules$rs$V1[1]]



cervical2$label = cervical2$Biopsy
cervical2$Biopsy = NULL
rules  <-  sbrl(cervical2, pos_sign = "Cancer", neg_sign = "Healthy", rule_maxlen = 2)
rn = rules$rulenames
rl = get.sbrl.rules(rules)


#### sbrl-cervical-premined
set.seed(1)
conditions <- sample(rules$rulenames, size = 10)
conditions <- gsub("\\{|\\}", "", conditions)
conditions <- gsub(",", ", ", conditions)
conditions %>% 
  as.data.frame() %>% 
  rownames_to_column() %>% 
  write.table("clipboard", sep = "\t", row.names = F)


### Example 2: Rental bikes
#### sbrl-bike
bike2 %>% count(label)
table(bike$cnt > 4000)
bike2 <- bike %>% 
  mutate(label = cnt > 4000) %>% 
  mutate(cnt = NULL) %>% 
  lapply(function(x){
    if(is.factor(x) || n_distinct(x) < 3){
      as.factor(x)
    } else {
      arules::discretize(x, method = 'interval', 3)
    }
  }) %>% 
  as.data.frame()

rules <- sbrl(bike2, pos_sign = TRUE, neg_sign = FALSE, rule_maxlen = 3)

