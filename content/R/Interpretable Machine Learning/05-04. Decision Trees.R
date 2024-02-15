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
  library(partykit)
  library(rpart)
})

# 05-05. Decision Tree

#### tree-artificial
##### Decision tree with artificial data.
##### Instances with a value greater than 3 for feature x1 end up in node 5.
##### All other instances are assigned to node 3 or node 4, depending on whether values of feature x2 exceed 1.
set.seed(42)
n <- 100
dat_sim <- data.frame(feature_x1 = rep(c(3,3,4,4), times = n), 
                      feature_x2 = rep(c(1,2,2,2), times = n),
                      y = rep(c(1,2,3,4), times = n))
dat_sim <- dat_sim[sample(1:nrow(dat_sim), size = 0.9 * nrow(dat_sim)), ]
dat_sim$y <- dat_sim$y + rnorm(nrow(dat_sim), sd = 0.2)
ct <- ctree(y ~ feature_x1 + feature_x2, data = dat_sim)
ct %>% 
  plot(inner_panel = node_inner(ct, pval = F, id = F),
       terminal_panel = node_boxplot(ct, id = F))

### 2. Example
#### tree-example
##### Regression tree fitted on the bike rental data.
##### The maximum allowed depth for the tree was set to 2.
##### The trend feature (days since 2011) and the temperature (temp) have been selected for the splits.
##### The boxplots show the distribution of bicycle counts in the terminal node.
X <- bike %>% select(all_of(bike.features.of.interest))
y <- bike %>% pull(cnt)
dat <- cbind(X, y)

###### increases readability of tree
x <- rpart(y ~ ., data = na.omit(dat), 
           method = 'anova',
           control = rpart.control(cp = 0, maxdepth = 2))
xp <- as.party(x)
plot(xp, digits = 9, id = F, terminal_panel = node_boxplot(xp, id = F),
     inner_panel = node_inner(xp, id = F, pval = F))

#### tree-importance
##### Importance of the features measured by how much the node purity is imporved on average.
imp <- round(100 * x$variable.importance / sum(x$variable.importance), 0)
imp.df <- data.frame(feature = names(imp),
                     importance = imp)
imp.df <- imp.df %>% 
  mutate(feature = reorder(factor(feature), importance))
imp.df %>% 
  ggplot() + 
  geom_point(aes(x = importance, y = feature)) +
  scale_y_discrete("")
