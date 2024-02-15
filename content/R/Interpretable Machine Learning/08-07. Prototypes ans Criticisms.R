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

bike <- bike %>% select(-atemp)
set.seed(42)

### Prototypes and criticisms for a data distribution with two features x1 and x2.
set.seed(1)
dat1 <- data.frame(x1 = rnorm(20, mean = 4, sd = 0.3), x2 = rnorm(20, mean = 1, sd = 0.3))
dat2 <- data.frame(x1 = rnorm(30, mean = 2, sd = 0.3), x2 = rnorm(30, mean = 2, sd = 0.2))
dat3 <- data.frame(x1 = rnorm(40, mean = 3, sd = 0.3), x2 = rnorm(40, mean = 3))
dat4 <- data.frame(x1 = rnorm(7, mean = 4, sd = 0.1), x2 = rnorm(7, mean = 2.5, 0.1))

dat <- rbind(dat1, dat2, dat3, dat4)
dat$type <- 'data'
dat$type[c(7, 23, 77)] <- 'prototype'
dat$type[c(81, 95)] <- 'criticism'


dat %>% 
  ggplot(aes(x = x1, y = x2)) +
  geom_point(alpha = 0.7) +
  geom_point(data = dat %>% filter(type != 'data'),
             aes(shape = type),
             size = 9, alpha = 1, color = 'blue') +
  scale_shape_manual(breaks = c('prototype', 'criticism'), values = c(18, 19))

# 1. Theory
## mmd
### The squred maximum mean discrepancy measure (MMD2) for a dataset with two features and different selections of prototypes.
set.seed(42)
n <- 40
#### create dataset from three gaussian in 2d
dt1 = data.frame(x1 = rnorm(n, mean = 1, sd = 0.1), x2 = rnorm(n, mean = 1, sd = 0.3))
dt2 = data.frame(x1 = rnorm(n, mean = 4, sd = 0.3), x2 = rnorm(n, mean = 2, sd = 0.3))
dt3 = data.frame(x1 = rnorm(n, mean = 3, sd = 0.5), x2 = rnorm(n, mean = 3, sd = 0.3))
dt4 = data.frame(x1 = rnorm(n, mean = 2.6, sd = 0.1), x2 = rnorm(n, mean = 1.7, sd = 0.1))
dt = rbind(dt1, dt2, dt3, dt4)

radial <- function(x1, x2, sigma = 1){
  dist <- sum((x1-x2)^2)
  exp(-dist/(2*sigma^2))
}

cross.kernel <- function(d1, d2){
  kk <- c()
  for(i in nrow(d1)){
    for(j in nrow(d2)){
      res <- radial(d1[i, ], d2[j, ])
      kk <- c(kk, res)
    }
  }
  mean(kk)
}

mmd2 <- function(d1, d2){
  cross.kernel(d1, d1) - 2 * cross.kernel(d1, d2) + cross.kernel(d2, d2)
}

#### create 3 variants of prototypes
pt1 <- rbind(dt1[c(1,2), ], dt4[1, ])
pt2 <- rbind(dt1[1, ], dt2[3, ], dt3[19, ])
pt3 <- rbind(dt2[3, ], dt3[19, ])

#### create plot with all data and density estimation
p <- dt %>% 
  ggplot(aes(x = x1, y = x2)) +
  stat_density_2d(geom = 'tile', 
                  aes(fill = after_stat(density)),
                  contour = F, alpha = 0.9) +
  geom_point() +
  scale_fill_gradient2(low = 'white', high = 'blue', guide = 'none') +
  scale_x_continuous(limits = c(0, NA)) +
  scale_y_continuous(limits = c(0, NA))

#### create plot for each prototype
p1 <- p + geom_point(data = pt1, color = 'red', size = 4) +
  geom_density_2d(data = pt1, color = 'red') +
  ggtitle(sprintf("%.3f MMD2", mmd2(dt, pt1)))
p2 <- p + geom_point(data = pt2, color = 'red', size = 4) +
  geom_density_2d(data = pt2, color = 'red') +
  ggtitle(sprintf("%.3f MMD2", mmd2(dt, pt2)))
p3 <- p + geom_point(data = pt3, color = 'red', size = 4) +
  geom_density_2d(data = pt3, color = 'red') +
  ggtitle(sprintf("%.3f MMD2", mmd2(dt, pt3)))

#### TODO : Add custom legend for prototypes

#### overlay mmd measure for each plot
gridExtra::grid.arrange(p, p1, p2, p3, ncol = 2)

## witness
### Evaluations of the witness function at different pts.
witness <- function(x, dist1, dist2, sigma = 1){
  k1 <- apply(dist1, 1, function(z) radial(x, z, sigma = sigma))
  k2 <- apply(dist2, 1, function(z) radial(x, z, sigma = sigma))
  mean(k1) - mean(k2)
}

w.points.indices <- c(125, 2, 60, 19, 100)
wit.points <- dt %>% slice(w.points.indices)
wit.points$witness <- apply(wit.points, 1, function(x) round(witness(x[c("x1", "x2")], dt, pt2, sigma = 1), 3))

p +
  geom_point(data = pt2, color = 'red') +
  geom_density_2d(data = pt2, color = 'red') +
  ggtitle(sprintf("%.3f MMD2", mmd2(dt, pt2))) +
  geom_label(data = wit.points, 
             aes(label = witness),
             alpha = 0.9, vjust = 'top') +
  geom_point(data = wit.points, color = 'black', shape = 17, size = 4)
