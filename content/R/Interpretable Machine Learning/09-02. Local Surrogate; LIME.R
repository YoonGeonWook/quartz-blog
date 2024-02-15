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
})

bike <- bike %>% select(-atemp)
set.seed(42)

# 1. LIME for Tabular Data
## lime-fitting 
### LIME algorithm for tabular data.
### A) Random Forest predictions given features x1 and x2.
###    Predicted classes : 1 (dark) or 0 (light).
### B) Instance of interest (big dot) and data sampled from a normal distribution (small dots).
### C) Assign higher weight to points near the instance of interest.
### D) Signs of the grid show the classifications of the locally learned model from the weighted samples.
###    The white line marks the decision boundary (P(class=1) = 0.5).

#### Creating dataset
##### Define range of set
lower_x1 <- -2; upper_x1 <- 2
lower_x2 <- -2; upper_x2 <- 1

##### Size of the training set for the black box classifier
n_training <- 20000
##### Size for the grid to plot the decision bundaries 
n_grid <- 100
##### Number of samples for LIME explanations
n_sample <- 500

##### Simulate y ~ x1 + x2
set.seed(1)
x1 <- runif(n_training, min = lower_x1, max = upper_x1)
x2 <- runif(n_training, min = lower_x2, max = upper_x2)
y <- get_y(x1, x2)
##### Add noise
y_noisy <- get_y(x1, x2, noise_prob = 0.01)
lime_training_df <- data.frame(x1 = x1, x2 = x2, y = as.factor(y), y_noisy = as.factor(y_noisy))


##### For scaling later on
x_means <- c(mean(x1), mean(x2))
x_sd <- c(sd(x1), sd(x2))


##### Learn model
rf <- randomForest::randomForest(y_noisy ~ x1 + x2, data = lime_training_df, ntrees = 100)
lime_training_df$predicted <- predict(rf, newdata = lime_training_df)

#### The decision boundaries
grid_x1 <- seq(lower_x1, upper_x1, length.out = n_grid)
grid_x2 <- seq(lower_x2, upper_x2, length.out = n_grid)
grid_df <- expand.grid(x1 = grid_x1, x2 = grid_x2)
grid_df$predicted <- predict(rf, newdata = grid_df) %>% as.character() %>% as.numeric()

##### The observation to be explained
explain_x1 <- 1
explain_x2 <- -0.5
explain_y_model <- predict(rf, newdata = data.frame(x1 = explain_x1, x2 = explain_x2))
df_explain <- data.frame(x1 = explain_x1, x2 = explain_x2, y_predicted = explain_y_model)

point_explain <- c(explain_x1, explain_x2)
point_explain_scaled <- (point_explain - x_means) / x_sd


##### Drawing the samples for the LIME explanations
x1_sample <- rnorm(n_sample, x_means[1], x_sd[1])
x2_sample <- rnorm(n_sample, x_means[2], x_sd[2])
df_sample <- data.frame(x1 = x1_sample, x2 = x2_sample)
##### Scale the samples
points_sample <- apply(df_sample, 1, function(x){
  (x - x_means) / x_sd
}) %>% t()

##### Add weights to the samples
kernel_width <- sqrt(dim(df_sample)[2]) * 0.15
distances <- get_distances(point_explain = point_explain_scaled, points_sample = points_sample)

df_sample$weights <- kernel(d = distances, kernel_width = kernel_width)
df_sample$predicted <- predict(rf, newdata = df_sample)


#### Trees
# mod <- rpart(predicted ~ x1 + x2, data = df_sample, weights = df_sample$weights)
# grid_df$explained <- predict(mod, newdata = grid_df, type = 'prob')[,2]

#### Logistic regression model
mod <- glm(predicted ~ x1 + x2, data = df_sample, weights = df_sample$weights, family = 'binomial')
grid_df$explained <- predict(mod, newdata = grid_df, type = 'response')

#### Logistic decision boundary
coefs <- coefficients(mod)
logistic_boundary_x1 <- grid_x1
logistic_boundary_x2 <- - (1/coefs['x2']) * (coefs['(Intercept)'] + coefs['x1'] * grid_x1)
logistic_boundary_df <- data.frame(x1 = logistic_boundary_x1, x2 = logistic_boundary_x2)
logistic_boundary_df <- logistic_boundary_df %>% 
  filter(x2 <= upper_x2, x2 >= lower_x2)

##### Create a smaller grid for visualization of local model boundaries
x1_steps <- unique(grid_df$x1)[seq(1, n_grid, length.out = 20)]
x2_steps <- unique(grid_df$x2)[seq(1, n_grid, length.out = 20)]
grid_df_small <- grid_df %>% 
  filter(x1 %in% x1_steps & x2 %in% x2_steps)
grid_df_small$explained_class <- round(grid_df_small$explained)

colors = c('#132B43', '#56B1F7')

##### Data with some noise
p_data <- lime_training_df %>% 
  ggplot() +
  geom_point(aes(x=x1, y=x2, fill=y_noisy, color=y_noisy),
             alpha=0.3, shape=21) +
  scale_fill_manual(values=colors) +
  scale_color_manual(values = colors) +
  my_theme(legend.position = 'none')

##### The decision boundaries of the learned black box classifier
p_boundaries <- grid_df %>% 
  ggplot() +
  geom_raster(aes(x=x1, y=x2, fill=predicted), alpha = 0.3, interpolate = T) +
  my_theme(legend.position = 'none') +
  ggtitle('A')

#### Drawing some samples
p_samples <- p_boundaries +
  geom_point(data = df_sample,
             aes(x=x1, y=x2)) +
  scale_x_continuous(limits = c(lower_x1, upper_x1)) +
  scale_y_continuous(limits = c(lower_x2, upper_x2))

##### The point to be explained
p_explain <- p_samples +
  geom_point(data = df_explain,
             aes(x=x1, y=x2), fill='yellow', shape=21, size=4) +
  ggtitle('B')

p_weighted <- p_boundaries + 
  geom_point(data = df_sample,
             aes(x=x1, y=x2, size=weights)) +
  scale_x_continuous(limits = c(lower_x1, upper_x1)) +
  scale_y_continuous(limits = c(lower_x2, upper_x2)) +
  geom_point(data = df_explain,
             aes(x=x1, y=x2), fill='yellow', shape=21, size=4) +
  ggtitle('C')

p_boundaries_lime <- grid_df %>% 
  ggplot() +
  geom_raster(aes(x=x1, y=x2, fill=predicted), alpha=0.3, interpolate=T) +
  geom_point(data = grid_df_small %>% filter(explained_class==1),
             aes(x=x1, y=x2, color=explained),
             size=2, shape=3) +
  geom_point(data = grid_df_small %>% filter(explained_class==0),
             aes(x=x1, y=x2, color=explained),
             size=2, shape=95) +
  geom_point(data = df_explain,
             aes(x=x1, y=x2),
             fill='yellow', shape=21, size=4) +
  geom_line(data = logistic_boundary_df,
            aes(x=x1, y=x2), color = 'white') +
  my_theme(legend.position = 'none') +
  ggtitle('D')

gridExtra::grid.arrange(p_boundaries, p_explain, p_weighted, p_boundaries_lime, ncol = 2)


## lime-fail
### Explanation of the prediction of instance x = 1.6.
### The predictions of the black box model depending on a single feature is shown as a thick line and the distribution of the data is shown with rugs.
### Three local surrogate models with different kernel widths are computed.
### The resulting linear regression model depends on the kernel width: 
###  Does the feature have a negative, positive or no effect for x=1.6?
set.seed(42)
df <- data.frame(x = rnorm(200, mean = 0, sd = 3))
df$x[df$x < -5] <- -5
df$y <- (df$x + 2)^2
df$y[df$x > 1] <- -df$x[df$x > 1] + 10 + -0.05 * df$x[df$x > 1]^2
# df$y <- df$y + rnorm(nrow(df), sd=0.05)
explain.p <- data.frame(x=1.6, y=8.5)


w1 <- kernel(d = get_distances(data.frame(x=explain.p$x), df), kernel_width = 0.1)
w2 <- kernel(d = get_distances(data.frame(x=explain.p$x), df), kernel_width = 0.75)
w3 <- kernel(d = get_distances(data.frame(x=explain.p$x), df), kernel_width = 2)


lm.1 <- lm(y ~ x, data = df, weights = w1)
lm.2 <- lm(y ~ x, data = df, weights = w2)
lm.3 <- lm(y ~ x, data = df, weights = w3)

df.all <- rbind(df, df, df)
df.all$lime <- c(predict(lm.1), predict(lm.2), predict(lm.3))
df.all$width <- factor(c(rep(c(0.1, 0.75, 2), each = nrow(df))))


df.all %>% 
  ggplot(aes(x=x, y=y)) +
  geom_line(lwd = 2.5) +
  geom_rug(sides = 'b') +
  geom_line(aes(x = x, y = lime, group = width, color = width, linetype = width)) +
  geom_point(data = explain.p,
             aes(x=x, y=y), 
             size=12, shape='x') +
  scale_color_viridis('Kernel width', discrete = T) +
  scale_linetype('Kernel width') +
  scale_y_continuous('Black Box Prediction')


## lime-tabular-train-black-box
ntree <- 100
trend_line <- lm(cnt ~ days_since_2011, data = bike)
bike.train.resid <- factor(resid(trend_line) > 0, levels = c(F, T), labels = c('below', 'above'))
bike.train.x <- bike %>% select(-cnt)
model <- caret::train(x = bike.train.x, bike.train.resid,
                      method = 'rf', ntree = ntree, maximise = F)
n_features_lime <- 2

## lime-tabular-example-explain-plot-1
### LIME explanations for two instances of the bike rental dataset.
### Warmer temperature and good weather situation have a positive effect on the prediction.
### The x-axis shows the feature effect : 
###  The weight times the actual feature value.
instance_indices <- c(295, 8)
set.seed(44)
bike.train.x$temp <- round(bike.train.x$temp,2)
pred <- Predictor$new(model = model, data = bike.train.x, class = 'above', type = 'prob')
lim1 <- LocalModel$new(predictor = pred, x.interest = bike.train.x %>% slice(instance_indices[1]), k = n_features_lime)
lim2 <- LocalModel$new(predictor = pred, x.interest = bike.train.x %>% slice(instance_indices[2]), k = n_features_lime)
wlim <- c(min(c(lim1$results$effect, lim2$results$effect)), max(c(lim1$results$effect, lim2$results$effect)))
a <- lim1$plot() +
  scale_y_continuous(limit = wlim) +
  geom_hline(aes(yintercept = 0)) +
  theme(axis.title.y = element_blank(),
        axis.ticks.y = element_blank())
b <- lim2$plot() +
  scale_y_continuous(limits = wlim) +
  geom_hline(aes(yintercept = 0)) +
  theme(axis.title.y = element_blank(),
        axis.ticks.y = element_blank())
grid.arrange(a,b,ncol=1)

# 2. LIME for Text
## load-text-classification-lime
ycomments %>% dim()
example_indices <- c(267, 173)
texts <- ycomments$CONTENT[example_indices]

## lime-text-variations
labeledTerms <- prepare_data(ycomments$CONTENT)
labeledTerms$class <- factor(ycomments$CLASS, levels = c(0,1), labels = c("no spam", "spam"))
labeledTerms2 <- prepare_data(ycomments, trained_corpus = labeledTerms)

rp <- rpart::rpart(class ~., data = labeledTerms)
predict_fun <- get_predict_fun(model = rp, train_corpus = labeledTerms)
tokenized
tokenized <- tokenize(texts[2])

set.seed(2)
variations <- create_variations(text = texts[2], pred_fun = predict_fun, prob = 0.7, n_variations = 5, class = 'spam')
colnames(variations) <- c(tokenized, "prob", 'weights')
example_sentence <- paste(colnames(variations)[variations[2, ] == 1], collapse =' ')


## lime-text-explanations
set.seed(42)
ycomments.predict <- get.ycomment.classifier(ycomments)
explanations <- data.table::rbindlist(lapply(seq_along(texts), function(i){
  explain_text(text = texts[i], pred_fun = ycomments.predict, class = 'spam', case = i, prob = 0.5)
}))
explanations <- data.frame(explanations)
explanations %>% 
  select(case, label_prob, feature, feature_weight) %>% 
  mutate(label_prob = round(label_prob, 7),
         feature_weight = round(feature_weight, 6)) %>% 
  write.table("clipboard", sep = "\t", row.names = F)
