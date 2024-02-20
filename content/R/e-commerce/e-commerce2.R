source("./e-commerce.R")

# 04. Build the recipe & workflow
## 04-01. Pre-processing data with recipes
# Assign libraries
suppressPackageStartupMessages({
  library(tidyverse)  # contains dplyr, ggplot and our data set
  library(vip)        # variable importance plots
  library(finetune)   # package for more advanced hyperparameter tuning  
  library(doParallel) # parallelisation package   
  library(tictoc)     # measure how long processes take
  library(tidymodels) # the main tidymodels packages all in one place - loaded last to overwrite any conflicts
  library(ggsci)
  library(patchwork)
  library(moments)
  library(GGally)
  library(corrplot)
  library(bonsai)
  library(workflowsets)
  library(hrbrthemes)
  library(viridis)
})
tidymodels_prefer()

df_mart %>% 
  ggplot(aes(x=Country, fill = factor(target))) +
  geom_bar(position = "dodge") +
  theme_minimal() +
  coord_flip() +
  labs(fill = "Target", x = "Country", y = "Count")

df_mart %>% 
  count(Country) %>% 
  arrange(desc(n)) %>% 
  mutate(ratio = n/sum(n))



# UK 이외의 기타 국가로 처리
df_mart <- df_mart %>% 
  mutate(Country = ifelse(Country=="United Kingdom", "UK", "ETC")) 

df_mart <- df_mart %>% 
  mutate(across(c(Country, peak_time, season), factor))

set.seed(123)

#### Create train-test set
split <- initial_split(df_mart, prop = 0.7, strata = target)
#### Create folds to for resampling on the Train set
train <- training(split)
test <- testing(split)
split

### Explore the data
skimr::skim(train %>% select(-c(bsym, CustomerID)))

categorical_vars <- train %>% 
  select(-c(bsym, CustomerID)) %>% 
  select_if(is.factor) %>% 
  names()
## Categorical variables
plot_categorical <- function(cat){
  cat <- ensym(cat)
  train %>%
    group_by(!!cat) %>%
    reframe(prop = mean(target, na.rm = T)) %>% 
    mutate(across(!!cat, ~reorder(.x, prop))) %>% 
    ggplot(aes(x = !!cat, y = prop, fill = !!cat)) +
    geom_col(color = 'black', alpha = 0.6) +
    geom_text(aes(label = scales::percent(prop, accuracy = 0.01)), vjust  = 1.5) +
    scale_color_nejm() +
    scale_fill_nejm() + 
    ggtitle(paste0("Target rate by level of ", as.character(cat))) +
    guides(fill = "none") +
    theme_light()
}
cat_p <- map(categorical_vars, plot_categorical)
cat_p[[1]] + cat_p[[2]] + cat_p[[3]]


#### 수치형 변수에 대해 왜도 계산
numeric_vars <- train %>% 
  select(-c(bsym, CustomerID)) %>% 
  select_if(is.numeric) %>% 
  select(-target) %>% 
  names()

num_skew <- map(numeric_vars, ~ skewness(train %>% pull(.x))) %>% 
  unlist()
names(num_skew) <- numeric_vars
num_skew

map_dfr(numeric_vars, ~summary(train %>% pull(.x))) %>% 
  mutate(var = numeric_vars, .before = Min.)

#### basic_rec 정의
basic_rec <- recipe(formula = target ~ ., data = train)
basic_rec <- basic_rec %>% 
  update_role(bsym, CustomerID, new_role = "ID")

after_trans <- basic_rec %>% 
  step_YeoJohnson(total_qty) %>% 
  prep(train) %>% 
  bake(train) %>% 
  ggplot(aes(x = total_qty)) + 
  geom_histogram(color = 'black') +
  ggtitle("After Transformation") +
  theme_light() 

before_trans <- function(num){
  num <- ensym(num)
  basic_rec %>% 
    prep(train) %>% 
    bake(train) %>% 
    ggplot(aes(x = !!num)) + 
    geom_histogram(color = 'black', fill = 'skyblue') +
    theme_light() +
    theme(axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())
}
before_trans_plot <- map(numeric_vars, before_trans)
before_trans_plot[[1]]+before_trans_plot[[2]]+before_trans_plot[[3]]+
  before_trans_plot[[4]]+before_trans_plot[[5]]+before_trans_plot[[6]]+
  before_trans_plot[[7]]+before_trans_plot[[8]]+before_trans_plot[[9]]+
  before_trans_plot[[10]]+before_trans_plot[[11]] +
  plot_layout(ncol = 3) +
  plot_annotation(title = "Before Transformation",
                  theme = theme(plot.title = element_text(hjust = 0.5))) 
  
after_trans <- function(num){
  num <- ensym(num)
  basic_rec %>% 
    step_YeoJohnson(!!num) %>% # Yeo-Johnson 변환
    prep(train) %>% 
    bake(train) %>% 
    ggplot(aes(x = !!num)) + 
    geom_histogram(color = 'black', fill = 'skyblue') +
    theme_light() +
    theme(axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank())
}
after_trans_plot <- map(numeric_vars, after_trans)
after_trans_plot[[1]]+after_trans_plot[[2]]+after_trans_plot[[3]]+
  after_trans_plot[[4]]+after_trans_plot[[5]]+after_trans_plot[[6]]+
  after_trans_plot[[7]]+after_trans_plot[[8]]+after_trans_plot[[9]]+
  after_trans_plot[[10]]+after_trans_plot[[11]] +
  plot_layout(ncol = 3) +
  plot_annotation(title = "After Yeo-Johnson Transformation",
                  theme = theme(plot.title = element_text(hjust = 0.5))) 


#### Yeo-Johnson 변환 후 target 변수와의 관계 확인 - boxplot with violin plot
plot_numerical <- function(num){
  num <- ensym(num)
  basic_rec %>% 
    step_YeoJohnson(all_numeric_predictors()) %>% 
    prep(train) %>% 
    bake(train) %>% 
    mutate(target = factor(target)) %>% 
    ggplot(aes(x = target, y = !!num, fill = target)) +
    geom_violin(width=1) +
    geom_boxplot(width = 0.3, alpha=0.7) +
    scale_fill_nejm() +
    coord_flip() + 
    theme_light() +
    guides(color = 'none') +
    theme(legend.position = "none")
}
num_p <- map(numeric_vars, plot_numerical)
num_p[[1]]+num_p[[2]]+num_p[[3]]+
  num_p[[4]]+num_p[[5]]+num_p[[6]]+
  num_p[[7]]+num_p[[8]]+num_p[[9]]+
  num_p[[10]]+num_p[[11]] + 
  plot_layout(ncol = 3)

#### 변수별 상관관계 - ANOVA 설명 분산량
mycor <- function(cnames, dat){
  x.num <- dat %>% pull(cnames[1])
  x.cat <- dat %>% pull(cnames[2])
  
  suppressWarnings({
    av <- anova(lm(x.num ~ x.cat))
  })
  sqrt(av$`Sum Sq`[1] / sum(av$`Sum Sq`))
}


cnames <- basic_rec %>% 
  step_YeoJohnson(all_numeric_predictors()) %>% 
  prep(train) %>% 
  bake(train) %>% 
  mutate(target = factor(target)) %>% 
  select_if(is.numeric) %>% names()
combs <- expand.grid(y = cnames, 
                     x = setdiff(names(train %>% select(-c(bsym, CustomerID))), "target"))
combs$cor <- apply(combs, 1, mycor, 
                  dat = basic_rec %>% 
                    step_YeoJohnson(all_numeric_predictors()) %>% 
                    prep(train) %>% 
                    bake(train) )
combs$lab <- sprintf("%.2f", combs$cor)
forder <- c(cnames, setdiff(unique(combs$x), cnames))

combs <- combs %>% 
  mutate(x = factor(x, levels = forder),
         y = factor(y, levels = rev(cnames)))

combs %>% 
  ggplot(aes(x = x, y = y, fill = cor, label = lab)) +
  geom_tile() +
  geom_label(fill = "white", size = 3) +
  theme_light() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  scale_x_discrete("") +
  scale_y_discrete("") +
  scale_fill_viridis("Variance\nexplained", begin = 0.2)

#### total_cnt 제거
basic_rec <- basic_rec %>% 
  step_rm(total_cnt)