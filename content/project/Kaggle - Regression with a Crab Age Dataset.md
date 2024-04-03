---
tags:
  - Kaggle
  - Crab_Data
  - mlr3
  - GBM
  - XGBoost
  - LightGBM
  - Catboost
  - LAD_ensemble
  - Stacking
date: 2024-03-25
---

![[Pasted image 20240403211039.png]]

# Dataset

이 대회의 데이터셋은 [Crab Age Prediction](https://www.kaggle.com/datasets/sidhus/crab-age-prediction)에서 학습된 딥러닝 모델에서 생성되었다. Feature distributions는 원본과 비슷하지만 완전히 동일하지는 않다. 


#### Evaluation

제출 데이터는 평균 절대 오차(MAE)를 사용해 평가된다:

$$
MAE = \frac{1}{n}\sum_{i=1}^n|x_i-y_i|
$$
여기서 `x_i`는 predicted target을, `y_i`은 ground truth을, `n`은 test set의 행의 수를 나타낸다. 


# Description

- `Sex`: 게의 생물학적 성별 - Male(`M`), Female(`F`), Indeterminate(`I`)
- `Length`: 게 머리의 가장 앞쪽 지점에서 등딱지 뒤쪽까지 측정한 길이 (단위: 피트)
	- 크기와 성장을 나태내는 지표로, 나이 및 체중과 같은 다른 변수와 상관관계가 있음
- `Diameter`: 게의 몸에서 가장 넓은 부분을 가로지르는 지름 (단위: 피트)
	- `Length`, `Height`와 함께 게의 전체적인 크기와 모양을 결정
- `Height`: 게의 세로 길이를 측정한 높이 (단위: 피트)
	- 게의 전체적인 몸 모양과 크기를 알 수 있는 지표
- `Weight`: 게의 전체 무게 (단위: 온스)
	- 무게는 게의 건강, 나이, 성장 주기의 단계를 나타내는 지표
- `Shucked Weight`: 껍데기를 제거한 무게 (단위: 온스)
- `Viscera Weight`: 내장의 무게 (단위: 온스)
- `Shell Weight`: 껍질 무게 (단위: 온스)
	- 껍질의 무게는 게의 성장, 나이 및 전반적인 건강에 대한 주요 단서
- `Age`: 게의 나이 (단위: 개월 수)

# Package Load

```r
# 현재 script 저장되어 있는 경로 지정
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

# Introduce the requred packages
library(tidyverse)
library(mlr3verse)
library(data.table)
library(mlr3mbo)
library(mlr3tuningspaces)
library(mlr3tuning)
library(bbotk)
library(ggplot2)
library(ggcorrplot)
library(patchwork)
library(skimr)
library(quantreg)
library(checkmate)
library(tictoc)
set.seed(123)

# Reduce annoying logging
lgr::get_logger("mlr3")$set_threshold("warn")
lgr::get_logger("bbotk")$set_threshold("warn")
```

# Reading Data Files

```r
# Load data
train = fread("../../data/playground-series-s3e16/train.csv")
test = fread("../../data/playground-series-s3e16/test.csv")
original = fread("../../data/playground-series-s3e16/CrabAgePrediction.csv")
submission = fread("../../data/playground-series-s3e16/sample_submission.csv")

sprintf("Dimension of the train synthetic dataset: (%s, %s)", nrow(train), ncol(train))
sprintf("Dimension of the test synthetic dataset: (%s, %s)", nrow(test), ncol(test))
sprintf("Dimension of the original dataset: (%s, %s)", nrow(original), ncol(original))
sprintf("Dimension of the submission dataset: (%s, %s)", nrow(submission), ncol(submission))

## [1] "Dimension of the train synthetic dataset: (74051, 10)"
## [1] "Dimension of the test synthetic dataset: (49368, 9)"
## [1] "Dimension of the original dataset: (3893, 9)"
## [1] "Dimension of the submission dataset: (49368, 2)"
```

> [!note]- code fold
> ```r
> #' Custom {skimr} summary stats
> #'
> #' `su` stands for _summary_. Returns different summary stats for different classes, distributions (unicode) for numeric class, raw data of first and last parts, etc.
> #'
> #' @details `su` and `su2` are the same except that `su2` provides extra _missingness_ functions `skimr::n_missing` and `skimr::complete_rate`.
> #'
> #' @param n_bins Number of histogram bars
> #' @param digits Integer indicating the number of decimal places (round) or significant digits (signif) to be used. Negative values are allowed.
> #' @param n_head_tail Number of `head` and `tail` of raw data.
> #' @param base `skimr::sfl` that sets skimmers for all column types.
> #' @param ... (none)
> #'
> #' @name custom_summary
> 
> #' @rdname custom_summary
> #' @export
> my_skim <- function(..., n_bins = 10, digits = 2, n_head_tail = 3, base = skimr::sfl()) {
>   
>   func <- skimr::skim_with(append = FALSE, base = base,
>                            numeric = skimr::sfl(
>                              mean  = function(x) round(mean(x, na.rm = TRUE), digits),
>                              sd    = function(x) round(sd(x,   na.rm = TRUE), digits),
>                              med   = function(x) round(median(x, na.rm = TRUE), digits),
>                              min   = function(x) round(min(x, na.rm = TRUE), digits),
>                              `5%`  = function(x) quantile(x, .05, na.rm = TRUE, names = FALSE),
>                              `95%` = function(x) quantile(x, .95, na.rm = TRUE, names = FALSE),
>                              max   = function(x) round(max(x, na.rm = TRUE), digits),
>                              dist  = function(x) skimr::inline_hist(x , n_bins = n_bins),
>                              head_tail = function(x)
>                                paste(c(round(head(x, n_head_tail), digits), round(tail(x, n_head_tail), digits)), collapse = ' ')
>                            ),
>                            factor    = skim_default_headtail('factor',    n_head_tail),
>                            character = skim_default_headtail('character', n_head_tail),
>                            Date      = skim_default_headtail('Date',      n_head_tail),
>                            list      = skim_default_headtail('list',      n_head_tail),
>                            logical   = skim_default_headtail('logical',   n_head_tail),
>                            AsIs      = skim_default_headtail('AsIs',      n_head_tail),
>                            complex   = skim_default_headtail('complex',   n_head_tail),
>                            difftime  = skim_default_headtail('difftime',  n_head_tail),
>                            POSIXct   = skim_default_headtail('POSIXct',   n_head_tail),
>                            ts        = skim_default_headtail('ts',        n_head_tail),
>   )
>   func(...)
> }
> 
> #' @rdname custom_summary
> #' @export
> my_skim2 <- function(..., n_bins = 10, digits = 2, n_head_tail = 3,
>                 base = skimr::sfl(missing     = skimr:::n_missing)) {
>   my_skim(..., n_bins = n_bins, digits = digits, n_head_tail = n_head_tail, base=base)
> }
> 
> # Internal func
> skim_default_headtail <- function(skim_type, n_head_tail) {
>   skim_default  <- skimr::get_default_skimmers(skim_type)
>   skim_headtail <- skimr::sfl(head_tail = function(x) {
>     # Workaround issues of `c` combining factors makes integers
>     if (skim_type == 'factor') x <- as.character(x)
>     paste(c(head(x, n_head_tail), tail(x, n_head_tail)), collapse = ' ')
>   })
>   
>   do.call(skimr::sfl, c(skim_default[[1]], skim_headtail$funs))
> }
> 
> 
> ```

```r
my_skim2(train)
## ── Data Summary ────────────────────────
##                            Values
## Name                       train 
## Number of rows             74051 
## Number of columns          10    
## Key                        NULL  
## _______________________          
## Column type frequency:           
##   character                1     
##   numeric                  9     
## ________________________         
## Group variables            None  
## 
## ── Variable type: character ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
##   skim_variable missing min max empty n_unique whitespace head_tail  
## 1 Sex                 0   1   1     0        3          0 I I M F I I
## 
## ── Variable type: numeric ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
##   skim_variable  missing     mean       sd      med  min     `5%`     `95%`      max dist       head_tail                         
## 1 id                   0 37025    21377.   37025    0    3702.    70348.    74050    ▇▇▇▇▇▇▇▇▇▇ 0 1 2 74048 74049 74050           
## 2 Length               0     1.32     0.29     1.38 0.19    0.738     1.68      2.01 ▁▁▁▂▃▅▇▇▂▁ 1.52 1.1 1.39 1.49 1.21 0.91      
## 3 Diameter             0     1.02     0.24     1.07 0.14    0.55      1.31      1.61 ▁▁▁▂▃▅▇▇▂▁ 1.18 0.82 1.11 1.2 0.96 0.68      
## 4 Height               0     0.35     0.09     0.36 0       0.188     0.475     2.83 ▂▇▁▁▁▁▁▁▁▁ 0.38 0.28 0.38 0.41 0.31 0.2      
## 5 Weight               0    23.4     12.6     23.8  0.06    3.60     44.2      80.1  ▅▆▆▇▆▂▁▁▁▁ 28.97 10.42 24.78 29.48 16.77 5.39
## 6 Shucked Weight       0    10.1      5.62     9.91 0.03    1.50     19.4      42.2  ▆▇▇▇▃▁▁▁▁▁ 12.73 4.52 11.34 12.3 8.97 2.06   
## 7 Viscera Weight       0     5.06     2.79     4.99 0.04    0.794     9.74     21.6  ▆▇▇▆▂▁▁▁▁▁ 6.65 2.32 5.56 7.54 2.92 1.03     
## 8 Shell Weight         0     6.72     3.58     6.93 0.04    1.11     12.7      28.5  ▅▆▇▆▂▁▁▁▁▁ 8.35 3.4 6.66 8.08 4.28 1.7       
## 9 Age                  0     9.97     3.18    10    1       6        16        29    ▁▂▇▇▂▁▁▁▁▁ 9 8 9 10 8 6
```

```r
my_skim2(test)
## ── Data Summary ────────────────────────
##                            Values
## Name                       test  
## Number of rows             49368 
## Number of columns          9     
## Key                        NULL  
## _______________________          
## Column type frequency:           
##   character                1     
##   numeric                  8     
## ________________________         
## Group variables            None  
## 
## ── Variable type: character ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
##   skim_variable missing min max empty n_unique whitespace head_tail  
## 1 Sex                 0   1   1     0        3          0 I I F F F M
## 
## ── Variable type: numeric ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
##   skim_variable  missing     mean       sd      med      min      `5%`      `95%`       max dist       head_tail                             
## 1 id                   0 98734.   14251.   98734.   74051    76519.    120950.    123418    ▇▇▇▇▇▇▇▇▇▇ 74051 74052 74053 123416 123417 123418
## 2 Length               0     1.32     0.29     1.39     0.19     0.738      1.68       2.04 ▁▁▁▂▃▅▇▇▂▁ 1.05 1.16 1.29 1.49 1.24 1.66         
## 3 Diameter             0     1.03     0.24     1.07     0.14     0.562      1.31       1.62 ▁▁▁▂▃▆▇▇▁▁ 0.76 0.89 0.99 1.16 0.95 1.3          
## 4 Height               0     0.35     0.09     0.36     0        0.188      0.475      2.83 ▂▇▁▁▁▁▁▁▁▁ 0.28 0.28 0.32 0.36 0.29 0.44         
## 5 Weight               0    23.5     12.6     23.8      0.06     3.64      44.2       80.1  ▅▆▇▇▆▂▁▁▁▁ 8.62 15.51 14.57 31.38 15.66 36.61    
## 6 Shucked Weight       0    10.1      5.61     9.98     0.03     1.55      19.3       42.2  ▆▇▇▇▃▁▁▁▁▁ 3.66 7.03 5.56 11.4 6.1 14.91         
## 7 Viscera Weight       0     5.07     2.79     4.99     0.01     0.808      9.72      21.6  ▆▆▇▆▂▁▁▁▁▁ 1.73 3.25 3.88 6.85 3.73 8.29         
## 8 Shell Weight         0     6.75     3.58     6.95     0.04     1.13      12.6       28.5  ▅▆▇▆▂▁▁▁▁▁ 2.72 3.97 4.82 8.79 4.96 10.49
```

```r
my_skim2(original)
## ── Data Summary ────────────────────────
##                            Values  
## Name                       original
## Number of rows             3893    
## Number of columns          9       
## Key                        NULL    
## _______________________            
## Column type frequency:             
##   character                1       
##   numeric                  8       
## ________________________           
## Group variables            None    
## 
## ── Variable type: character ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
##   skim_variable missing min max empty n_unique whitespace head_tail  
## 1 Sex                 0   1   1     0        3          0 F M I I I I
## 
## ── Variable type: numeric ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
##   skim_variable  missing  mean    sd   med  min  `5%` `95%`   max dist       head_tail                     
## 1 Length               0  1.31  0.3   1.36 0.19 0.738  1.72  2.04 ▁▁▁▂▃▆▇▇▂▁ 1.44 0.89 1.04 0.62 1.06 0.79 
## 2 Diameter             0  1.02  0.25  1.06 0.14 0.55   1.36  1.62 ▁▁▂▃▃▆▇▇▂▁ 1.18 0.65 0.78 0.46 0.78 0.61 
## 3 Height               0  0.35  0.1   0.36 0    0.188  0.5   2.83 ▂▇▁▁▁▁▁▁▁▁ 0.41 0.21 0.25 0.16 0.26 0.21 
## 4 Weight               0 23.6  13.9  22.8  0.06 3.57  48.1  80.1  ▆▇▇▇▆▃▁▁▁▁ 24.64 5.4 7.95 2.01 10.35 4.07
## 5 Shucked Weight       0 10.2   6.28  9.54 0.03 1.47  21.0  42.2  ▆▇▇▆▃▁▁▁▁▁ 12.33 2.3 3.23 0.77 4.51 1.5  
## 6 Viscera Weight       0  5.14  3.1   4.86 0.01 0.774 10.7  21.6  ▆▇▇▆▃▁▁▁▁▁ 5.58 1.37 1.6 0.52 2.34 1.35  
## 7 Shell Weight         0  6.8   3.94  6.66 0.04 1.09  13.6  28.5  ▆▇▇▆▂▁▁▁▁▁ 6.75 1.56 2.76 0.64 2.98 1.42 
## 8 Age                  0  9.95  3.22 10    1    6     16    29    ▁▂▇▇▂▁▁▁▁▁ 9 6 6 5 6 8
```

 > [!info] <b>💡</b> 대회 데이터, 원본 데이터 모두에 결측치는 없다. 또한 train/test set에 대한 요약 통계량을 보면 비슷한 분포를 갖고 있음을 알 수 있다.

# EDA

먼저 관심 있는 변수 `Age`에 대해 시각화 해보자.

> [!note]- code fold
> ```r
> p1 = ggplot(train, aes(x = Age)) +
>   geom_density(fill = "steelblue",  alpha = 0.4) +
>   scale_x_continuous(breaks = seq(0, 30, 5)) +
>   labs(title = "Competition Dataset")
> p2 = ggplot(original, aes(x = Age)) +
>   geom_density(fill = "orange",  alpha = 0.4) +
>   scale_x_continuous(breaks = seq(0, 30, 5)) +
>   labs(title = "Original Dataset")
> (p1 + p2) &
>   theme_minimal() &
>   plot_layout(nrow = 1)
> ```

![[Pasted image 20240326191111.png]]

Target `Age`에 대한 분포를 간단히 살펴본 결과, competition의 데이터와 원본 데이터가 거의 동일함을 알 수 있다. 이제 `Age`가 어떤 피쳐들과 연관되어 있는지 살펴보자. 

> [!note]- code fold
> ```r
> p1 = ggcorrplot(cor(train %>% select(-c(id, Sex))), type = "lower", lab = T, colors = c("yellow", "orange", "red"), lab_col = "white", tl.srt = 45)
> p2 = ggcorrplot(cor(original %>% select(-c(Sex))), type = "lower", lab = T, colors = c("yellow", "orange", "red"), lab_col = "white", tl.srt = 45)
> gridExtra::grid.arrange(p1, p2, nrow = 1)
> ```

![[Pasted image 20240326193150.png]]

수치형 피쳐들과 target `Age` 간 상관관계는 competition과 원본 데이터셋에서 거의 유사하다. `Age`와 가장 높은 상관관계를 갖는 것은 껍질 무게 `Shell Weight`이고, 가장 낮은 상관관계를 갖는 것은 껍질 제거 후 무게 `Shucked Weight`이다. 

이제 행별로 중복을 체크 해보자. 

```r
cat(
  " There are", nrow(train), "observations in the train competition dataset\n", 
  "There are", train %>% select(-id) %>% n_distinct(), "unique observations in the train competition dataset\n",
  "There are", train %>% select(-c(id, Age)) %>% n_distinct(), "unique observations (only features) in the train competition dataset"
)
##  There are 74051 observations in the train competition dataset
##  There are 74051 unique observations in the train competition dataset
##  There are 74051 unique observations (only features) in the train competition dataset

cat(
  " There are", nrow(test), "observations in the test competition dataset\n", 
  "There are", test %>% select(-id) %>% n_distinct(), "unique observations in the test competition dataset"
)
##  There are 49368 observations in the test competition dataset
##  There are 49368 unique observations in the test competition dataset

cat(
  " There are", nrow(original), "observations in the original dataset\n", 
  "There are", original %>% n_distinct(), "unique observations in the original dataset"
)
##  There are 3893 observations in the original dataset
##  There are 3893 unique observations in the original dataset
```

주어진 데이터셋에는 중복된 행이 없다는 것을 확인했다. 다음으로 성별 `Sex`와 나이 `Age`의 관계를 살펴보자.


> [!note]- code fold
> ```r
> # Relationship b/w `Sex` and `Age`
> p1 = ggplot(train %>% mutate(Sex = factor(Sex, levels = c("I", "M", "F"))), aes(x = Sex, y = Age, fill = Sex)) + 
>   geom_boxplot(width = 0.5) +
>   labs(title = "Competition Dataset")
> p2 = ggplot(original %>% mutate(Sex = factor(Sex, levels = c("I", "M", "F"))), aes(x = Sex, y = Age, fill = Sex)) + 
>   geom_boxplot(width = 0.5) +
>   labs(title = "Original Dataset")
> (p1 + p2) *
>   scale_fill_viridis_d(end = 0.8) *
>   scale_y_continuous(breaks = seq(0, 30, 5)) *
>   theme_minimal() *
>   plot_layout(nrow = 1)
> ```

![[Pasted image 20240326200832.png]]

> [!note]- code fold
> ```r
> # Relationship b/w `Shell Weight` and `Age`
> p1 = ggplot(train, aes(x = `Shell Weight`, y = Age)) +
>   geom_point(shape = 21, color = 'white', fill = "steelblue", size = 2) +
>   labs(title = "Competition Dataset")
> p2 = ggplot(original, aes(x = `Shell Weight`, y = Age)) +
>   geom_point(shape = 21, color = 'white', fill = "orange", size = 2) +
>   labs(title = "Original Dataset")
> (p1 + p2) *
>   scale_y_continuous(breaks = seq(0, 30, 5)) *
>   theme_minimal() *
>   plot_layout(nrow = 1)
> ```

![[Pasted image 20240326201356.png]]


> [!note]- code fold
> ```r
> # Relationship b/w `Diameter` and `Age`
> p1 = ggplot(train, aes(x = Diameter, y = Age)) +
>   geom_point(shape = 21, color = 'white', fill = "steelblue", size = 2) +
>   labs(title = "Competition Dataset")
> p2 = ggplot(original, aes(x = Diameter, y = Age)) +
>   geom_point(shape = 21, color = 'white', fill = "orange", size = 2) +
>   labs(title = "Original Dataset")
> (p1 + p2) *
>   scale_y_continuous(breaks = seq(0, 30, 5)) *
>   theme_minimal() *
>   plot_layout(nrow = 1)
> ```

![[Pasted image 20240326201713.png]]

위 플롯들을 보면, `Sex` \& `Age`의 관계, `Shell Weight` & `Age`의 관계, `Diameter` & `Age`의 관계는 원본 데이터와 비슷함을 알 수 있다. 

# Modeling

먼저 원본 데이터를 학습 데이터에 병합하자. 

```r
train = train %>% 
  mutate(generated = 1)
test = test %>% 
  mutate(generated = 1)
original = original %>% 
  mutate(generated = 0)
train = train %>% select(-id) %>% 
  rbind(original)

colnames(train) = c(paste0("V", 1:8), "Age", "generated")
colnames(test_baseline) = c(paste0("V", 1:8), "generated")
```

## Gradient Boosting Machine tuning

- Search Space:

| Hyperparameter      | Type  | Description                                    | Range                                                                                                               |
|:------------------- |:----- |:---------------------------------------------- |:------------------------------------------------------------------------------------------------------------------- |
| `distribution`      | `chr` | 반응변수의 분포 지정(손실 함수)                | `"gaussian"`(squared error), `"laplace"`(absolute lose)<br>`"tdist"`(t-dist loss), `"bernoulli"`(binary outcome) 등 |
| `shrinkage`         | `dbl` | 각 트리에 적용되는 학습률 지정                 | $[0, 1]$                                                                                                            |
| `n.trees`           | `int` | 총 트리 수 지정                                | $[1, \infty)$                                                                                                       |
| `n.minobsinnode`    | `int` | 터미널 노드에 있어야 하는 최솟 관측치 수 지정  | $[1, \infty)$                                                                                                       |
| `interaction.depth` | `int` | 트리의 최대 깊이 지정                          | $[1, \infty)$                                                                                                       |
| `bag.fraction`      | `dbl` | 각 반복에서 학습에 사용되는 데이터의 비율 지정 | $[0,1]$                                                                                                             |


```r
# Set the task
task = as_task_regr(train, target = "Age")

lrn_gbm = lrn("regr.gbm", distribution = 'laplace', n.trees = 1000)

## GBM tuning
measures = msr("regr.mae")
search_space = ps(
  bag.fraction = p_dbl(lower = 0, upper = 1),
  shrinkage = p_dbl(lower = 1e-04, upper = 1, logscale = T),
  n.minobsinnode = p_int(lower = 1, upper = 100),
  interaction.depth = p_int(lower = 1, upper = 10)
)
```

- 튜닝 알고리즘: Bayesian Optimization with Holdout resampling

```r
tuner_bo = tnr("mbo")
future::plan("multisession", workers = 10)
instance = tune(
  tuner = tuner_bo,
  task = task,
  learner = lrn_gbm,
  resampling = rsmp("holdout"),
  measures = measures,
  search_space = search_space,
  term_evals = 25
)
instance_gbm$result
##    bag.fraction shrinkage n.minobsinnode interaction.depth learner_param_vals  x_domain regr.mae
##           <num>     <num>          <int>             <int>             <list>    <list>    <num>
## 1:    0.6793745 -1.993978             69                 7          <list[8]> <list[4]> 1.366524
```

위 초모수 중에는 로그 변환되어 있는 값들이 있으므로, GBM 모델에 전달되는 값은 다시 exp 변환이 적용된 값을 사용해야 한다.

```r
unlist(instance_gbm$result_learner_param_vals)
##           keep.data             n.cores        distribution             n.trees        bag.fraction 
##             "FALSE"                 "1"           "laplace"              "1000" "0.679374491563067" 
##           shrinkage      n.minobsinnode   interaction.depth 
## "0.136152798438148"
```

## XGBoost tuning

- Search Space: 

| Hyperparameter     | Type  | Description                                                | Range/Levels                                                                                 |
|:------------------ |:----- |:---------------------------------------------------------- |:-------------------------------------------------------------------------------------------- |
| `objective`        | `chr` | 목표 손실함수 지정                                         | `"reg:squarederror"`, `"reg:squredlogerror"`, `reg:pseudohubererror"`, `"reg:absoluteerror"` |
| `nrounds`          | `int` | 총 트리 수 지정                                            | $[1, \infty)$                                                                                |
| `boosting`         | `chr` | 어떤 부스팅 알고리즘 사용할지 지정                         | `"gbtree"`, `"dart"`, `"gblinear"`                                                           |
| `tree_method`      | `chr` | 트리 구축 방법 지정                                        | `"auto"`, `"exact"`, `"approx"`, `"hist"`, `"gpu_hist"`                                      |
| `eta`              | `dbl` | 학습률 지정                                                | $[0,1]$                                                                                      |
| `colsample_bytree` | `dbl` | 트리 구축 시 사용되는 피쳐 비율                            | $[0,1]$                                                                                      |
| `gamma`            | `dbl` | 리프 노드를 추가로 분할 하는 데 필요한 최소 손실 감소 지정 | $[0, \infty)$                                                                                |
| `max_depth`        | `int` | 트리의 최대 깊이 지정                                      | $[0, \infty)$                                                                                |
| `subsample`        | `dbl` | 각 트리 학습에 사용되는 데이터 비율 지정                   | $[0, 1]$                                                                                             |

- 튜닝 알고리즘: Bayesian Optimization with Holdout resampling

```r
lrn_xgb = lrn("regr.xgboost",
              nrounds = 1000,
              objective = "reg:absoluteerror",
              eval_metric = "mae",
              verbose = 2)
search_space = ps(
  booster = p_fct(c("gbtree", "dart")), 
  tree_method = p_fct("hist", depends = (booster %in% c("gbtree", "dart"))),
  eta = p_dbl(lower = 1e-04, upper = 1, logscale = T),
  colsample_bytree = p_dbl(lower = 0, upper = 1),
  gamma = p_dbl(lower = 1e-05, upper = 7, logscale = T),
  max_depth = p_int(lower = 1, upper = 15),
  subsample = p_dbl(lower = 1e-01, upper = 1)
)


future::plan("multisession", workers = 10)
instance = tune(
  tuner = tuner_bo,
  task = task,
  learner = lrn_xgb,
  resampling = rsmp("holdout"),
  measures = measures,
  search_space = search_space,
  term_evals = 25
)

instance_xgb$result
##    booster tree_method       eta colsample_bytree     gamma max_depth subsample learner_param_vals  x_domain regr.mae
##     <char>      <char>     <num>            <num>     <num>     <int>     <num>             <list>    <list>    <num>
## 1:    dart        hist -4.670661        0.7883241 -8.455197         5 0.8017724         <list[13]> <list[7]> 1.374055

unlist(instance_xgb$result_learner_param_vals)
##                nrounds                nthread                verbose     early_stopping_set              objective            eval_metric 
##                 "1000"                    "1"                    "2"                 "none"    "reg:absoluteerror"                  "mae" 
##                booster            tree_method                    eta       colsample_bytree                  gamma              max_depth 
##                 "dart"                 "hist"  "0.00936607984116761"    "0.788324117660522" "0.000212791589953104"                    "5" 
##              subsample 
##     "0.80177236199379"
```

##  LightGBM tuning

- Search Space:

| Hyperparameter     | Type  | Description                                  | Range/Levels                                               |
|:------------------ |:----- |:-------------------------------------------- |:---------------------------------------------------------- |
| `objective`        | `chr` | 모델의 목적 함수 지정                        | `"regression"`,`"regresion_l1"`, `"huber"`, `"poisson"` 등 |
| `boosting`         | `chr` | 부스팅 유형 지정                             | `"gbdt"`, `"rf"`, `"dart"`, `"goss"`                       |
| `num_iterations`   | `int` | 부스팅 반복 횟수 지정                        | $[0, \infty)$                                              |
| `max_depth`        | `int` | 트리 최대 깊이 지정                          | $(-\infty, \infty)$, 음수는 제한 없음을 의미               |
| `num_leaves`       | `int` | 하나의 트리가 가질 수 있는 최대 리프 수 지정 | $[1, 131072]$                                              |
| `learning_rate`    | `dbl` | 학습률 지정                                  | $[0, 1]$                                                   |
| `bagging_fraction` | `dbl` | 각 부스팅 단계에서 사용할 데이터 비율 지정   | $[0,1]$                                                    |
| `feature_fraction` | `dbl` | 각 부스팅 단계에서 사용할 피쳐 비율 지정     | $[0, 1]$                                                   |
| `lambda_l1`        | `dbl` | L1 정규화 가중치 지정                        | $[0, \infty)$                                              |
| `lambda_l2`        | `dbl` | L2 정규화 가중치 지정                        | $[0, \infty)$                                              |

- 튜닝 알고리즘: Bayesian Optimization with Holdout resampling

```r
lrn_lgbm = lrn("regr.lightgbm",
               num_iterations = 1000,
               objective = "regression_l1",
               boosting = "gbdt")

search_space = ps(``
  max_depth = p_int(lower = 1, upper = 15),
  num_leaves = p_int(lower = 2, upper = 1024),
  learning_rate = p_dbl(lower = 1e-04, upper = 0.1, logscale = T),
  bagging_fraction = p_dbl(lower = 0.4, upper = 1),
  feature_fraction = p_dbl(lower = 0.7, upper = 1),
  lambda_l1 = p_dbl(lower = 1e-08, upper = 10.0),
  lambda_l2 = p_dbl(lower = 1e-08, upper = 10.0)
)

### BO building blocks
bayesopt_ego = mlr_loop_functions$get("bayesopt_ego")
surrogate = srlrn(
  lrn("regr.km", 
      covtype = "matern5_2", 
      optim.method = "BFGS",
      control = list(trace = F))
)
acq_function = acqf("ei")
acq_optimizer = acqo(
  optimizer = opt("random_search", batch_size = 100),
  terminator = trm("stagnation", iters = 100, threshold = 1e-5)
)
tuner_bo = tnr(
  "mbo",
  loop_function = bayesopt_ego,
  surrogate = surrogate,
  acq_function = acq_function,
  acq_optimizer = acq_optimizer
)
future::plan("multisession", workers = 10)
instance = tune(
  tuner = tuner_bo,
  task = task,
  learner = lrn_lgbm,
  resampling = rsmp("holdout"),
  measures = measures,
  search_space = search_space,
  term_evals = 25
)

instance_lgbm$result
##    max_depth num_leaves learning_rate bagging_fraction feature_fraction lambda_l1 lambda_l2 learner_param_vals  x_domain regr.mae
##        <int>      <int>         <num>            <num>            <num>     <num>     <num>             <list>    <list>    <num>
## 1:         8         70     -3.646907        0.9854945        0.8244073  8.628138  9.672471         <list[13]> <list[7]> 1.362118

unlist(instance_lgbm$result_learner_param_vals)
##          num_threads              verbose            objective  convert_categorical       num_iterations             boosting 
##                  "1"                 "-1"      "regression_l1"               "TRUE"               "1000"               "gbdt" 
##            max_depth           num_leaves        learning_rate     bagging_fraction     feature_fraction            lambda_l1 
##                  "8"                 "70" "0.0260716424607687"  "0.985494470596313"  "0.824407345056534"   "8.62813830512857" 
##            lambda_l2 
##   "9.67247068914742"
```

## CatBoost tuning

- Search Space:

| Hyperparameter     | Type  | Description                                           | Range/Levels                                              |
|:------------------ |:----- |:----------------------------------------------------- |:--------------------------------------------------------- |
| `loss_function`    | `chr` | 손실함수 지정                                         | `"MAE"`, `"MAPE"`, `"Poisson"`, `"Quantile"`, `"RMSE` 등  |
| `bootstrap_type`   | `chr` | 부트스트랩 유형 지정                                  | `"Bayesian"`, `"Bernoulli"`, `"MVS"`, `"Poisson"`, `"No"` |
| `iterations`       | `int` | 최대 트리 수 지정                                     | $[1, \infty)$                                             |
| `depth`            | `int` | 트리 최대 깊이 지정                                   | $[1, 16]$                                                 |
| `learning_rate`    | `dbl` | 학습률 지정                                           | $[0.001, 1]$                                              |
| `l2_leaf_reg`      | `dbl` | L2 정규화 계수 지정                                   | $[0,\infty)$                                              |
| `random_strength`  | `dbl` | 트리 구조 선택 시 사용되는 무작위성의 양 지정         | $[0,\infty)$                                              |
| `grow_policy`      | `chr` | 트리 성장 방식 지정                                   | `"SymmetricTree"`, `"Depthwise"`, `"Lossguide"`           |
| `min_data_in_leaf` | `int` | 트리의 각 리프 노드에 있어야 하는 최소 관측치 수 지정 | $[1, \infty)$                                                          |

```r
lrn_cb = lrn("regr.catboost",
             iterations = 1000,
             loss_function = "MAE",
             bootstrap_type = "Bayesian")

search_space = ps(
  depth = p_int(lower = 1, upper = 15),
  learning_rate = p_dbl(lower = 1e-03, upper = 0.1, logscale = T),
  l2_leaf_reg = p_dbl(lower = 1e-04, upper = 100, logscale = T),
  random_strength = p_dbl(lower = 0, upper = 100),
  grow_policy = p_fct(c("SymmetricTree", "Depthwise", "Lossguide")),
  min_data_in_leaf = p_int(lower = 1, upper = 30, depends = (grow_policy %in% c("Depthwise", "Lossguide")))
)

tuner_bo = tnr("mbo")
future::plan("multisession", workers = 10)
instance = tune(
  tuner = tuner_bo,
  task = task,
  learner = lrn_cb,
  resampling = rsmp("holdout"),
  measures = measures,
  search_space = search_space,
  term_evals = 25
)

instance_cb$result
##    depth learning_rate l2_leaf_reg random_strength grow_policy min_data_in_leaf learner_param_vals  x_domain regr.mae
##    <int>         <num>       <num>           <num>      <char>            <int>             <list>    <list>    <num>
## 1:     7     -4.151767   -7.149902       0.7074093   Depthwise                3         <list[13]> <list[6]> 1.364169

unlist(instance_cb$result_learner_param_vals)
##          loss_function          logging_level           thread_count    allow_writing_files          save_snapshot 
##                  "MAE"               "Silent"                    "1"                "FALSE"                "FALSE" 
##             iterations         bootstrap_type                  depth          learning_rate            l2_leaf_reg 
##                 "1000"             "Bayesian"                    "7"   "0.0157365896320895" "0.000784940744324096" 
##        random_strength            grow_policy       min_data_in_leaf 
##    "0.707409344613552"            "Depthwise"                    "3"
```

## Stacking Model

- GBM, XGBoost, LightGBM, Catboost 각 모델의 예측치를 피쳐로 하는 Stacking model 구축
- `mlr3`에는 Least Absolute Deviation regression에 대한 learner가 따로 없으므로 `quantreg` 패키지의 [`rq()`](https://www.rdocumentation.org/packages/quantreg/versions/5.97/topics/rq) 함수를 `R6`클래스로 정의해 `LearnerRegrRQ`를 만들자.

```r
#' @title Quantile Regression Learner
#' 
#' @name mlr_learners_regr.rq
#' 
#' @description
#' Quantile regression
#' Calls [quantreg::rq()].
#' 
#' @templateVar id regr.rq
#' @template learner
#' 
#' @export
#' @template seealso_learner
#' @template example
LearnerRegrRQ = R6::R6Class("LearnerRegrRQ",
  inherit = mlr3::LearnerRegr,
  
  public = list(
    #' @description
        #' Create a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps = ps(
        tau           = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train"),
        model         = p_lgl(default = TRUE, tags = "train"),
        x             = p_lgl(default = FALSE, tags = "train"),
        y             = p_lgl(default = FALSE, tags = "train"),
        contrasts     = p_uty(tags = "train"),
        method        = p_fct(c("br", "fn", "pfn", "bdp", "lasso"), default = "br", tags = "train"),
        se            = p_fct(c("iid", "nid", "ker", "boot"), default = "iid", tags = "predict"),
        verbose       = p_lgl(default = FALSE, tags = "predict")
      )
      
      super$initialize(
        id = "regr.rq",
        param_set = ps,
        predict_types = c("response", "se"),
        feature_types = c("logical", "integer", "numeric", "factor", "character"),
        properties = c("weights"),
        packages = c("mlr3learners", "quantreg"),
        label = "Quantile Regression",
        man = "mlr3learners::mlr_learners_regr.rq"
      )
    } 
  ),
  
  private = list(
    .train = function(task) {
      pv = self$param_set$get_values(tags = "train")
      if ("weights" %in% task$properties) {
        pv$weights = task$weights$weight
      }
      
      # method 인자가 필요하지 않거나 올바른 값이 설정되었는지 확인
      args_list = list(formula = task$formula(), data = task$data(), tau = pv$tau)
      if (!is.null(pv$method) && pv$method %in% c("fn", "fnb", "br")) {
        args_list$method = pv$method
      }
      if (!is.null(pv$model)) {
        args_list$model = pv$model
      }
      if (!is.null(pv$x)) {
        args_list$x = pv$x
      }
      if (!is.null(pv$y)) {
        args_list$y = pv$y
      }
      
      # quantreg::rq() 함수에 필요한 인자들만 전달
      self$model = do.call(quantreg::rq, args_list)
    },
    
    .predict = function(task) {
      pv = self$param_set$get_values(tags = "predict")
      newdata = ordered_features(task, self)
      se_fit = self$predict_type == "se"
      prediction = invoke(predict, object = self$model, newdata = newdata, se.fit = se_fit, .args = pv)
      
      if (se_fit) {
        list(response = unname(prediction$fit), se = unname(prediction$se.fit))
      } else {
        list(response = unname(prediction))
      }
    }
  )
)

ordered_features = function(task, learner) {
  cols = names(learner$state$data_prototype) %??% learner$state$feature_names
  task$data(cols = intersect(cols, task$feature_names))
}
```

![[Pasted image 20240402010419.png]]

- Stacking model을 이루는 각 base model의 예측치는 2-fold CV로 측정해 과적합을 예방할 수 있다.

```r
po_gbm_cv = po("learner_cv", learner = lrn_gbm, resampling.folds = 2, id = "gbm_cv")
po_xgb_cv = po("learner_cv", learner = lrn_xgb, resampling.folds = 2, id = "xgboost_cv")
po_lgbm_cv = po("learner_cv", learner = lrn_lgbm, resampling.folds = 2, id = "lightgbm_cv")
po_cb_cv = po("learner_cv", learner = lrn_cb, resampling.folds = 2, id = "catboost_cv")

gr_level_0 = gunion(list(po_gbm_cv, po_xgb_cv, po_lgbm_cv, po_cb_cv))
gr_combined = gr_level_0 %>>% po("featureunion")

# Super learner: LAD regressor
lrn_lad = LearnerRegrRQ$new()
lrn_lad$param_set$values = list(tau = 0.5)

stack_lad = gr_combined %>>% po("learner", lrn_lad)
stack_lad$plot(horizontal = TRUE)
```

![[Pasted image 20240402010629.png]]

- `mlr3`의 `benchmark` 기능으로 이 스태킹을 구현할 수 있지만, 분석하고자 하는 데이터는 원본 데이터와 합성 데이터를 합친 것이기 때문에 일반화 성능을 측정할 때 합성 데이터에 대해서만 측정할 수도 있다. 
- 우선, 아래와 같이 각 모델을 병합된 데이터로 훈련시킨 후, 합성 데이터로만 일반화 성능을 측정해보자: K-fold CV 사용

```r
# generated = 1인 데이터로만 stacking model의 features 구성하기
cv = rsmp("cv", folds = 10)
cv$instantiate(task)

cv$instance

gbm_cv_scores = list(); gbm_preds = list()
xgb_cv_scores = list(); xgb_preds = list()
lgbm_cv_scores = list(); lgbm_preds = list()
cb_cv_scores = list(); cb_preds = list()
ens_cv_scores = list(); ens_preds = list()

for (i in 1:10){
  cat("---------------------------------------------------------------\n")
  gbm_learner = lrn_gbm$clone()
  xgb_learner = lrn_xgb$clone()
  lgbm_learner = lrn_lgbm$clone()
  cb_learner = lrn_cb$clone()
  lad_learner = lrn_lad$clone()
  
  test_tsk = task$clone()$data(rows = cv$test_set(i)) %>% 
    filter(generated == 1) %>% 
    as_task_regr(target = 'Age')
  
  gbm_learner$train(task, row_ids = cv$train_set(i))
  gbm_pred_1 = gbm_learner$predict(test_tsk)
  gbm_pred_2 = gbm_learner$predict_newdata(newdata = test_baseline)
  gbm_score_fold = gbm_pred_1$score(msr("regr.mae"))
  gbm_cv_scores = append(gbm_cv_scores, gbm_score_fold)
  gbm_preds = append(gbm_preds, list(gbm_pred_2$response))
  cat("Fold", i, "==> GBM oof MAE is ==>", gbm_score_fold, "\n")
  
  xgb_learner$train(task, row_ids = cv$train_set(i))
  xgb_pred_1 = xgb_learner$predict(test_tsk)
  xgb_pred_2 = xgb_learner$predict_newdata(newdata = test_baseline)
  xgb_score_fold = xgb_pred_1$score(msr("regr.mae"))
  xgb_cv_scores = append(xgb_cv_scores, xgb_score_fold)
  xgb_preds = append(xgb_preds, list(xgb_pred_2$response))
  cat("Fold", i, "==> XGBoost oof MAE is ==>", xgb_score_fold, "\n")
  
  lgbm_learner$train(task, row_ids = cv$train_set(i))
  lgbm_pred_1 = lgbm_learner$predict(test_tsk)
  lgbm_pred_2 = lgbm_learner$predict_newdata(newdata = test_baseline)
  lgbm_score_fold = lgbm_pred_1$score(msr("regr.mae"))
  lgbm_cv_scores = append(lgbm_cv_scores, lgbm_score_fold)
  lgbm_preds = append(lgbm_preds, list(lgbm_pred_2$response))
  cat("Fold", i, "==> LightGBM oof MAE is ==>", lgbm_score_fold, "\n")
  
  cb_learner$train(task, row_ids = cv$train_set(i))
  cb_pred_1 = cb_learner$predict(test_tsk)
  cb_pred_2 = cb_learner$predict_newdata(newdata = test_baseline)
  cb_score_fold = cb_pred_1$score(msr("regr.mae"))
  cb_cv_scores = append(cb_cv_scores, cb_score_fold)
  cb_preds = append(cb_preds, list(cb_pred_2$response))
  cat("Fold", i, "==> Catboost oof MAE is ==>", cb_score_fold, "\n")
  
  
  stack_data = data.frame(GBM = gbm_pred_1$response, XGB = xgb_pred_1$response, LGBM = lgbm_pred_1$response, CB = cb_pred_1$response,
                          Age = test_tsk$data()$Age)
  stack_data = as_task_regr(stack_data, target = 'Age')
  lad_learner$train(stack_data)
  lad_pred = lad_learner$predict(stack_data)
  
  stack_data_test = data.frame(GBM = gbm_pred_2$response, XGB = xgb_pred_2$response, LGBM = lgbm_pred_2$response, CB = cb_pred_2$response)
  lad_pred_test = lad_learner$predict_newdata(newdata = stack_data_test)
  
  ens_score = lad_pred$score(msr("regr.mae"))
  ens_cv_scores = append(ens_cv_scores, ens_score)
  ens_preds = append(ens_preds, list(lad_pred_test$response))
  cat("Fold", i, "==> LAD ensemble oof MAE is ==>", ens_score, "\n")
  cat("---------------------------------------------------------------\n")
}

## ---------------------------------------------------------------
## Fold 1 ==> GBM oof MAE is          ==> 1.350713 
## Fold 1 ==> XGBoost oof MAE is      ==> 1.360883 
## Fold 1 ==> LightGBM oof MAE is     ==> 1.349277 
## Fold 1 ==> Catboost oof MAE is     ==> 1.350295 
## Fold 1 ==> LAD ensemble oof MAE is ==> 1.345693 
## ---------------------------------------------------------------
## ---------------------------------------------------------------
## Fold 2 ==> GBM oof MAE is          ==> 1.366546 
## Fold 2 ==> XGBoost oof MAE is      ==> 1.376786 
## Fold 2 ==> LightGBM oof MAE is     ==> 1.361139 
## Fold 2 ==> Catboost oof MAE is     ==> 1.368145 
## Fold 2 ==> LAD ensemble oof MAE is ==> 1.359603 
## ---------------------------------------------------------------
## ---------------------------------------------------------------
## Fold 3 ==> GBM oof MAE is          ==> 1.377278 
## Fold 3 ==> XGBoost oof MAE is      ==> 1.359011 
## Fold 3 ==> LightGBM oof MAE is     ==> 1.343973 
## Fold 3 ==> Catboost oof MAE is     ==> 1.348503 
## Fold 3 ==> LAD ensemble oof MAE is ==> 1.342466 
## ---------------------------------------------------------------
## ---------------------------------------------------------------
## Fold 4 ==> GBM oof MAE is          ==> 1.354687 
## Fold 4 ==> XGBoost oof MAE is      ==> 1.367154 
## Fold 4 ==> LightGBM oof MAE is     ==> 1.347348 
## Fold 4 ==> Catboost oof MAE is     ==> 1.352126 
## Fold 4 ==> LAD ensemble oof MAE is ==> 1.34495 
## ---------------------------------------------------------------
## ---------------------------------------------------------------
## Fold 5 ==> GBM oof MAE is          ==> 1.356053 
## Fold 5 ==> XGBoost oof MAE is      ==> 1.354567 
## Fold 5 ==> LightGBM oof MAE is     ==> 1.33048 
## Fold 5 ==> Catboost oof MAE is     ==> 1.337845 
## Fold 5 ==> LAD ensemble oof MAE is ==> 1.328581 
## ---------------------------------------------------------------
## ---------------------------------------------------------------
## Fold 6 ==> GBM oof MAE is          ==> 1.370625 
## Fold 6 ==> XGBoost oof MAE is      ==> 1.364186 
## Fold 6 ==> LightGBM oof MAE is     ==> 1.344058 
## Fold 6 ==> Catboost oof MAE is     ==> 1.350118 
## Fold 6 ==> LAD ensemble oof MAE is ==> 1.342113 
## ---------------------------------------------------------------
## ---------------------------------------------------------------
## Fold 7 ==> GBM oof MAE is          ==> 1.339123 
## Fold 7 ==> XGBoost oof MAE is      ==> 1.345509 
## Fold 7 ==> LightGBM oof MAE is     ==> 1.328713 
## Fold 7 ==> Catboost oof MAE is     ==> 1.334171 
## Fold 7 ==> LAD ensemble oof MAE is ==> 1.328012 
## ---------------------------------------------------------------
## ---------------------------------------------------------------
## Fold 8 ==> GBM oof MAE is          ==> 1.369543 
## Fold 8 ==> XGBoost oof MAE is      ==> 1.379594 
## Fold 8 ==> LightGBM oof MAE is     ==> 1.371251 
## Fold 8 ==> Catboost oof MAE is     ==> 1.372786 
## Fold 8 ==> LAD ensemble oof MAE is ==> 1.366846 
## ---------------------------------------------------------------
## ---------------------------------------------------------------
## Fold 9 ==> GBM oof MAE is          ==> 1.370098 
## Fold 9 ==> XGBoost oof MAE is      ==> 1.380159 
## Fold 9 ==> LightGBM oof MAE is     ==> 1.361718 
## Fold 9 ==> Catboost oof MAE is     ==> 1.369566 
## Fold 9 ==> LAD ensemble oof MAE is ==> 1.360734 
## ---------------------------------------------------------------
## ---------------------------------------------------------------
## Fold 10 ==> GBM oof MAE is          ==> 1.376444 
## Fold 10 ==> XGBoost oof MAE is      ==> 1.386378 
## Fold 10 ==> LightGBM oof MAE is     ==> 1.367396 
## Fold 10 ==> Catboost oof MAE is     ==> 1.375223 
## Fold 10 ==> LAD ensemble oof MAE is ==> 1.36636 
## ---------------------------------------------------------------


results = list(gbm_cv_scores = gbm_cv_scores, gbm_preds = gbm_preds, 
               xgb_cv_scores = xgb_cv_scores, xgb_preds = xgb_preds, 
               lgbm_cv_scores = lgbm_cv_scores, lgbm_preds = lgbm_preds,
               cb_cv_scores = cb_cv_scores, cb_preds = cb_preds, 
               ens_cv_scores = ens_cv_scores, ens_preds = ens_preds) 

data.frame(GBM = mean(unlist(results$gbm_cv_scores)), XGBoost = mean(unlist(results$xgb_cv_scores)),
           LightGBM = mean(unlist(results$lgbm_cv_scores)), CatBoost = mean(unlist(results$cb_cv_scores)),
           `LAD Ensemble` = mean(unlist(results$ens_cv_scores))) %>% 
  pivot_longer(everything()) %>% 
  mutate(name = factor(name, levels = rev(c("GBM", "XGBoost", "LightGBM", "CatBoost", "LAD.Ensemble")))) %>% 
  ggplot(aes(x = name, y = value, fill = name)) +
  geom_col(color = "white") +
  geom_text(aes(label = round(value, 6)), hjust = -0.1) +
  scale_y_continuous(breaks = seq(0, 1.6, 0.2)) +
  ylim(c(0, 1.5)) +
  ggsci::scale_fill_nejm() + 
  coord_flip() +
  theme_minimal() +
  theme(legend.position = 'none', axis.title.y = element_blank()) +
  labs(y = "CV_score")
```

![[Pasted image 20240402171742.png]]

```r
gbm_preds_test = data.frame(results$gbm_preds) %>% apply(1, mean)
xgb_preds_test = data.frame(results$xgb_preds) %>% apply(1, mean)
lgbm_preds_test = data.frame(results$lgbm_preds) %>% apply(1, mean)
cb_preds_test = data.frame(results$cb_preds) %>% apply(1, mean)
ens_preds_test = data.frame(results$ens_preds) %>% apply(1, mean)

submission$Age = as.integer(round(gbm_preds_test))
submission %>% write.csv("../data/playground-series-s3e16/GBM_V1_submission.csv", row.names = F)

submission$Age = as.integer(round(xgb_preds_test))
submission %>% write.csv("../data/playground-series-s3e16/XGBoost_V1_submission.csv", row.names = F)

submission$Age = as.integer(round(lgbm_preds_test))
submission %>% write.csv("../data/playground-series-s3e16/LightGBM_V1_submission.csv", row.names = F)

submission$Age = as.integer(round(cb_preds_test))
submission %>% write.csv("../data/playground-series-s3e16/Catboost_V1_submission.csv", row.names = F)

submission$Age = as.integer(round(ens_preds_test))
submission %>% write.csv("../data/playground-series-s3e16/LAD_Ensemble_V1_submission.csv", row.names = F)
```

![[Pasted image 20240402210734.png]]

- 리더보드 258/1431의 성능
- 추가적인 Feature engineering을 하면 더 좋은 성능을 낼 수 있으나, 여기서 stop...

