---
tags:
  - mlr3
  - XGBoost
  - Early_Stopping
  - Callback
date: 2024-04-04
description: Simultaneously optimize hyperparameters and use early stopping.
---

![[Pasted image 20240404140615.png]]


이번 글에서는 XGBoost의 Early Stopping 기능을 사용해서 모델 학습 시 오버피팅을 줄이는 방법에 대해 살펴본다. 먼저 Early Stopping과 과적합(overfitting)에 대한 간단한 설명을 시작으로, XGBoost의 Early Stopping 기능을 사용하여 `Spam Classification` 데이터셋에 대해 학습한다. 마지막으로 초모수를 튜닝하는 동시에 Early Stopping을 사용하는 방법에 대해 다룬다. 

# Early Stopping

반복 프로세스(iterative process)에서 모델 학습 시 Early Stopping을 사용하면 과적합을 줄일 수 있다. 과적합은 모델이 train data에서는 좋은 성능을 내지만 test data와 같은 unseen data에서는 성능이 저하될 때 발생하는 문제이다. Early Stopping 기능을 사용할 때는 test set에서 성능을 모니터링하고, 특정 반복 횟수에서 성능이 저하되면 학습을 중지시킨다.

## XGBoost with Early Stopping

재현성(Reproducibility)을 위해 고정된 seed로 시작한다.

```r
set.seed(7832)

library(mlr3verse)
library(ggplot2)
library(data.table)
```

XGBoost에서 모델 학습에 Early Stopping 기능을 사용하여 최적의 부스팅 횟수(`nrounds`)와 같은 값을 찾을 수 있다. 우선은 `partition()` 함수를 통해 데이터셋을 train-test로 분할하자. 여기서는 전체 관측치의 80%를 모델 학습에 사용하고, 20%를 성능을 모니터링하기 위한 test set으로 사용한다. 

```r
task = tsk("spam")
split = partition(task, ratio = 0.8, stratify = T)
task$set_row_roles(rows = split$test, roles = "test")
```

`early_stopping_set` 파라미터는 성능을 모니터링하는 데 사용되는 데이터셋(예: test set)을 제어한다. 또한 `early_stopping_rounds`로 성능이 증가해야 하는 범위를 정의하고 `nrounds`로 최대 부스팅 라운드 수를 지정한다. 아래 예에서는 최소 100개에서 최대 1000개의 부스팅 라운드 수를 갖는 모델을 구축한다.

- 예: 204번째 라운드까지 성능이 증가하다가 205번째부터 304번째까지 성능이 개선되지 않으면 최종 라운드 수는 304개가 되는 것이다.

```r
learner = lrn("classif.xgboost",
  nrounds = 1000,
  early_stopping_rounds = 100,
  early_stopping_set = "test",
  eval_metric = "error")

learner$train(task)
```

학습된 모델의 `$evaluation_log`에는 train & test set의 성능 점수가 저장되어 있다. 아래 그림은 train set의 classification error는 감소하는 반면, test set의 error는 20 라운드 이후 증가함을 보여준다. 

```r
data = melt(learner$model$evaluation_log, id.vars = "iter", variable.name = "set", value.name = "error")
ggplot(data, aes(x = iter, y = error, group = set)) +
  geom_line(aes(color = set)) +
  geom_vline(aes(xintercept = learner$model$best_iteration), color = "grey") +
  scale_x_continuous(breaks = seq(0, 150, 25)) +
  scale_color_manual(values = c("#f8766d", "#00b0f6"), labels = c("Train", "Test")) +
  labs(x = "Rounds", y = "Classification Error", color = "Set") +
  theme_minimal()
```

![[Pasted image 20240404143948.png]] Figure 1: Comparison b/w train and test set classification error.

학습된 모델 결과인 `learner$model`의 `$best_iteration` 슬롯에는 최적 부스팅 라운드 수가 들어있다. 

```r
learner$model$best_iteration
## [1] 48
```

`learner$predict()`를 통한 예측은 최적의 모델이 아닌 마지막 반복의 모델을 사용한다는 점을 유의하자. 최적 부스팅 라운드 및 초모수 조합을 튜닝하는 방법을 알아보자. 

# Tuning

이제 XGBoost의 초모수를 튜닝하고 최적의 부스팅 라운드 수를 한 번에 찾아보자. 이를 위해 튜닝 프로세스 도중에 Early Stopping 기능을 제공하는 `early stopping callback`을 사용해야 한다. 초모수 조합의 성능은 k-fold CV와 같은 리샘플링 전략(resampling strategy)을 통해 평가된다. 각 리샘플링 반복에서 새로운 XGBoost 모델을 학습하고 Early Stopping을 사용해 최적의 부스팅 라운드 수(`nrounds`)를 찾는다. 따라서 3-fold CV를 적용할 경우 하나의 초모수 조합에 대해 3가지 최적의 `nrounds` 값이 도출된다. Callback은 이 3가지 값 중에 최대값을 선택하여 아카이브에 기록한다. 최종 모델이 전체 데이터셋에 적합되기 때문에 최대값을 고르는 것이다. 

우선 XGBoost learner를 로드하고 Early Stopping parameters를 설정하자. 

```r
learner = lrn("classif.xgboost",
  nrounds = 1000,
  early_stopping_rounds = 100,
  early_stopping_set = "test")
```

그런 다음 `mlr3tuningspaces` 패키지에서 pre-defined tuning space를 로드하자. 여기에는 XGBoost에서 가장 일반적으로 사용되는 tuning space가 포함되어 있다.

```r
tuning_space = lts("classif.xgboost.default")
as.data.table(tuning_space)
##                   id lower upper logscale
##               <char> <num> <num>   <lgcl>
## 1:               eta 1e-04     1     TRUE
## 2:           nrounds 1e+00  5000    FALSE
## 3:         max_depth 1e+00    20    FALSE
## 4:  colsample_bytree 1e-01     1    FALSE
## 5: colsample_bylevel 1e-01     1    FALSE
## 6:            lambda 1e-03  1000     TRUE
## 7:             alpha 1e-03  1000     TRUE
## 8:         subsample 1e-01     1    FALSE
```

이 tuning/search space를 learner에게 전달한다. 

```r
learner = lts(learner)
```

이 default tuning space에는 `nrounds` 초모수가 포함되어 있다. Early Stopping 기능을 사용하기 위해서는, 이를 상한값으로 덮어써야(overwrite) 한다. 

```r
learner$param_set$set_values(nrounds = 1000)
```

이제 적은 수의 batch를 사용하는 Random Search로 초모수 튜닝을 실행하자. 

```r
instance = tune(
  tuner = tnr("random_search", batch_size = 2),
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 3),
  measure = msr("classif.ce"),
  term_evals = 4,
  callbacks = clbk("mlr3tuning.early_stopping")
)
```

`instace$archive`를 확인해보면 최적의 부스팅 라운드 수(`max_rounds`)는 초모수 조합에 따라 많이 다름을 알 수 있다. 

```r
as.data.table(instance$archive)[, .(batch_nr, max_nrounds, eta, max_depth, colsample_bylevel, lambda, alpha, subsample)]
##    batch_nr max_nrounds       eta max_depth colsample_bylevel     lambda      alpha subsample
##       <int>       <num>     <num>     <int>             <num>      <num>      <num>     <num>
## 1:        1         596 -2.653145        14         0.6386892 -2.3525352 -0.0101181 0.4277725
## 2:        1        1000 -8.693775        11         0.8460060 -0.3043504 -4.2723771 0.2201520
## 3:        2        1000 -7.835692        17         0.8326459 -2.7548611  3.2816116 0.1788330
## 4:        2         107 -2.404063        20         0.6913686 -0.6684124 -1.5078548 0.8922117
```

최적의 초모수 조합에서는 `nrounds` 값이 `max_rounds`로 대체되고 Early Stopping 기능이 비활성화된다. 

```r
instance$result_learner_param_vals
## $nrounds
## [1] 596
## 
## $nthread
## [1] 1
## 
## $verbose
## [1] 0
## 
## $early_stopping_set
## [1] "none"
## 
## $eta
## [1] 0.0704294
## 
## $max_depth
## [1] 14
## 
## $colsample_bytree
## [1] 0.5763381
## 
## $colsample_bylevel
## [1] 0.6386892
## 
## $lambda
## [1] 0.09512769
## 
## $alpha
## [1] 0.9899329
## 
## $subsample
## [1] 0.4277725
```

마지막으로 전체 데이터셋에 최종 모델을 적합하면 된다. 

```r
learner = lrn("classif.xgboost")
learner$param_set$values = instance$result_learner_param_vals
learner$train(task)
```

이제 이렇게 학습한 모델을 사용해서 새로운 unseen data에 대한 예측을 수행할 수 있다. 

- 출처: [Early Stopping with XGBoost: mlr-org/gallery](https://mlr-org.com/gallery/optimization/2022-11-04-early-stopping-with-xgboost/)
