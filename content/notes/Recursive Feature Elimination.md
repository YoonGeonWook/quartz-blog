---
tags:
  - mlr3
  - RFE
  - RFE-CV
date: 2024-04-02
---

# Recursive Feature Elimination on the Sonar Data set

머신러닝 모델의 성능, 해석, 강건성(robustness)를 개선하기 위해 변수 선택(Feature Selection)을 사용할 수 있다. Recursive Feature Elimination (RFE)은 이 변수 선택의 한 종류로 생각하면 된다. 여기서는 `Sonar` 데이터셋에 대해 Gradient Boosting Machine (GBM)과 Linear Support Vector Machine (SVM)을 사용해 RFE를 적용하는 방법을 소개한다.

## RFE

RFE는 고차원 데이터셋에 널리 사용되는 변수 선택 방법으로, 사용자가 지정한 변수 개수에 도달할 때까지 모델에서 예측력이 가장 낮은 변수를 반복적으로 제거하는 식으로 작동한다. 여기서 예측력이라 함은 모델에 내장된 변수 중요도(feature importance)를 말한다.

[Guyon et al. (2002)](https://link.springer.com/article/10.1023/A:1012487302797)는 cancer classification 문제에서 informative genes을 선택하기 위해 SVM-RFE를 도입했다. 이들이 사용한 모델은 Linear SVM인데, Linear SVM은 모델이 자체적으로 변수 중요도를 지원하지 않기 때문에 가중치 벡터를 사용하여 RFE를 사용했다. 

- scikit-learn의 [RFE](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)에서는 변수 중요도를 지원하지 않는 모델에 대해서는 가중치 벡터 `coef_`를 사용해 RFE를 적용한다.
- RFE 작동 방식: 
	- 먼저 사용할 모델을 정의하고, 해당 모델을 사용해 모든 feature(full models)들을 활용해 데이터를 학습했을 때 각 feature의 변수 중요도를 계산한다. 
	- 그리고 중요도가 낮은 feature부터 하나씩 제거해가면서 지정한 변수 개수가 될 때까지 이 과정을 반복한다. 

`mlr3`에서 RFE를 사용하기 위해서는 Learner의 속성에 `"importance"`가 들어있어야 한다.

```r
library(mlr3verse)
as.data.table(mlr_learners)[sapply(properties, function(x) "importance" %in% x),
                            .(key, label)]
## Key: <key>
##                          key                              label
##                       <char>                             <char>
##  1:         classif.catboost                  Gradient Boosting
##  2:      classif.featureless Featureless Classification Learner
##  3:              classif.gbm                  Gradient Boosting
##  4: classif.imbalanced_rfsrc           Imbalanced Random Forest
##  5:         classif.lightgbm                  Gradient Boosting
##  6:     classif.randomForest                      Random Forest
##  7:           classif.ranger                      Random Forest
##  8:            classif.rfsrc                      Random Forest
##  9:            classif.rpart                Classification Tree
## 10:          classif.xgboost          Extreme Gradient Boosting
## 11:            regr.catboost                  Gradient Boosting
## 12:         regr.featureless     Featureless Regression Learner
## 13:                 regr.gbm                  Gradient Boosting
## 14:            regr.lightgbm                  Gradient Boosting
## 15:        regr.randomForest                      Random Forest
## 16:              regr.ranger                      Random Forest
## 17:               regr.rfsrc                      Random Forest
## 18:               regr.rpart                    Regression Tree
## 19:             regr.xgboost          Extreme Gradient Boosting
## 20:               surv.aorsf              Oblique Random Forest
## 21:                surv.bart Bayesian Additive Regression Trees
## 22:            surv.gamboost Boosted Generalized Additive Model
## 23:                 surv.gbm                  Gradient Boosting
## 24:              surv.mboost Boosted Generalized Additive Model
## 25:              surv.ranger                      Random Forest
## 26:               surv.rfsrc                      Random Forest
## 27:             surv.xgboost                  Gradient Boosting
##                          key                              label
```

`fs()` 함수를 통해 RFE optimizer를 정의:

```r
optimizer = fs("rfe",
  n_features = 1,
  feature_number = 1,
  aggregation = "rank"
)
```

- `n_features`: 최종적으로 남기고자 하는 변수 개수 지정 
	- default: 전체 변수의 절반
- `feature_fraction`: 각 반복에서 유지할 변수의 비율
	- default: `0.5`
- `feature_number`: 각 반복에서 제거할 변수 개수
- `subset_sizes`: 각 반복마다 유지할 변수 개수 벡터
	- 내림차순이어야 함
- `recursive`: 
	- `TRUE` - 각 반복에서 변수 중요도를 계산
	- `FALSE` - 첫 번째 반복에서 계산한 변수 중요도를 계속 사용
- `aggregation`: 변수 중요도가 집계되는 방식 설정
	- default: `"rank"` - 각 반복의 중요도 벡터에 대해 순위가 매겨지고, 평균 순위가 낮은 변수가 제거됨
	- `"mean"` - 리샘플링 반복에 걸쳐 각 변수 중요도를 평균 내고, 평균 중요도 값이 낮은 변수 제거

이렇게 정의한 optimizer는 변수를 제거해가며, 변수 개수가 `n_features`와 같아지면 중지한다. `feature_fraction`, `feature_number`, `subset_sizes`는 각 반복에서 제거할 변수 개수를 결정하는 parameter이다. `feature_number`는 각 반복에서 고정된 개수만큼 제거하고, `feature_fraction`은 제거할 변수 수의 비율을 지정하고, `subset_sizes`는 각 반복에서 제거되는 변수 개수를 정확히 지정하는 벡터를 지정한다. 이 세 개의 파라미터는 하나가 사용되면 나머지는 사용할 수 없다. 

## Task

여기서 예시로 사용할 데이터는 `Sonar` 데이터셋인데, 이 데이터의 목적은 sonar signal이 금속(metal; `"M"`) 실린더에 반사되었는지, 바위(rock; `"R"`)에 반사되었는지를 예측하는 것이다. 총 60개의 수치형 변수가 들어있다.

> [!note]- code fold
> ```r
> library(ggplot2)
> library(data.table)
> library(tidyverse)
> data = as.data.table(task) %>% 
>   melt(id.vars = task$target_names, measure.vars = task$feature_names)
> data = data[c("V1", "V10", "V11", "V12", "V13", "V14"), ,on = "variable"]
> 
> ggplot(data, aes(x = value, fill = Class)) +
>   geom_density(alpha = 0.5) +
>   facet_wrap(~variable, ncol = 6, scales = "free") +
>   scale_fill_viridis_d(end = 0.8) +
>   theme_minimal() +
>   theme(axis.title.x = element_blank())
> ```


![[Pasted image 20240402235723.png]] Figure 1: `Sonar` 데이터의 첫 5개 변수의 분포

## Gradient Boosting Machine

예측 유형을 `"prob"`으로 설정하여 `GBM Learner`를 정의하자. 

```r
learner = lrn("classif.gbm",
  distribution = "bernoulli",
  predict_type = "prob")
```

이제 `fsi()` 함수를 통해 변수 선택 문제를 인스턴스로 정의하자. 각 반복에서 변수들의 서브셋에 대한 성능을 평가하기 위해서 Task, Learner, Resampling Strategy, Measures를 선택해야 한다. Optimizer의 `n_features`가 변수 선택 반복의 중지 시점을 결정하기 때문에 따로 중단 기준을 설정할 필요 없이 `"none"` terminator를 사용하면 된다. 

```r
instance = fsi(
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 6),
  measures = msr("classif.auc"),
  terminator = trm("none")
)
```

이제 `fs()`로 정의했던 optimizer의 `$optimize()` 메서드에 인스턴스를 전달하여 RFE를 실행한다.

```r
optimizer$optimize(instance)
```

이렇게 실행한 RFE 과정과 결과에 대한 내용(각 반복에서 선택된 변수 집합, 최적의 변수 집합, 각 반복에 상응하는 추정 성능값 등)은 `instance$archive`에 저장된다.

아래 그림은 60개에서 변수를 하나씩 줄여가며 각 fold에서의 성능(AUC)을 평균내어 계산된 성능 추정값의 경로이다. 처음에는 변수 개수가 감소함에 따라 성능이 증가함을 알 수 있다. 변수가 너무 많이 제거되면 성능이 급격히 하락한다.

> [!note]- code fold
> ```r
> library(viridisLite)
> library(mlr3misc)
> data = as.data.table(instance$archive)
> data[, n := map_int(importance, length)]
> 
> ggplot(data, aes(x = n, y = classif.auc)) +
>   geom_line(color = viridis(n = 1, begin = 0.5), linewidth = 1) +
>   geom_point(fill = viridis(n = 1, begin = 0.5), shape = 21, size = 3, stroke = 0.5, alpha = 0.8) +
>   xlab("Number of Features") +
>   scale_x_reverse() + 
>   theme_minimal()
> ```

![[Pasted image 20240403004636.png]] Figure 2: 변수 개수에 따른 GBM 모델의 성능

각 변수 개수별 중요도 값은 `instance$archive`에 들어있다. 

```r
as.data.table(instance$archive)[, .(features, classif.auc, importance)]
##                       features classif.auc                                                importance
##                         <list>       <num>                                                    <list>
##  1: V1,V10,V11,V12,V13,V14,...   0.8953512 57.83333,57.16667,53.83333,53.00000,52.83333,51.33333,...
##  2: V1,V10,V11,V12,V13,V15,...   0.9033586 58.33333,56.66667,53.83333,52.33333,50.66667,49.33333,...
##  3: V1,V10,V11,V12,V13,V15,...   0.8905669 55.16667,54.33333,51.83333,50.66667,50.16667,50.16667,...
##  4: V1,V10,V11,V12,V13,V15,...   0.9005219 56.33333,56.16667,52.50000,49.83333,49.83333,49.16667,...
##  5: V1,V10,V11,V12,V13,V15,...   0.9022167 54.66667,54.50000,50.33333,49.83333,48.16667,48.00000,...
## ---                                                                                                 
## 56:        V11,V12,V16,V36,V49   0.8711766              4.666667,3.500000,2.666667,2.166667,2.000000
## 57:            V11,V12,V16,V49   0.8530297                       3.833333,2.500000,2.333333,1.333333
## 58:                V11,V12,V49   0.8258255                                2.500000,2.333333,1.166667
## 59:                    V11,V12   0.7966814                                         1.833333,1.166667
## 60:                        V11   0.7723782                                                         1
```

## Support Vector Machine

이번에는 Linear kernel을 사용하는 Linear SVM에 대해 optimal feature set을 선택하는 작업을 해보자. 

```r
learner = lrn("classif.svm",
  type = "C-classification",
  kernel = "linear",
  predict_type = "prob")
```

`SVM Learner`는 기본적으로 변수 중요도를 지원하지 않는다. 해당 모델의 Properties를 확인해보면 `"importance"`가 없다. 

```r
learner$properties
## [1] "multiclass" "twoclass"
```

그렇기 때문에 모델 가중치 벡터를 변수 중요도 값으로 사용해야 한다. `"mlr3fselect.svm_rfe"` callback을 사용하면 RFE optimizer와 함께 Linear SVM을 사용할 수 있다. 이 callback은 모델 내부적으로 `$importance()` 메서드를 learner에 추가한다. 

```r
instance = fsi(
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 6),
  measures = msr("classif.auc"),
  terminator = trm("none"),
  callback = clbk("mlr3fselect.svm_rfe")
)
optimizer$optimize(instance)
```

아래 그림은 변수 개수에 따른 SVM의 각 fold의 평균 성능(AUC)을 나타낸다. 변수 개수가 줄면서 성능이 크게 향상됨을 알 수 있다. 

> [!note]- code fold
> ```r
> data = as.data.table(instance$archive)
> data[, n := map_int(importance, length)]
> 
> ggplot(data, aes(x = n, y = classif.auc)) +
>   geom_line(color = viridis(n = 1, begin = 0.5), linewidth = 1) +
>   geom_point(fill = viridis(n = 1, begin = 0.5), shape = 21, size = 3, stroke = 0.5, alpha = 0.8) +
>   xlab("Number of Features") +
>   scale_x_reverse() + 
>   theme_minimal()
> ```


![[Pasted image 20240403011253.png]] Figure 3: 변수 개수에 따른 SVM의 성능

데이터의 차원이 높을수록, 즉 변수가 많을수록 각 반복마다 변수를 많이 제거하는 것이 효율적이다. 각 반복에서 변수의 25%를 제거해보자. 

```r
optimizer = fs("rfe",
               n_features = 1,
               feature_fraction = 0.75,
               aggregation = "rank"
)
instance = fsi(
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 6),
  measures = msr("classif.auc"),
  terminator = trm("none"),
  callback = clbk("mlr3fselect.svm_rfe")
)
optimizer$optimize(instance)
```

> [!note]- code fold
> ```r
> data = as.data.table(instance$archive)
> data[, n := map_int(importance, length)]
> 
> ggplot(data, aes(x = n, y = classif.auc)) +
>   geom_line(color = viridis(n = 1, begin = 0.5), linewidth = 1) +
>   geom_point(fill = viridis(n = 1, begin = 0.5), shape = 21, size = 3, stroke = 0.5, alpha = 0.8) +
>   xlab("Number of Features") +
>   scale_x_reverse() + 
>   theme_minimal()
> ```

![[Pasted image 20240403011604.png]] Figure 4: 변수 선택의 최적화 경로


## Recursive Feature Elimination with Cross Validation

RFE의 가장 큰 단점은 사용자가 몇 개의 변수를 남겨야 할 지를 직접 정의해야 한다는 것이다. 이를 극복하기 위해 등장한 것이 RFE-CV이다. RFE-CV는 최적 변수 집합을 결정하기 전에 최적의 변수 개수를 먼저 추정한다. 리샘플링 반복에서 각각에서 RFE를 실행하고, 평균 성능이 가장 좋은 변수 개수를 결정한다. 그 다음 최적 변수 개수로 한 번 더 RFE를 수행해 최적 변수 집합을 결정한다.

![[Pasted image 20240403013128.png]] Figure 5: 3-fold CV를 통해 최적의 변수 개수 추정. 각 train-test split마다 RFE가 실행한다(RFE1 ~ RFE3). 평균 성능이 가장 좋은 변수 개수(녹색 사각형)가 최종 변수 개수로 결정된다. 채택된 변수 개수로 전체 데이터셋에 대해 최종 RFE를 수행하여 최적 변수 집합을 결정한다. 

다시 `fs()`를 사용해 RFE-CV optimizer를 정의해보자. RFE optimizer와 유일한 차이점은 집계 기능`aggregation`을 사용하지 않는다는 것이다. 

```r
optimizer = fs("rfecv",
  n_features = 1,
  feature_number = 1)
```

리샘플링 전략으로 6-fold CV를 사용해 총 6번의 RFE를 수행한다.

```r
learner = lrn("classif.svm",
  type = "C-classification",
  kernel = "linear",
  predict_type = "prob")
instance = fsi(
  task = task,
  learner = learner,
  resampling = rsmp("cv", folds = 6),
  measures = msr("classif.auc"),
  terminator = trm("none"),
  callback = clbk("mlr3fselect.svm_rfe")
)
optimizer$optimize(instance)
```

> [!warning] 
> 최적 변수 개수의 성능은 전체 데이터셋에서 계산되며 이를 최종 모델의 성능으로 간주해서는 안 된다. 모델의 최종 성능은 nested resampling으로 추정해야 한다. 

최적의 변수 개수에 대한 최적화 경로를 시각화 해보자. 점들은 각 변수 개수에 대한 리샘플링 반복에서의 평균 성능을 나타낸다. 최고 성능은 24개에서 나타난다. 

> [!note]- code fold
> ```r
> data = as.data.table(instance$archive)[!is.na(iteration), ]
> aggr = data[, list("y" = mean(unlist(.SD))), by = "batch_nr", .SDcols = "classif.auc"]
> aggr[, batch_nr := 61 - batch_nr]
> 
> data[, n := map_int(importance, length)]
> ggplot(aggr, aes(x = batch_nr, y = y)) +
>   geom_line(color = viridis(1, begin = 0.5), linewidth = 1) +
>   geom_point(fill = viridis(1, begin = 0.5), shape = 21, size = 3, stroke = 0.5, alpha = 0.8) +
>   geom_vline(xintercept = aggr[y == max(y)]$batch_nr,
>              colour = viridis(1, begin = 0.33), linetype = 3, linewidth = 1) +
>   xlab("Number of Features") +
>   scale_x_reverse() +
>   theme_minimal()
> ```

![[Pasted image 20240403015658.png]] Figure 6: 최적 변수 개수 추정. 최고 성능은 24개 변수에서 달성된다(파란색 점선).

`instance$archive`를 살펴보면 변수 개수에 따른 리샘플링 반복을 나타내는 `"iteration"` 열이 추가된다. 최종 RFE에 대한 최적 변수 집합은 전체 데이터셋에 대해 평가되기 때문에 `"iteration"`의 값이 `NA`로 되어 있다. 

```r
as.data.table(instance$archive)[, .(features, classif.auc, iteration, importance)]
##                        features classif.auc iteration                                                      importance
##                          <list>       <num>     <int>                                                          <list>
##   1: V1,V10,V11,V12,V13,V14,...   0.9210526         1 2.9310844,1.4694181,1.4393676,1.3544473,1.1578352,0.8796203,...
##   2: V1,V10,V11,V12,V13,V14,...   0.9100000         2 1.5944918,1.2786595,1.2385845,1.1794179,0.8988513,0.8918989,...
##   3: V1,V10,V11,V12,V13,V14,...   0.7993197         3       2.462275,1.685524,1.493480,1.225240,1.133875,1.085772,...
##   4: V1,V10,V11,V12,V13,V14,...   0.7622378         4 3.0390728,2.0786068,1.7582644,1.1908765,1.1349204,0.7851106,...
##   5: V1,V10,V11,V12,V13,V14,...   0.8437500         5 2.6391709,2.0938501,0.9838008,0.9651096,0.8443965,0.8110389,...
##  ---                                                                                                                 
## 393: V1,V11,V12,V16,V19,V23,...   0.9654500        NA 3.1230459,1.3255368,1.0918174,0.8910188,0.7794044,0.7598469,...
## 394: V1,V11,V12,V16,V19,V23,...   0.9669360        NA 3.0863545,1.2918273,1.1824120,0.9695328,0.7858436,0.7786829,...
## 395:  V1,V11,V12,V16,V23,V3,...   0.9671218        NA 2.5586015,0.9979432,0.9350143,0.8480575,0.7909376,0.7522515,...
## 396:  V1,V11,V12,V16,V23,V3,...   0.9664716        NA 2.1931049,1.0844848,1.0180228,0.9876487,0.8248918,0.7561749,...
## 397:  V1,V11,V12,V16,V23,V3,...   0.9638711        NA 2.1450328,1.3479311,1.0535960,0.7576391,0.6950540,0.5852394,...
```

## Final Model

최종 모델은 최적의 변수 집합으로 전체 데이터셋에 학습한다. 최적 변수 집합은 `instance$result_feature_set`에 저장되어 있다. 

```r
task$select(instance$result_feature_set)
learner$train(task)
```

이제 학습된 모델을 사용해 새로운 외부 데이터를 예측할 수 있다.


- https://mlr-org.com/gallery/optimization/2023-02-07-recursive-feature-elimination/

