---
tags:
  - Kaggle
  - mlr3
  - XGBoost
date: 2024-04-08
---
![[Pasted image 20240409182332.png]]



# 개요

- 목표: 고객이 계좌를 유지할지 해지할지(churns)를 예측
- 평가: AUROC
- 합성 데이터셋(Synthetically-Generated Datasets)
	- 원본 데이터에서 합성된 데이터임

# 데이터 설명

- 원본 데이터를 모델 학습에 포함했을 시 모델 성능이 향상되는지 확인

은행 고객 이탈(bank customer churn) 데이터에는 고객에 대한 정보가 들어있다.

- Train data: 

| Column            | Type    | # of Unique | Description                                 |
|:----------------- |:------- |:----------- |:------------------------------------------- |
| `id`              | `chr`   | 165034      | 행 번호                                     |
| `CustomerID`      | `chr`   | 23221       | 고객 식별 ID                                |
| `Surname`         | `chr`   | 2797        | 성(last name)                               |
| `CreditScore`     | `num`   | 457         | 고객의 신용점수                             |
| `Geography`       | `fctr`  | 3           | 고객이 거주하는 국가(프랑스, 스페인, 독일)  |
| `Gender`          | `fctr`` | 2           | 고객의 성별(남성, 여성)                     |
| `Age`             | `num`   | 71          | 고객의 나이                                 |
| `Tenure`          | `fctr`  | 11          | 고객이 은행에 가입한 기간                   |
| `Balance`         | `num`   | 30075       | 고객의 계좌 잔액                            |
| `NumOfProducts`   | `fctr`  | 4           | 고객이 이용 중인 은행 상품 수               |
| `HasCrCard`       | `fctr`  | 2           | 고객의 신용카드 소지 여부(1: 예, 0: 아니오) |
| `IsActiveMember`  | `fctr`  | 2           | 활생 고객 여부(1: 예, 0: 아니오)            |
| `EstimatedSalary` | `num`   | 55298       | 고객 추정 급여                              |
| `Exited`          | `fctr`  | 2           | 고객 이탈 여부(1: 예, 0: 아니오)            |

# Package Load

```r
# Packages Load
suppressWarnings(
  suppressMessages({
    library(tidyverse)
    library(mlr3verse)
    library(mlr3mbo)
    library(mlr3tuningspaces)
    library(mlr3tuning)
    library(bbotk)
    library(data.table)
    library(formattable)
    library(patchwork)
    library(ggpubr)
    library(cowplot)
    library(rcompanion)
    library(cvms)
    library(tictoc)
    library(DiagrammeR)
    library(iml)
    library(DALEX)
	library(DALEXtra)
  })
)
```

# 1. Data Exploration

이 대회에서 사용되는 데이터는 [Bank Customer Churn Prediction dataset](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction)으로 학습시킨 딥러닝 모델로 생성한 합성 데이터이다. 변수들의 분포는 원본(`original`) 데이터와 비슷하지만 완전히 동일하지는 않다. 

```r
set.seed(123)

train = fread("../data/playground-series-s4e1/train.csv")
test = fread("../data/playground-series-s4e1/test.csv")
submission = fread("../data/playground-series-s4e1/sample_submission.csv")
original = fread("../data/playground-series-s4e1/Churn_Modelling.csv")

sprintf("Dimension of the train dataset: (%s, %s)", nrow(train), ncol(train))
sprintf("Dimension of the test dataset: (%s, %s)", nrow(test), ncol(test))
sprintf("Dimension of the original dataset: (%s, %s)", nrow(original), ncol(original))
## [1] "Dimension of the train dataset: (165034, 14)"
## [1] "Dimension of the test dataset: (110023, 13)"
## [1] "Dimension of the original dataset: (10002, 14)"

original = original %>% 
  mutate(RowNumber = RowNumber - 1) %>% 
  rename(id = RowNumber)

train = train %>% 
  mutate(across(c(id, CustomerId), as.character)) %>% 
  mutate(across(c(Geography, Gender, Tenure, NumOfProducts, HasCrCard, IsActiveMember, Exited), as.factor)) %>% 
  mutate(CreditScore = as.numeric(CreditScore))
test = test %>% 
  mutate(across(c(id, CustomerId), as.character)) %>% 
  mutate(across(c(Geography, Gender, Tenure, NumOfProducts, HasCrCard, IsActiveMember), as.factor)) %>% 
  mutate(CreditScore = as.numeric(CreditScore))
original = original %>% 
  mutate(across(c(id, CustomerId), as.character)) %>% 
  mutate(across(c(Geography, Gender, Tenure, NumOfProducts, HasCrCard, IsActiveMember, Exited), as.factor)) %>% 
  mutate(CreditScore = as.numeric(CreditScore))
```

## 1.1 Train Data

```r
head(train)
```

| id  | CustomerId | Surname        | CreditScore | Geography | Gender | Age | Tenure | Balance  | NumOfProducts | HasCrCard | IsActiveMember | EstimatedSalary | Exited |
| --- | ---------- | -------------- | ----------- | --------- | ------ | --- | ------ | -------- | ------------- | --------- | -------------- | --------------- | ------ |
| 0   | 15674932   | Okwudilichukwu | 668         | France    | Male   | 33  | 3      | 0.0      | 2             | 1         | 0              | 181449.97       | 0      |
| 1   | 15749177   | Okwudiliolisa  | 627         | France    | Male   | 33  | 1      | 0.0      | 2             | 1         | 1              | 49503.50        | 0      |
| 2   | 15694510   | Hsueh          | 678         | France    | Male   | 40  | 10     | 0.0      | 2             | 1         | 0              | 184866.69       | 0      |
| 3   | 15741417   | Kao            | 581         | France    | Male   | 34  | 2      | 148882.5 | 1             | 1         | 1              | 84560.88        | 0      |
| 4   | 15766172   | Chiemenam      | 716         | Spain     | Male   | 33  | 5      | 0.0      | 2             | 1         | 1              | 15068.83        | 0      |
| 5   | 15771669   | Genovese       | 588         | Germany   | Male   | 36  | 4      | 131778.6 | 1             | 1         | 0              | 136024.31       | 1      |

```r
train %>% select_if(is.numeric) %>% 
  pivot_longer(everything()) %>% 
  group_by(name) %>% 
  reframe(
    count = n(),
    Mean = mean(value),
    SD = sd(value),
    Min = min(value),
    `25%` = quantile(value, 0.25),
    `50%` = quantile(value, 0.50),
    `75%` = quantile(value, 0.75),
    Max = max(value)
  )
```

|name|count|Mean|SD|Min|25%|50%|75%|Max|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Age|165034|38.12589|8.867205|18.00|32.00|37|42.0|92.0|
|Balance|165034|55478.08669|62817.663278|0.00|0.00|0|119939.5|250898.1|
|CreditScore|165034|656.45437|80.103340|350.00|597.00|659|710.0|850.0|
|EstimatedSalary|165034|112574.82273|50292.865585|11.58|74637.57|117948|155152.5|199992.5|

```r
train %>% 
  select(where(is.character), where(is.factor)) %>% 
  pivot_longer(everything()) %>% 
  group_by(name) %>% 
  reframe(
    n_unique = n_distinct(value)
  ) 
```

| name           | n_unique |
| -------------- | -------: |
| CustomerId     |    23221 |
| Exited         |        2 |
| Gender         |        2 |
| Geography      |        3 |
| HasCrCard      |        2 |
| IsActiveMember |        2 |
| NumOfProducts  |        4 |
| Surname        |     2797 |
| Tenure         |       11 |
| id             |   165034 |

## 1.2 Test Data

> [!note]- code fold
> ```r
> head(test)
> ```

|id|CustomerId|Surname|CreditScore|Geography|Gender|Age|Tenure|Balance|NumOfProducts|HasCrCard|IsActiveMember|EstimatedSalary|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|165034|15773898|Lucchese|586|France|Female|23|2|0.0|2|0|1|160976.75|
|165035|15782418|Nott|683|France|Female|46|2|0.0|1|1|0|72549.27|
|165036|15807120|K?|656|France|Female|34|7|0.0|2|1|0|138882.09|
|165037|15808905|O'Donnell|681|France|Male|36|8|0.0|1|1|0|113931.57|
|165038|15607314|Higgins|752|Germany|Male|38|10|121263.6|1|1|0|139431.00|
|165039|15672704|Pearson|593|France|Female|22|9|0.0|2|0|0|51907.72|
> [!note]- code fold
> ```r
> test %>% select_if(is.numeric) %>% 
>   pivot_longer(everything()) %>% 
>   group_by(name) %>% 
>   reframe(
>     count = n(),
>     Mean = mean(value),
>     SD = sd(value),
>     Min = min(value),
>     `25%` = quantile(value, 0.25),
>     `50%` = quantile(value, 0.50),
>     `75%` = quantile(value, 0.75),
>     Max = max(value)
>   )
> ```

| name            | count  | Mean         | SD          | Min    | 25%      | 50%      | 75%      | Max      |
| --------------- | ------: | ------------: | -----------: | ------: | --------: | --------: | --------: | --------: |
| Age             | 110023 | 38.12221     | 8.86155     | 18.00  | 32.00    | 37.0     | 42.0     | 92.0     |
| Balance         | 110023 | 55333.61135  | 62788.51968 | 0.00   | 0.00     | 0.0      | 120145.6 | 250898.1 |
| CreditScore     | 110023 | 656.53079    | 80.31541    | 350.00 | 597.00   | 660.0    | 710.0    | 850.0    |
| EstimatedSalary | 110023 | 112315.14777 | 50277.04824 | 11.58  | 74440.33 | 117832.2 | 154631.4 | 199992.5 |

> [!note]- code fold
> ```r
> test %>% 
>   select(where(is.character), where(is.factor)) %>% 
>   pivot_longer(everything()) %>% 
>   group_by(name) %>% 
>   reframe(
>     n_unique = n_distinct(value)
>   ) 
> ```

| name           | n_unique |
| -------------- | --------: |
| CustomerId     | 19698    |
| Gender         | 2        |
| Geography      | 3        |
| HasCrCard      | 2        |
| IsActiveMember | 2        |
| NumOfProducts  | 4        |
| Surname        | 2708     |
| Tenure         | 11       |
| id             | 110023   |

## 1.3 Original Data

```r
head(original)
```

| id  | CustomerId | Surname  | CreditScore | Geography | Gender | Age | Tenure | Balance   | NumOfProducts | HasCrCard | IsActiveMember | EstimatedSalary | Exited |
| ---: | ----------: | --------: | -----------: | ---------: | ------: | ---: | ------: | ---------: | -------------: | ---------: | --------------: | ---------------: | ------: |
| 0   | 15634602   | Hargrave | 619         | France    | Female | 42  | 2      | 0.00      | 1             | 1         | 1              | 101348.88       | 1      |
| 1   | 15647311   | Hill     | 608         | Spain     | Female | 41  | 1      | 83807.86  | 1             | 0         | 1              | 112542.58       | 0      |
| 2   | 15619304   | Onio     | 502         | France    | Female | 42  | 8      | 159660.80 | 3             | 1         | 0              | 113931.57       | 1      |
| 3   | 15701354   | Boni     | 699         | France    | Female | 39  | 1      | 0.00      | 2             | 0         | 0              | 93826.63        | 0      |
| 4   | 15737888   | Mitchell | 850         | Spain     | Female | 43  | 2      | 125510.82 | 1             | NA        | 1              | 79084.10        | 0      |
| 5   | 15574012   | Chu      | 645         | Spain     | Male   | 44  | 8      | 113755.78 | 2             | 1         | 0              | 149756.71       | 1      |


> [!note]- code fold
> ```r
> original %>% select_if(is.numeric) %>% 
>   pivot_longer(everything()) %>% 
>   group_by(name) %>% 
>   reframe(
>     count = n(),
>     Mean = mean(value, na.rm = T),
>     SD = sd(value, na.rm = T),
>     Min = min(value, na.rm = T),
>     `25%` = quantile(value, 0.25, na.rm = T),
>     `50%` = quantile(value, 0.50, na.rm = T),
>     `75%` = quantile(value, 0.75, na.rm = T),
>     Max = max(value, na.rm = T)
>   )
> ```

| name            | count | Mean         | SD          | Min    | 25%      | 50%       | 75%      | Max      |
| --------------- | -----: | ------------: | -----------: | ------: | --------: | ---------: | --------: | --------: |
| Age             | 10002 | 38.92231     | 10.48720    | 18.00  | 32.00    | 37.00     | 44.0     | 92.0     |
| Balance         | 10002 | 76491.11288  | 62393.47414 | 0.00   | 0.00     | 97198.54  | 127647.8 | 250898.1 |
| CreditScore     | 10002 | 650.55509    | 96.66162    | 350.00 | 584.00   | 652.00    | 718.0    | 850.0    |
| EstimatedSalary | 10002 | 100083.33114 | 57508.11780 | 11.58  | 50983.75 | 100185.24 | 149383.7 | 199992.5 |

> [!note]- code fold
> ```r
> original %>% 
>   select(where(is.character), where(is.factor)) %>% 
>   pivot_longer(everything()) %>% 
>   group_by(name) %>% 
>   reframe(
>     n_unique = n_distinct(value)
>   ) 
> ```

| name           | n_unique |
| -------------- | --------:|
| CustomerId     |    10000 |
| Exited         |        2 |
| Gender         |        2 |
| Geography      |        4 |
| HasCrCard      |        3 |
| IsActiveMember |        3 |
| NumOfProducts  |        4 |
| Surname        |     2932 |
| Tenure         |       11 |
| id             |    10000 |

# 2. EDA


## 2.1 결측치 확인

```r
train %>% is.na() %>% colSums()
##              id      CustomerId         Surname     CreditScore 
##               0               0               0               0 
##       Geography          Gender             Age          Tenure 
##               0               0               0               0 
##         Balance   NumOfProducts       HasCrCard  IsActiveMember 
##               0               0               0               0 
## EstimatedSalary          Exited 
##               0               0

test %>% is.na() %>% colSums()
##              id      CustomerId         Surname     CreditScore 
## t              0               0               0               0 
##       Geography          Gender             Age          Tenure 
##               0               0               0               0 
##         Balance   NumOfProducts       HasCrCard  IsActiveMember 
##               0               0               0               0 
## EstimatedSalary 
##               0
```

- 주어진 데이터에는 결측치가 없다. 

## 2.2 Target 변수 분석

> [!note]- code fold
> ```r
> p1 = train %>% 
>   count(Exited) %>% 
>   mutate(prop = n/sum(n)) %>% 
>   ggplot(aes(x = '', y = prop, fill = Exited)) +
>   geom_col(color = 'black') +
>   geom_text(aes(label = scales::percent(prop, suffix = '%', accuracy = 0.1)),
>             position = position_stack(vjust = 0.5)) +
>   coord_polar(theta = 'y') +
>   theme_void() +
>   theme(legend.position = 'none')
> 
> p2 = train %>% 
>   count(Exited) %>% 
>   mutate(prop = n/sum(n)) %>% 
>   ggplot(aes(x = Exited, y = n, fill = Exited)) +
>   geom_col(color = 'black') +
>   scale_y_continuous(name = "count", breaks = seq(0, 120000, 20000)) +
>   theme_minimal()
> (p1 + p2) +
>   plot_annotation(
>     title = "Target Value Analysis - Competition Data",
>     theme = theme(plot.title = element_text(hjust = 0.5))
>   )
> ```

![[Pasted image 20240408224456.png]]

> [!note]- code fold
> ```r
> p1 = original %>% 
>   count(Exited) %>% 
>   mutate(prop = n/sum(n)) %>% 
>   ggplot(aes(x = '', y = prop, fill = Exited)) +
>   geom_col(color = 'black') +
>   geom_text(aes(label = scales::percent(prop, suffix = '%', accuracy = 0.1)),
>             position = position_stack(vjust = 0.5)) +
>   coord_polar(theta = 'y') +
>   theme_void() +
>   theme(legend.position = 'none')
> 
> p2 = original %>% 
>   count(Exited) %>% 
>   mutate(prop = n/sum(n)) %>% 
>   ggplot(aes(x = Exited, y = n, fill = Exited)) +
>   geom_col(color = 'black') +
>   scale_y_continuous(name = "count",
> 				   breaks = seq(0, 8000, 2000)) +
>   theme_minimal()
> (p1 + p2) +
>   plot_annotation(
>     title = "Target Value Analysis - Original Data",
>     theme = theme(plot.title = element_text(hjust = 0.5))
>   )
> ```

![[Pasted image 20240408224303.png]]
- `Exited`과 `Not Exited`의 클래스 분포는 원본과 대회 데이터가 거의 비슷함
- 8:2 정도의 클래스 불균형이 있음
- 실제로는 `Exited`인 고객들에게 관심을 갖고 이들의 패턴을 분석하고자 함


## 2.3 범주형 변수 분석

- 범주형 변수: `Geography`, `Gender`, `Tenure`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`

> [!note]- code fold
> ```r
> # Geography
> p1 = train %>% 
>   count(Geography) %>% 
>   mutate(prop = n / sum(n)) %>% 
>   ggplot(aes(x = '', y = prop, fill = Geography)) +
>   geom_col(color = 'black') +
>   geom_text(aes(label = scales::percent(prop, suffix = '%', accuracy = 0.1)),
>                 position = position_stack(vjust = 0.5)) +
>   coord_polar(theta = 'y') +
>   theme_void() +
>   theme(legend.position = "none")
> p2 = train %>% 
>   count(Geography) %>% 
>   mutate(prop = n / sum(n)) %>% 
>   ggplot(aes(x = Geography, y = n, fill = Geography)) +
>   geom_col(color = 'black') +
>   scale_y_continuous(name = "count", 
>                      breaks = seq(0, 80000, 20000)) +
>   theme_minimal() +
>   theme(legend.position = "left")
> 
> (p1 + p2) +
>   plot_annotation(
>     title = "Geography",
>     theme = theme(plot.title = element_text(hjust = 0.45))
>   )
> 
> # Gender
> p1 = train %>%
>   count(Gender) %>%
>   mutate(prop = n / sum(n)) %>%
>   ggplot(aes(x = '', y = prop, fill = Gender)) +
>   geom_col(color = 'black') +
>   geom_text(aes(label = scales::percent(prop, suffix = '%', accuracy = 0.1)),
>             position = position_stack(vjust = 0.5)) +
>   coord_polar(theta = 'y', start = pi/2) +
>   theme_void() +
>   theme(legend.position = "none")
> p2 = train %>%
>   count(Gender) %>%
>   mutate(prop = n / sum(n)) %>%
>   ggplot(aes(x = Gender, y = n, fill = Gender)) +
>   geom_col(color = 'black') +
>   scale_y_continuous(name = "count",
>                      breaks = seq(0, 80000, 20000)) +
>   theme_minimal() +
>   theme(legend.position = "left")
> (p1 + p2) +
>   plot_annotation(
>     title = "Gender",
>     theme = theme(plot.title = element_text(hjust = 0.45))
>   )
> 
> # Tenure
> p1 = train %>% 
>   count(Tenure) %>% 
>   mutate(prop = n / sum(n)) %>% 
>   ggplot(aes(x = '', y = prop, fill = Tenure)) +
>   geom_col(color = 'black') +
>   geom_text(aes(label = scales::percent(prop, suffix = '%', accuracy = 0.1)),
>             position = position_stack(vjust = 0.5),
>             size = 3.5) +
>   coord_polar(theta = 'y') +
>   theme_void() +
>   theme(legend.position = "none")
> p2 = train %>% 
>   count(Tenure) %>% 
>   mutate(prop = n / sum(n)) %>% 
>   ggplot(aes(x = Tenure, y = n, fill = Tenure)) +
>   geom_col(color = 'black') +
>   scale_y_continuous(name = "count", 
>                      breaks = seq(0, 17500, 2500)) +
>   theme_minimal() +
>   theme(legend.position = "left")
> (p1 + p2) +
>   plot_annotation(
>     title = "Tenure",
>     theme = theme(plot.title = element_text(hjust = 0.45))
>   )
> 
> # NumOfProducts
> p1 = train %>% 
>   count(NumOfProducts) %>% 
>   mutate(prop = n / sum(n)) %>% 
>   ggplot(aes(x = '', y = prop, fill = NumOfProducts)) +
>   geom_col(color = 'black') +
>   geom_text(aes(label = scales::percent(prop, suffix = '%', accuracy = 0.1)),
>             position = position_stack(vjust = 0.5),
>             size = 3.5) +
>   coord_polar(theta = 'y', start = pi/2) +
>   theme_void() +
>   theme(legend.position = "none")
> p2 = train %>% 
>   count(NumOfProducts) %>% 
>   mutate(prop = n / sum(n)) %>% 
>   ggplot(aes(x = NumOfProducts, y = n, fill = NumOfProducts)) +
>   geom_col(color = 'black') +
>   scale_y_continuous(name = "count", 
>                      breaks = seq(0, 80000, 10000)) +
>   theme_minimal() +
>   theme(legend.position = "left")
> (p1 + p2) +
>   plot_annotation(
>     title = "NumOfProducts",
>     theme = theme(plot.title = element_text(hjust = 0.45))
>   )
> 
> # HasCrCard
> p1 = train %>% 
>   count(HasCrCard) %>% 
>   mutate(prop = n / sum(n)) %>% 
>   ggplot(aes(x = '', y = prop, fill = HasCrCard)) +
>   geom_col(color = 'black') +
>   geom_text(aes(label = scales::percent(prop, suffix = '%', accuracy = 0.1)),
>             position = position_stack(vjust = 0.5),
>             size = 3.5) +
>   coord_polar(theta = 'y', start = pi/2) +
>   theme_void() +
>   theme(legend.position = "none")
> p2 = train %>% 
>   count(HasCrCard) %>% 
>   mutate(prop = n / sum(n)) %>% 
>   ggplot(aes(x = HasCrCard, y = n, fill = HasCrCard)) +
>   geom_col(color = 'black') +
>   scale_y_continuous(name = "count", 
>                      breaks = seq(0, 120000, 20000)) +
>   theme_minimal() +
>   theme(legend.position = "left")
> (p1 + p2) +
>   plot_annotation(
>     title = "HasCrCard",
>     theme = theme(plot.title = element_text(hjust = 0.45))
>   )
> 
> # IsActiveMember
> p1 = train %>% 
>   count(IsActiveMember) %>% 
>   mutate(prop = n / sum(n)) %>% 
>   ggplot(aes(x = '', y = prop, fill = IsActiveMember)) +
>   geom_col(color = 'black') +
>   geom_text(aes(label = scales::percent(prop, suffix = '%', accuracy = 0.1)),
>             position = position_stack(vjust = 0.5),
>             size = 3.5) +
>   coord_polar(theta = 'y', start = pi/2) +
>   theme_void() +
>   theme(legend.position = "none")
> p2 = train %>% 
>   count(IsActiveMember) %>% 
>   mutate(prop = n / sum(n)) %>% 
>   ggplot(aes(x = IsActiveMember, y = n, fill = IsActiveMember)) +
>   geom_col(color = 'black') +
>   scale_y_continuous(name = "count", 
>                      breaks = seq(0, 80000, 10000)) +
>   theme_minimal() +
>   theme(legend.position = "left")
> (p1 + p2) +
>   plot_annotation(
>     title = "IsActiveMember",
>     theme = theme(plot.title = element_text(hjust = 0.45))
>   )
> ```

![[Pasted image 20240408234906.png]]

![[Pasted image 20240408234934.png]]

![[Pasted image 20240408234952.png]]

![[Pasted image 20240408235008.png]]

![[Pasted image 20240408235023.png]]

![[Pasted image 20240408235036.png]]


- `IsActiveMember`, `Tenure`, `Gender`는 각 범주가 균등하게 분포함
- `HasCrCard`, `NumOfProducts`, `Geography`는 범주가 균등하지 않게 분포함
- `Tenure`는 연속형/순서형 변수로 간주할 수 있지만, 여기서는 level이 11개인 범주형 변수로 취급

## 2.4 수치형 변수 분석

- 수치형 변수: `CreditScore`, `Age`, `Balance`, `EstimatedSalary`

> [!note]- code fold
> ```r
> train %>%
>   ggplot(aes(x = CreditScore, y = ..count.., fill = Exited)) +
>   geom_histogram(bins = 50, alpha = 0.8, color = 'white', position = 'identity') +
>   scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
>   scale_y_continuous(breaks = seq(0, 8000, 1000)) +
>   theme_minimal()
> train %>%
>   ggplot(aes(x = Age, y = ..count.., fill = Exited)) +
>   geom_histogram(bins = 50, alpha = 0.8, color = 'white', position = 'identity') +
>   scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
>   scale_y_continuous(breaks = seq(0, 16000, 2000)) +
>   scale_x_continuous(breaks = seq(20, 90, 10)) +
>   theme_minimal()
> train %>%
>   ggplot(aes(x = Balance, y = ..count.., fill = Exited)) +
>   geom_histogram(bins = 50, alpha = 0.8, color = 'white', position = 'identity') +
>   scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
>   scale_y_continuous(breaks = seq(0, 70000, 10000)) +
>   scale_x_continuous(breaks = seq(0, 250000, 50000)) +
>   theme_minimal()
> train %>%
>   ggplot(aes(x = EstimatedSalary, y = ..count.., fill = Exited)) +
>   geom_histogram(bins = 50, alpha = 0.8, color = 'white', position = 'identity') +
>   scale_fill_manual(values = c("#00AFBB", "#E7B800")) +
>   scale_y_continuous(breaks = seq(0, 5000, 1000)) +
>   scale_x_continuous(breaks = seq(0, 200000, 25000)) +
>   theme_minimal()
> ```

![[Pasted image 20240409012212.png]]

![[Pasted image 20240409012219.png]]

![[Pasted image 20240409012225.png]]

![[Pasted image 20240409012232.png]]

- `Balance` 변수는 대부분이 0으로 구성되고 right-skewed 되어 있음
- 다른 수치형 변수들도 약간씩 분포가 치우쳐져 있음
- `Exited` 클래스별 수치형 변수의 분포는 거의 비슷함

## 2.5 상관관계

- 수치형 변수들 간 관계: 상관관계의 절대값
- 수치형 변수 vs 범주형 변수: 수치형 변수에 대해 범주형 변수를 ANOVA 모형에 적합한 후 결정계수 $R^2$의 제곱근으로 측정
- 범주형 변수들 간 관계: Cramér's V coefficient를 통해 파악

> [!check] Cramér's V coefficient
> - 범주형 변수 간 연관성을 측정하는 통계량 
> - 카이제곱 통계량에 기반을 둔 방법
> - 0~1 사이의 값을 갖고, 1에 가까울수록 연관되어 있음 $$V = \sqrt{\frac{\chi^2}{n(q-1)}}$$ 여기서 $\chi^2 = \sum\frac{(O_{ij}-E_{ij})^2}{E_{ij}}$ ($O_{ij}$: 관찰 빈도, $E_{ij}$: 기대 빈도), $q$: 두 범주형 변수의 각 범주 개수 중 최소값 i.e., $\min(r, c)$ 
> - `rcompanion::cramerV()`로 구현

변수들 간 연관성을 시각화해 본 결과 target `Exited`과 강한 연관성을 가진 변수는 보이지 않고, feature들 간에도 연관성이 전반적으로 약한 것으로 보인다.  

- `Exited`는 `NumOfProducts`와 가장 강한 연관성을 보이고, 그 다음으로 `Geography`, `IsActiveMember` 순서로 연관되어 있음

> [!note]- code fold
> ```r
> numeric_vars = train %>% select_if(is.numeric) %>% names()
> category_vars = train %>% select_if(is.factor) %>% names()
> var_names = c("Exited", numeric_vars, category_vars[-7])
> combs = expand.grid(y = var_names, x = var_names)
> combs = combs %>% 
>   mutate(mode = case_when(y %in% numeric_vars & x %in% numeric_vars   ~ '1',  # 1: 수치형 vs 수치형
>                           y %in% numeric_vars & x %in% category_vars  ~ '2',  # 2: 수치형 vs 범주형
>                           y %in% category_vars & x %in% numeric_vars  ~ '3',  # 3: 수치형 vs 범주형 >> 순서 바꿔야 함
>                           y %in% category_vars & x %in% category_vars ~ '4')) # 4: 범주형 vs 범주형
> my_cor = function(cnames, data) {
>   y = data %>% pull(cnames[1])
>   x = data %>% pull(cnames[2])
>   
>   if (cnames[3] %in% c('1', '2', '3')) {
>     if (cnames[3] == '3') {
>       y = data %>% pull(cnames[2])
>       x = data %>% pull(cnames[1])
>     }
>     suppressWarnings({
>       av = anova(lm(y ~ x))
>     })
>     return(av[[2]][1] / sum(av[[2]]))
>   } else {
>     return(cramerV(x = x, y = y))
>   }
> }
> combs$cor = apply(combs, 1, my_cor, data = train)
> combs[["lab"]] = sprintf("%.4f", combs$cor)
> combs = combs %>% 
>   mutate(x = factor(x, levels = var_names),
>          y = factor(y, levels = rev(var_names)))
> combs %>% 
>   ggplot(aes(x = x, y = y, fill = cor, label = lab)) +
>   geom_tile(color = 'white', width = 0.95, height = 0.95) +
>   geom_label(fill = 'white', size = 3) +
>   viridis::scale_fill_viridis(name = "Relationship", begin = 0.25) +
>   theme_minimal() +
>   theme(axis.text.x = element_text(angle = 45, hjust = 1),
>         axis.title = element_blank())
> ```

![[Pasted image 20240409014240.png]]


# 3. Modeling

- 사용할 모델: XGBoost

## 3.1 데이터 준비

- `id`, `CustomerId`, `Surname`은 별다른 정보를 제공하지 않으므로 모델 학습에 사용하지 않는다. 

```r
train = train %>% select(-c(id, CustomerId, Surname))
test = test %>% select(-c(id, CustomerId, Surname))
original = original %>% select(-c(id, CustomerId, Surname))

task = as_task_classif(train, target = "Exited", id = "train_data")
task$positive = "1"
task
## <TaskClassif:train_data> (165034 x 11)
## * Target: Exited
## * Properties: twoclass
## * Features (10):
##   - fct (5): Gender, Geography, HasCrCard,
##     IsActiveMember, NumOfProducts
##   - dbl (4): Age, Balance, CreditScore,
##     EstimatedSalary
##   - int (1): Tenure

task_test = as_task_classif(
  test %>% mutate(Exited = factor(NA, levels = 0:1)),
  target = "Exited", id = "test_data"
)
task_test$positive = "1"
task_test
## <TaskClassif:test_data> (110023 x 11)
## * Target: Exited
## * Properties: twoclass
## * Features (10):
##   - fct (5): Gender, Geography, HasCrCard,
##     IsActiveMember, NumOfProducts
##   - dbl (4): Age, Balance, CreditScore,
##     EstimatedSalary
##   - int (1): Tenure
```

- Train/Test Data에 모두 결측치가 존재하지 않으므로 Imputation은 필요하지 않다.

## 3.2 범주형 변수 인코딩

- One Hot Encoder

```r
graph = po("removeconstants", id = "removeconstants_preencoding") %>>% 
  po("collapsefactors", no_collapse_above_prevalence = 0.01) %>>%
  po("encode", method = "one-hot", id = 'low_cardinality_encode') %>>% 
  po("removeconstants", id = "removeconstants_postencoding")
graph$plot()
```

![[Pasted image 20240409025700.png|300]]


```r
task_encoded = graph$train(task)[[1]]
task_test_encoded = graph$predict(task_test)[[1]]
task_encoded$head() 
```

| Exited | Age | Balance  | CreditScore | EstimatedSalary | Gender.Female | Gender.Male | Geography.France | Geography.Germany | Geography.Spain | HasCrCard.0 | HasCrCard.1 | IsActiveMember.0 | IsActiveMember.1 | NumOfProducts.1 | NumOfProducts.2 | NumOfProducts.3 | Tenure.0 | Tenure.1 | Tenure.2 | Tenure.3 | Tenure.4 | Tenure.5 | Tenure.6 | Tenure.7 | Tenure.8 | Tenure.9 | Tenure.10 |
| ------ | --- | -------- | ----------- | --------------- | ------------- | ----------- | ---------------- | ----------------- | --------------- | ----------- | ----------- | ---------------- | ---------------- | --------------- | --------------- | --------------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | --------- |
| 0      | 33  | 0.0      | 668         | 181449.97       | 0             | 1           | 1                | 0                 | 0               | 0           | 1           | 1                | 0                | 0               | 1               | 0               | 0        | 0        | 0        | 1        | 0        | 0        | 0        | 0        | 0        | 0        | 0         |
| 0      | 33  | 0.0      | 627         | 49503.50        | 0             | 1           | 1                | 0                 | 0               | 0           | 1           | 0                | 1                | 0               | 1               | 0               | 0        | 1        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0         |
| 0      | 34  | 148882.5 | 581         | 84560.88        | 0             | 1           | 1                | 0                 | 0               | 0           | 1           | 0                | 1                | 1               | 0               | 0               | 0        | 0        | 1        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0         |
| 0      | 33  | 0.0      | 716         | 15068.83        | 0             | 1           | 0                | 0                 | 1               | 0           | 1           | 0                | 1                | 0               | 1               | 0               | 0        | 0        | 0        | 0        | 0        | 1        | 0        | 0        | 0        | 0        | 0         |
| 1      | 36  | 131778.6 | 588         | 136024.31       | 0             | 1           | 0                | 1                 | 0               | 0           | 1           | 1                | 0                | 1               | 0               | 0               | 0        | 0        | 0        | 0        | 1        | 0        | 0        | 0        | 0        | 0        | 0         |
| 0      | 30  | 144772.7 | 593         | 29792.11        | 1             | 0           | 1                | 0                 | 0               | 0           | 1           | 1                | 0                | 1               | 0               | 0               | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 0        | 1        | 0        | 0         |

```r
task_test_encoded$head()
```

|Exited|Age|Balance|CreditScore|EstimatedSalary|Gender.Female|Gender.Male|Geography.France|Geography.Germany|Geography.Spain|HasCrCard.0|HasCrCard.1|IsActiveMember.0|IsActiveMember.1|NumOfProducts.1|NumOfProducts.2|NumOfProducts.3|Tenure.0|Tenure.1|Tenure.2|Tenure.3|Tenure.4|Tenure.5|Tenure.6|Tenure.7|Tenure.8|Tenure.9|Tenure.10|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|NA|23|0.0|586|160976.75|1|0|1|0|0|1|0|0|1|0|1|0|0|0|1|0|0|0|0|0|0|0|0|
|NA|46|0.0|683|72549.27|1|0|1|0|0|0|1|1|0|1|0|0|0|0|1|0|0|0|0|0|0|0|0|
|NA|34|0.0|656|138882.09|1|0|1|0|0|0|1|1|0|0|1|0|0|0|0|0|0|0|0|1|0|0|0|
|NA|36|0.0|681|113931.57|0|1|1|0|0|0|1|1|0|1|0|0|0|0|0|0|0|0|0|0|1|0|0|
|NA|38|121263.6|752|139431.00|0|1|0|1|0|0|1|1|0|1|0|0|0|0|0|0|0|0|0|0|0|0|1|
|NA|22|0.0|593|51907.72|1|0|1|0|0|1|0|1|0|0|1|0|0|0|0|0|0|0|0|0|0|1|0|


## 3.3 Baseline Model

Hyperparameter Tuning & Cross-Validation 이전에 기본 세팅으로 XGBoost 모형을 적합하여 성능을 측정해 보자. 

```r
xgb_clf = lrn("classif.xgboost", predict_type = "prob",
              objective = "binary:logistic",
              verbose = 1,
              eval_metric = "auc",
              nrounds = 1000,
              early_stopping_rounds = 10,
              early_stopping_set = "test")
xgb_clf$train(task_encoded)
## [1]	train-auc:0.875628	test-auc:0.871950 
## Multiple eval metrics are present. Will use test_auc for early stopping.
## Will train until test_auc hasn't improved in 10 rounds.
## 
## [2]	train-auc:0.883410	test-auc:0.879504 
## [3]	train-auc:0.885483	test-auc:0.881195 
## [4]	train-auc:0.887355	test-auc:0.883168 
## [5]	train-auc:0.888049	test-auc:0.883759 
## [6]	train-auc:0.888929	test-auc:0.884309 
## [7]	train-auc:0.889512	test-auc:0.884922 
## [8]	train-auc:0.890106	test-auc:0.885050 
## [9]	train-auc:0.890731	test-auc:0.885478 
## [10]	train-auc:0.891281	test-auc:0.885567 
## [11]	train-auc:0.892173	test-auc:0.885953 
## [12]	train-auc:0.892694	test-auc:0.886183 
## [13]	train-auc:0.893115	test-auc:0.886109 
## [14]	train-auc:0.893624	test-auc:0.886137 
## [15]	train-auc:0.894083	test-auc:0.886210 
## [16]	train-auc:0.894629	test-auc:0.886274 
## [17]	train-auc:0.894938	test-auc:0.886400 
## [18]	train-auc:0.895341	test-auc:0.886312 
## [19]	train-auc:0.895662	test-auc:0.886199 
## [20]	train-auc:0.896162	test-auc:0.886297 
## [21]	train-auc:0.896544	test-auc:0.886334 
## [22]	train-auc:0.896818	test-auc:0.886324 
## [23]	train-auc:0.897191	test-auc:0.886304 
## [24]	train-auc:0.897718	test-auc:0.886311 
## [25]	train-auc:0.898014	test-auc:0.886336 
## [26]	train-auc:0.898533	test-auc:0.886300 
## [27]	train-auc:0.898964	test-auc:0.886314 
## Stopping. Best iteration:
## [17]	train-auc:0.894938	test-auc:0.886400
```

```r
xgb_clf$model$best_iteration
## [1] 17
```

> [!note]- code fold
> ```r
> melt(xgb_clf$model$evaluation_log, id.vars = "iter", variable.name = "set", value.name = "auc") %>%
> ggplot(aes(x = iter, y = auc, group = set)) +
>   geom_line(aes(color = set), lwd = 2) +
>   # ylim(c(0.7, 1)) +
>   geom_vline(aes(xintercept = xgb_clf$model$best_iteration), color = "black", lwd = 1, lty = 'dashed') +
>   # scale_y_continuous(breaks = seq(0.5, 1, 0.05)) +
>   scale_color_manual(values = c("#f8766d", "#00b0f6"), labels = c("Train", "Test")) +
>   labs(x = "Rounds", y = "AUROC", color = "Set") +
>   theme_minimal()
> ```

![[Pasted image 20240409160107.png]]

- 초반 라운드에서 validation score가 향상되다가 17번째 라운드 이후 100번 동안 성능에 개선이 일어나지 않으므로 조기 중단 수행

## 3.4 Evaluation

앞서 `partition()`으로 분할한 test set `split$test`에 대해 `Confusion Matrix`를 계산해 보자. 

```r
pred_base = xgb_clf$predict(task_encoded, row_ids = split$test)
pred_base$confusion %>% 
  as.data.table() %>% 
  plot_confusion_matrix(target_col = "truth",
                        prediction_col = "response",
                        counts_col = "N",
                        add_sums = T)
```

![[Pasted image 20240409160202.png|600]]

- Confusion Matrix를 보면, 실제로 은행을 유지한(`Not Exited`) 고객 26023명 중 24632명(94.7%)이 정확히 분류되었고, 해지한(`Exited`) 고객 6984명 중 3860명(55.3%)만 올바르게 분류되었다. 
- 이 기본 모델은 `Exited = 0`인 경우에는 잘 작동하지만, `Exited = 1`인 경우에 심각하게 저조한 성능을 보인다.
- 은행을 떠나는 (`Exited = 1`) 고객에게 더 많은 비용을 들이기 때문에 이러한 고객에 대해 더 많은 정보를 포착하고자 한다.

# 4. Hyperparameter Tuning & Cross-Validation

위와 같은 문제는 target인 `Exited`에 클래스 불균형이 있기 때문에 발생한다. XGBoost의 `scale_pos_weight` 초모수는 target label의 pos와 neg 간 클래스 불균형이 존재할 때 사용할 수 있는 것인데, 클래스 불균형이 심할 경우 이 초모수를 양수 값을 사용하면 어느 정도 보완할 수 있다. 


## 4.1 Bayesian Optimization with Holdout Resampling

- 아래에서 사용한 Tuning 알고리즘은 Bayesian Optimization with Holdout Resampling이다. 

```r
# XGBoost tuning
measure = msr("classif.auc")
search_space = ps(
  booster = p_fct(c("gbtree", "dart")), 
  tree_method = p_fct("hist"),
  eta = p_dbl(lower = 1e-04, upper = 1, logscale = T),
  colsample_bytree = p_dbl(lower = 0, upper = 1),
  gamma = p_dbl(lower = 1e-05, upper = 7, logscale = T),
  max_depth = p_int(lower = 1, upper = 15),
  subsample = p_dbl(lower = 1e-01, upper = 1),
  lambda = p_dbl(lower = 1e-4, upper = 1000, logscale = T),
  alpha = p_dbl(lower = 1e-4, upper = 1000, logscale = T),
  scale_pos_weight = p_dbl(lower = 1, upper = 10)
)
tuner_bo = tnr("mbo")

future::plan("multisession", workers = 10)
instance_xgb = tune(
  tuner = tuner_bo,
  task = task_encoded,
  learner = xgb_clf,
  resampling = rsmp("holdout"),
  measures = measure,
  search_space = search_space,
  term_evals = 25,
  callbacks = clbk("mlr3tuning.early_stopping")
)
```

튜닝 결과, 최적의 초모수 조합은 아래와 같다:

```r
instance_xgb$result_learner_param_vals %>% unlist()
##                nrounds                nthread                verbose     early_stopping_set              objective 
##                  "295"                    "1"                    "1"                 "none"      "binary:logistic" 
##            eval_metric                booster            tree_method                    eta       colsample_bytree 
##                  "auc"                 "dart"                 "hist"   "0.0651233058873209"    "0.563559353351593" 
##                  gamma              max_depth              subsample                 lambda                  alpha 
## "4.25543389771865e-05"                    "3"    "0.895265799760818"     "17.3746937786693"    "0.788251918939708" 
##       scale_pos_weight 
##     "9.99775123596191"
```

## 4.2 일반화 성능 추정: 10-fold CV

이제 최적의 초모수 조합을 사용한 XGBoost를 10-fold CV를 통해 일반화 성능을 추정해 보자:

```r
xgb_clf$param_set$values = instance_xgb$result_learner_param_vals

design = benchmark_grid(
  tasks = task_encoded,
  learners = xgb_clf,
  resamplings = rsmp("cv", folds = 10)
)
bmr = benchmark(design)
## INFO  [16:45:44.710] [mlr3] Running benchmark with 10 resampling iterations
## INFO  [16:45:44.973] [mlr3] Applying learner 'classif.xgboost' on task 'train_data' (iter 1/10)
## INFO  [16:45:45.307] [mlr3] Applying learner 'classif.xgboost' on task 'train_data' (iter 2/10)
## INFO  [16:45:45.635] [mlr3] Applying learner 'classif.xgboost' on task 'train_data' (iter 3/10)
## INFO  [16:45:45.997] [mlr3] Applying learner 'classif.xgboost' on task 'train_data' (iter 4/10)
## INFO  [16:45:46.364] [mlr3] Applying learner 'classif.xgboost' on task 'train_data' (iter 5/10)
## INFO  [16:45:46.753] [mlr3] Applying learner 'classif.xgboost' on task 'train_data' (iter 6/10)
## INFO  [16:45:47.197] [mlr3] Applying learner 'classif.xgboost' on task 'train_data' (iter 7/10)
## INFO  [16:45:47.686] [mlr3] Applying learner 'classif.xgboost' on task 'train_data' (iter 8/10)
## INFO  [16:45:48.211] [mlr3] Applying learner 'classif.xgboost' on task 'train_data' (iter 9/10)
## INFO  [16:45:49.050] [mlr3] Applying learner 'classif.xgboost' on task 'train_data' (iter 10/10)
## INFO  [16:50:12.636] [mlr3] Finished benchmark

bmr$aggregate(msr("classif.auc"))
##       nr    task_id      learner_id resampling_id iters classif.auc
##    <int>     <char>          <char>        <char> <int>       <num>
## 1:     1 train_data classif.xgboost            cv    10   0.8897431
## Hidden columns: resample_result
```

- 튜닝 전 Base Model의 AUROC 성능과 큰 차이를 보이지 않음

## 4.3 최종 모델

이제 최적의 초모수 조합을 사용해 Train Data에서의 성능을 측정해 보자. 

```r
xgb_clf$train(task_encoded)
pred_tuned = xgb_clf$predict(task_encoded, row_ids = task_encoded$row_roles$test)

pred_tuned$confusion %>%
  as.data.table() %>%
  plot_confusion_matrix(target_col = "truth",
                        prediction_col = "response",
                        counts_col = "N",
                        add_sums = T)
```

![[Pasted image 20240409170150.png|600]]

- `Exited = 0`인 26023명 중 16179명(62.2%)을 정확히 분류했으며, `Exited = 1`인 6984명 중 6429명(92.1%)를 정확히 분류했다.
- `Exited = 0`에 대한 정확도는 감소했지만, `Exited = 1` 클래스에 대한 정확도는 36.8% 증가했다.
- 보통 `Exited = 1`에 대해 관심을 가지므로 이는 모델의 성능이 많이 오른 것이라고 판단

# 5. Visualization

```r
xgboost::xgb.plot.tree(model = xgb_clf$model, trees = 1)
```

![[Pasted image 20240409172130.png]]


```r
## Feature Importance
xgb_exp = explain_mlr3(
  model = xgb_clf,
  data = task_encoded$data(rows = task_encoded$row_roles$test, cols = task_encoded$feature_names),
  y = as.numeric(task_encoded$data(rows = task_encoded$row_roles$test, cols = task_encoded$target_names)$Exited == 0),
  colorize = F,
  label = "XGBoost Explanation"
)
xgb_effect = model_parts(xgb_exp, B = 10, type = "raw")
plot(xgb_effect, show_boxplots = F)
```

![[Pasted image 20240409180441.png]]


# 6. Submission

```r
submission$Exited = xgb_clf$predict(task_test_encoded)$prob[, 1]
submission %>% head()
```

|id|Exited|
|---|---|
|165034|0.2048851|
|165035|0.9726685|
|165036|0.2073597|
|165037|0.7533769|
|165038|0.8504049|
|165039|0.3271881|

```r
submission %>% fwrite("../data/playground-series-s4e1/submission_xgb.csv", row.names = F)
```


![[Pasted image 20240409181846.png]]

- 추가적인 Feature Engineering을 하면 성능 개선 가능
- 다른 모델들을 추가로 사용해 Ensemble 모델을 사용해서 성능 개선 가능

