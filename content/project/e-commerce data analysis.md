---
sticker: emoji//1f525
date: 2023-12-08
tags:
  - ML
  - e-commerce
  - glmnet
  - Linear-regression
  - XGBoost
  - LightGBM
  - datamart
  - pre-processing
  - tidymodels
  - Online-Retail
---
해당 글은 [패스트캠퍼스](https://fastcampus.co.kr/data_online_msignature) 강의 내용을 참고하여 작성하였습니다. 

![[ecommerce.jpg|center|500]]

## 01. 문제해결 프로세스 기획

사용하고자 하는 데이터셋은 [UCI Repository](https://archive.ics.uci.edu/dataset/352/online+retail)에서 제공하는 "Online Retail" 데이터로 2010년 ~ 2011년에 영국 온라인 쇼핑몰에서 발생한 거래 이력을 포함하고 있다. 이 데이터를 가지고 아래와 같은 시나리오를 생각해보자. 

> [!quote] 시나리오
> A사는 이커머스 플랫폼을 운영 중에 있다. 해당 플랫폼 서비스가 성장하기 위해선 아래와 같은 3가지 조건이 필요하다:</br>
> 	① 신규 고객의 유입
> 	② 유입 고객 재구매
> 	③ 재구매 고객 충성 고객화</br></br>
> A사는 현재 마케팅 비용을 투자해 '①신규고객유입'은 적극적으로 수행하고 있으나 '②유입고객재구매'가 미미한 상황이다. 충성고객의 Spending Power는 전체 매출에 많은 비중을 차지하므로 충성고객으로의 성장이 이어지지 않으면 플랫폼의 성장도 멈추게 된다. 
> 
> 따라서 '②유입고객재구매'를 촉진시킬 방법을 고민 중에 있다. 

![[seven-step-problem-solving-process.png|center]] Figure 1: 문제해결 프로세스 7단계

위 시나리오를 기반으로 문제해결 프로세스 7단계에 따라 아래와 같은 프로세스를 정의해보자.

> [!example] 문제해결 프로세스 
> #### Step 1. 문제정의
> - 문제 현상: 신규 고객 유입 후 재구매로 이어지는 고객의 감소
> - 예상 피해: `(재구매고객 → 충성고객)`의 부진으로 매출 성장 정체 및 신규고객 유입을 위한 마케팅 비용 증가로 인한 영업이익 감소
> #### Step 2. 기대효과
> - 재구매 고객 증가 → 충성 고객 증가 → 매출 성장 → 영업 이익 증가
> - 선순환 체계 구축으로 인한 서비스 성장
> #### Step 3. 해결방안
> 	1) EDA 및 일회성 데이터분석을 통해 재구매 고객의 특성을 분석하고 이를 토대로 마케팅 기획
> 	2) 제구매 가능성이 높은 고객을 예측하는 모델링 후 이를 활용한 타겟 마케팅 진행
> #### Step 4. 우선순위
> - 1번을 빠르게 수행한 후 파일럿 테스트 진행 및 성과 측정
> - 1번의 효과가 좋지 않다면, 2번 진행 후 파일럿 테스트 재실행
> #### Step 5. 데이터분석
> - 결정된 우선순위에 따라 데이터 분석 및 모델링 진행
> #### Step 6. 성과 측정
> - 최종 마케팅 후 성능을 평가하기 위한 지표 수립
> - 분석 및 모델링을 통해 추출한 타겟 고객군과 대조군을 설정하여 A/B 테스트 수행
> - A/B 테스트 결과 마케팅(재구매) 반응률 비교를 통해 통계적으로 유의미한지 검증: t-test
> - 유의미한 결과를 얻을 때까지 파일럿 테스트를 수정해 가며 진행
> #### Step 7. 모델 운영
> - 파일럿 테스트 후 결과가 유의미하다면 정규 마케팅으로 운영하기 위한 작업 준비
> - 모델을 실행하기 위한 주기와 추출 타겟 고객군의 범위 결정
> - 정해진 주기에 따라 타겟 고객군의 추출 자동화
> - 이를 마케팅 시스템과 연계하여 타겟 마케팅을 주기적으로 운영 및 평가

- 현재 Step 1~4 단계가 완료되어 데이터 분석을 수행해야 하는 단계라고 간주하자. 

## 02. Data Readiness Check & Sampling

[UCI Repository](https://archive.ics.uci.edu/dataset/352/online+retail)의 "Online Retail" 데이터셋을 A사에서 수집한 고객 거래 이력 데이터로 간주하고 기본적인 전처리를 우선 수행해보자.

### 02-01. Data info check

주어진 데이터는 e-commerce 온라인 구매에 대한 데이터로 데이터 명세 정보는 아래와 같다: 


|     변수      |      설명      |    유형     |
|:-------------:|:--------------:|:-----------:|
|  `InvoiceNo`  | 송장번호 | Categorical |
|  `StockCode`  |    재고코드    | Categorical |
| `Description` |    상세설명    | Categorical |
|  `Quantity`   |      수량      |   Integer   |
| `InvoiceDate` |    송장날짜    |    Date     |
|  `UnitPrice`  |    개당가격    | Continuous  |
| `CustomerID`  |     고객ID     | Categorical |
|   `Country`   |      국가      | Categorical |

> [!example]- 기초적인 전처리
> ```r
> library(tidyverse)
> df <- read_csv("../../data/e-commerce/Online Retail.csv", col_type = "ccciTdcc")
> df %>% head()
> ```
> ```
> # A tibble: 6 × 8
>   InvoiceNo StockCode Description                           Quantity InvoiceDate         UnitPrice CustomerID Country       
>   [chr]     [chr]     [chr]                                    [int] [dttm]                  [dbl] [chr]      [chr]         
> 1 489434    85048     "15CM CHRISTMAS GLASS BALL 20 LIGHTS"       12 2009-12-01 07:45:00      6.95 13085      United Kingdom
> 2 489434    79323P    "PINK CHERRY LIGHTS"                        12 2009-12-01 07:45:00      6.75 13085      United Kingdom
> 3 489434    79323W    "WHITE CHERRY LIGHTS"                       12 2009-12-01 07:45:00      6.75 13085      United Kingdom
> 4 489434    22041     "RECORD FRAME 7\" SINGLE SIZE"              48 2009-12-01 07:45:00      2.1  13085      United Kingdom
> 5 489434    21232     "STRAWBERRY CERAMIC TRINKET BOX"            24 2009-12-01 07:45:00      1.25 13085      United Kingdom
> 6 489434    22064     "PINK DOUGHNUT TRINKET POT"                 24 2009-12-01 07:45:00      1.65 13085      United Kingdom
> ```
> 
> #### Data 크기 및 type 확인
> 
> ```r
> df %>% glimpse()
> ```
> ``` 
> Rows: 1,067,370
> Columns: 8
> $ InvoiceNo   [chr] "489434", "489434", "489434", "489434", "489434", …
> $ StockCode   [chr] "85048", "79323P", "79323W", "22041", "21232", "22…
> $ Description [chr] "15CM CHRISTMAS GLASS BALL 20 LIGHTS", "PINK CHERR…
> $ Quantity    [int] 12, 12, 12, 48, 24, 24, 24, 10, 12, 12, 24, 12, 10…
> $ InvoiceDate [dttm] 2009-12-01 07:45:00, 2009-12-01 07:45:00, 2009-12…
> $ UnitPrice   [dbl] 6.95, 6.75, 6.75, 2.10, 1.25, 1.65, 1.25, 5.95, 2.…
> $ CustomerID  [chr] "13085", "13085", "13085", "13085", "13085", "1308…
> $ Country     [chr] "United Kingdom", "United Kingdom", "United Kingdo…
> ```
> 
> #### 결측치 확인
> ```r
> df %>% is.na() %>% colSums()
> ```
> ```
>   InvoiceNo   StockCode Description    Quantity InvoiceDate   UnitPrice  CustomerID     Country 
>           0           0        4382           0           0           0      243007           0
> ```
> 
> - 고객 식별 정보인 `CustomerID` 변수에 결측치가 있는 경우 추후 해당 고객 정보를 분석에 사용할 수 없으므로 사전에 정보를 알 수 없는 이 같은 데이터는 삭제하자. 
> 
> ```r
> ### 결측치 삭제
> df <- df %>% drop_na("CustomerID")
> df %>% is.na() %>% colSums()
> ```
> ```
>   InvoiceNo   StockCode Description    Quantity InvoiceDate   UnitPrice  CustomerID     Country 
>           0           0           0           0           0           0           0           0
> ```
> 
> #### Outlier 확인
> 
> ```r
> df %>% 
>   select_if(is.numeric) %>% 
>   summary()
> ```
> ```
>     Quantity           UnitPrice       
>  Min.   :-80995.00   Min.   :    0.00  
>  1st Qu.:     2.00   1st Qu.:    1.25  
>  Median :     5.00   Median :    1.95  
>  Mean   :    12.41   Mean   :    3.68  
>  3rd Qu.:    12.00   3rd Qu.:    3.75  
>  Max.   : 80995.00   Max.   :38970.00
> ```
> 
> - 구매 수량 `Quantity`는 음수값을 가질 수 없으나 반품 물품에 대해 음수로 나타나 있으므로 확인 후 처리해야 한다. 여기서는 `Quantity`가 음수인 것을 이상치(outlier)라고 간주하고 삭제한다. 
> 
> ```r
> ### Quantity가 음수인 데이터 확인: 반품/회수 물량일 수 있음
> df %>% filter(Quantity < 0)
> ```
> ```
> # A tibble: 18,744 × 8
>    InvoiceNo StockCode Description                       Quantity InvoiceDate         UnitPrice CustomerID Country       
>    [chr]     [chr]     [chr]                                [int] [dttm]                  [dbl] [chr]      [chr]         
>  1 C489449   22087     PAPER BUNTING WHITE LACE               -12 2009-12-01 10:33:00      2.95 16321      Australia 
>  2 C489449   85206A    CREAM FELT EASTER EGG BASKET            -6 2009-12-01 10:33:00      1.65 16321      Australia 
>  3 C489449   21895     POTTING SHED SOW 'N' GROW SET           -4 2009-12-01 10:33:00      4.25 16321      Australia 
>  4 C489449   21896     POTTING SHED TWINE                      -6 2009-12-01 10:33:00      2.1  16321      Australia 
>  5 C489449   22083     PAPER CHAIN KIT RETRO SPOT             -12 2009-12-01 10:33:00      2.95 16321      Australia 
>  6 C489449   21871     SAVE THE PLANET MUG                    -12 2009-12-01 10:33:00      1.25 16321      Australia 
>  7 C489449   84946     ANTIQUE SILVER TEA GLASS ETCHED        -12 2009-12-01 10:33:00      1.25 16321      Australia 
>  8 C489449   84970S    HANGING HEART ZINC T-LIGHT HOLDER      -24 2009-12-01 10:33:00      0.85 16321      Australia
>  9 C489449   22090     PAPER BUNTING RETRO SPOTS              -12 2009-12-01 10:33:00      2.95 16321      Australia     
> 10 C489459   90200A    PURPLE SWEETHEART BRACELET              -3 2009-12-01 10:44:00      4.25 17592      United Kingdom
> # ℹ 18,734 more rows
> # ℹ Use `print(n = ...)` to see more rows
> ```
> 
> ```r
> ### Quantity 음수값 제거
> df <- df %>% 
>   filter(Quantity > 0)
> ```
> 
> #### 중복 데이터 확인
> 
> ```r
> df %>% 
>   mutate(duplicated = duplicated(.)) %>% 
>   count(duplicated)
> ```
> ```
> # A tibble: 2 × 2
>   duplicated      n
>   [lgl]       [int]
> 1 FALSE      779494
> 2 TRUE        26125
> ```
> 
> ```r
> ### 중복 데이터 확인
> df %>% 
>   filter(duplicated(df) | duplicated(df, fromLast = T))
> ```
> ```
> # A tibble: 50,838 × 8
>    InvoiceNo StockCode Description                       Quantity InvoiceDate         UnitPrice CustomerID Country       
>    [chr]     [chr]     [chr]                                [int] [dttm]                  [dbl] [chr]      [chr]         
>  1 489517    21913     VINTAGE SEASIDE JIGSAW PUZZLES           1 2009-12-01 11:34:00      3.75 16329      United Kingdom 
>  2 489517    21912     VINTAGE SNAKES & LADDERS                 1 2009-12-01 11:34:00      3.75 16329      United Kingdom 
>  3 489517    21821     GLITTER STAR GARLAND WITH BELLS          1 2009-12-01 11:34:00      3.75 16329      United Kingdom 
>  4 489517    22319     HAIRCLIPS FORTIES FABRIC ASSORTED       12 2009-12-01 11:34:00      0.65 16329      United Kingdom 
>  5 489517    22130     PARTY CONE CHRISTMAS DECORATION          6 2009-12-01 11:34:00      0.85 16329      United Kingdom
>  6 489517    21912     VINTAGE SNAKES & LADDERS                 1 2009-12-01 11:34:00      3.75 16329      United Kingdom 
>  7 489517    21491     SET OF THREE VINTAGE GIFT WRAPS          1 2009-12-01 11:34:00      1.95 16329      United Kingdom 
>  8 489517    22130     PARTY CONE CHRISTMAS DECORATION          6 2009-12-01 11:34:00      0.85 16329      United Kingdom 
>  9 489517    22319     HAIRCLIPS FORTIES FABRIC ASSORTED       12 2009-12-01 11:34:00      0.65 16329      United Kingdom
> 10 489517    21913     VINTAGE SEASIDE JIGSAW PUZZLES           1 2009-12-01 11:34:00      3.75 16329      United Kingdom
> # ℹ 50,828 more rows
> # ℹ Use `print(n = ...)` to see more rows
> ```
> 
> - 26,125개의 데이터가 중복으로 들어가 있다. 데이터의 정확성과 품질을 위해 중복 데이터를 삭제해야 한다. 
> 
> ```r
> ### 중복 삭제 후 재확인
> df <- df %>% distinct()
> df %>% 
>   mutate(duplicated = duplicated(.)) %>% 
>   count(duplicated)
> ```
> ```
> # A tibble: 1 × 2
>   duplicated      n
>   [lgl]       [int]
> 1 FALSE      779494
> ```


이렇게 주어진 데이터에 대한 기본적인 정보 확인과 전처리를 마치고 나면 아래와 같이 기존 1,067,370개의 ㅣ행에서 779,494개의 행으로 줄어든 것을 확인할 수 있다.

```r
### 1,067,370 → 779,494
df %>% dim()
```

### 02-02. Data Readiness Check

앞서 데이터의 기본 정보를 확인했으니 현재 가진 데이터로 기획한 문제해결 프로세스를 적용할 수 있는지 점검해야 한다. 

#### (1) Target Label 생성

> [!todo]- Target label 생성 코드
> - 당월 구매 고객이 다음 달에 재구매 시 해당 고객을 재구매 고객으로 정의하자. 예를 들어 2011년 01월에 구매한 고객이 2011년 02월에 다시 구매를 한다면 해당 고객이 재구매 고객이 되는 것이다. 
> 
> ```r
> library(lubridate)
> 
> ### 기준년월 변수 생성: bsym - %Y-%m 형식
> df <- df %>% 
>   mutate(bsym = format(InvoiceDate, "%Y-%m"))
> ### 원본 데이터 저장: df_origin
> df_origin <- df
> # 데이터 적재 기간 확인: 2009-12-01 ~ 2011-12-09 (약 2년)
> min(df$InvoiceDate); max(df$InvoiceDate)
> ```
> ```
> [1] "2009-12-01 07:45:00 UTC"
> [1] "2011-12-09 12:50:00 UTC"
> ```
> 
> - 기본적인 전처리가 완료된 데이터(`df`)를 `df_origin`으로 저장해 보존하고, 이후 있을 전처리를 진행해보자. 
> - Target label은 기준년월(`bsym`)과 고객(`CustomerID`)으로만 정의되기 때문에 두 변수만을 고유하게 갖는 데이터로 재구성
> 
> ```r
> df <- df %>% 
>   distinct(bsym, CustomerID)
> df %>% dim()
> ```
> ```
> [1] 25598     2
> ```
> 
> ```r
> df %>% head()
> ```
> 
> ```
> # A tibble: 6 × 2
>   bsym    CustomerID
>   [chr]   [chr]     
> 1 2009-12 13085     
> 2 2009-12 13078     
> 3 2009-12 15362     
> 4 2009-12 18102     
> 5 2009-12 12682     
> 6 2009-12 18087
> ```
> 
> - 2009년 12월부터 2011년 12월까지 `bsym`별로 당월 구매 고객이 내월 구매 고객일 경우 `1`을 갖는 binary `target` 변수 생성
> 
> 
> ```r
> # 주어진 bsym에 구매 고객이 내월 구매 고객일 경우 1, 그렇지 않으면 0 값을 갖는 target 변수 생성: process_bsym()
> process_bsym <- function(bsym_value, df) {
>   df_left <- filter(df, bsym == bsym_value)
>   bsym_1m <- ymd(paste0(bsym_value, "-01")) %m+% months(1) %>% format("%Y-%m")
>   df_right <- df %>% 
>     filter(bsym == bsym_1m) %>% 
>     distinct(CustomerID) %>% 
>     mutate(target = 1)
>   
>   df_merge <- left_join(df_left, df_right, by = "CustomerID") %>% 
>     mutate(target = ifelse(is.na(target), 0, target))
>   
>   return(df_merge)
> }
> ```

```r
# 모든 bsym 값에 대해 process_bsym 함수 적용
df_all <- map_df(unique(df$bsym), ~process_bsym(.x, df))
df_all %>% head()
```
```
# A tibble: 6 × 3
  bsym    CustomerID target
  <chr>   <chr>       <dbl>
1 2009-12 13085           1
2 2009-12 13078           1
3 2009-12 15362           0
4 2009-12 18102           1
5 2009-12 12682           1
6 2009-12 18087           1
```
#### (2) Target Ratio 확인

> [!todo]- Target ratio 확인 코드
> - 연도 및 월별로 재구매 고객 여부 `target`의 비율 확인:
> ```r
> #### bsym 기준 target ratio:
> options(pillar.sigfig = 6)
> df_target <- df_all %>% 
>   group_by(bsym) %>% 
>   reframe(total_y = sum(target),
>           count_y = n(),
>           ratio = total_y/count_y)
> df_target %>% 
>   print(n = nrow(.))
> ```
> ```
> # A tibble: 25 × 4
>    bsym    total_y count_y    ratio
>    [chr]     [dbl]   [int]    [dbl]
>  1 2009-12     337     955 0.352880
>  2 2010-01     262     720 0.363889
>  3 2010-02     314     774 0.405685
>  4 2010-03     378    1057 0.357616
>  5 2010-04     345     942 0.366242
>  6 2010-05     368     966 0.380952
>  7 2010-06     392    1041 0.376561
>  8 2010-07     351     928 0.378233
>  9 2010-08     365     911 0.400659
> 10 2010-09     463    1145 0.404367
> 11 2010-10     657    1497 0.438878
> 12 2010-11     524    1607 0.326073
> 13 2010-12     324     885 0.366102
> 14 2011-01     262     741 0.353576
> 15 2011-02     290     758 0.382586
> 16 2011-03     304     974 0.312115
> 17 2011-04     368     856 0.429907
> 18 2011-05     410    1056 0.388258
> 19 2011-06     365     991 0.368315
> 20 2011-07     388     949 0.408851
> 21 2011-08     425     935 0.454545
> 22 2011-09     489    1266 0.386256
> 23 2011-10     622    1364 0.456012
> 24 2011-11     371    1665 0.222823
> 25 2011-12       0     615 0
> ```
> 
> - 현재 주어진 데이터는 2011년 12월이 마지막이기 때문에 내월이 없는 2011년 12월은 `target = 1`을 가질 수 없다. 따라서 분석 대상에서 제외:
> ```r
> df_target <- df_target %>% filter(bsym != '2011-12')
> df_all <- df_all %>% filter(bsym != '2011-12')
> 
> df_target$ratio %>% summary()
> ```
> ```
>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
>  0.2228  0.3623  0.3796  0.3784  0.4047  0.4560
> ```
> 
> - `bsym`별 `target`의 비율은 22.3% ~ 45.6%의 값을 가지며, 대부분 36%~40% 정도의 비율을 가지고 있다. 
> - 전체 데이터의 `target` 비율을 보면 다음과 같이 약 37.5% 정도의 내달 재구매 비율을 보이고 있다.
> 	- 이정도의 class imbalance에 대해서는 해소하지 않고 진행해도 무방함
> ```
> > df_all$target %>% mean()
> [1] 0.3752151
> ```
> ```r
> #### 마지막 행에 합계 추가
> df_target <- df_target %>% 
>   bind_rows(
>     tibble(bsym = "total",
>            total_y = sum(.$total_y),
>            count_y = sum(.$count_y),
>            ratio = 1)  
>   )
> ```

### 02-03. Data Sampling

```r
df_all %>% dim()
```
```
[1] 24983     3
```

현재 `df_all`에는 약 25,000건의 데이터가 있지만, 만일 데이터의 크기가 1억건 혹은 고차원인 매우 큰 데이터일 경우에는 모델링에 부담이 커진다. 이를 대비해 해당 데이터를 모집단으로 간주하고 표본 데이터를 추출(sampling)하여 데이터를 줄일 수 있다. 

#### Under-sampling: Stratified sampling

![[stratified-sampling.png|center]]

관심변수(`target`)에 대해 함께 살펴보고 싶은 변수가 존재할 때(여기선 `bsym`), 해당 변수를 층화변수로 삼아 층화추출법(stratified sampling)을 적용할 수 있다. 층화추출법은 모집단을 층화변수로 분류/나눈 후 각 층/소집단 별로 독립적으로 표본을 뽑는 샘플링 방법이다. 이를 통해 모집단에서 층화변수와 관심변수가 갖는 관계를 표본 데이터에서도 유지할 수 있다. 즉, 표본의 대표성을 확보하면서 데이터의 크기도 줄일 수 있는 방법이다. 

현재 `df_all`을 모집단 데이터로 간주하고 이로부터 표본 데이터를 추출해보자. 이 때 층화추출법을 사용하기 위해선 우선 각 층, `bsym`별로 표본 크기가 주어져야 한다. 이 과정을 표본 배분(sample allocation)이라고 한다. 여기서는 모집단 `df_all`의 `bsym`별 층 크기를 유지하는 비례 배분(proportional allocation) 방식을 적용한다. 비례 배분에 따른 표본 층 크기는 다음과 같다. 
$$ 
n_h = n\times\frac{N_h}{\sum_{l=1}^HN_l}
$$
전체 `df_all`의 30%만 추출한다고 하자. 필요한 층 `bsym`별 표본 층 크기 `nh`는 다음과 같다: 

```r
#### 전체 데이터의 30%: 7,495개
round(nrow(df_all) * 0.3)
```
```
[1] 7495
```

비례 배분을 적용한 모집단 및 표본 층 크기(`Nh` & `nh`)는 다음과 같다:

```r
#### 모집단 층 크기: Nh, 표본 층 크기: nh
df_target <- df_target %>% 
  mutate(N = nrow(df_all)) %>% 
  mutate(n = round(nrow(df_all) * 0.3)) %>% 
  relocate(count_y, .after = n) %>% 
  rename(Nh = count_y) %>% 
  mutate(nh = round(n*Nh/N)) %>% 
  group_by(bsym)

df_target %>% print(n=25)
```
```
# A tibble: 25 × 7
# Groups:   bsym [25]
   bsym    total_y    ratio     N     n    Nh    nh
   <chr>     <dbl>    <dbl> <int> <dbl> <int> <dbl>
 1 2009-12     337 0.352880 24983  7495   955   287
 2 2010-01     262 0.363889 24983  7495   720   216
 3 2010-02     314 0.405685 24983  7495   774   232
 4 2010-03     378 0.357616 24983  7495  1057   317
 5 2010-04     345 0.366242 24983  7495   942   283
 6 2010-05     368 0.380952 24983  7495   966   290
 7 2010-06     392 0.376561 24983  7495  1041   312
 8 2010-07     351 0.378233 24983  7495   928   278
 9 2010-08     365 0.400659 24983  7495   911   273
10 2010-09     463 0.404367 24983  7495  1145   344
11 2010-10     657 0.438878 24983  7495  1497   449
12 2010-11     524 0.326073 24983  7495  1607   482
13 2010-12     324 0.366102 24983  7495   885   266
14 2011-01     262 0.353576 24983  7495   741   222
15 2011-02     290 0.382586 24983  7495   758   227
16 2011-03     304 0.312115 24983  7495   974   292
17 2011-04     368 0.429907 24983  7495   856   257
18 2011-05     410 0.388258 24983  7495  1056   317
19 2011-06     365 0.368315 24983  7495   991   297
20 2011-07     388 0.408851 24983  7495   949   285
21 2011-08     425 0.454545 24983  7495   935   281
22 2011-09     489 0.386256 24983  7495  1266   380
23 2011-10     622 0.456012 24983  7495  1364   409
24 2011-11     371 0.222823 24983  7495  1665   500
25 total      9374 1        24983  7495 24983  7495
```

이 때 반올림으로 인해 표본 층크기가 1개 더 크게 나온다:
```r
df_target$nh[-25] %>% sum()
```
```
[1] 7496
```

2011년 11월 표본 크기를 한 개 줄여서 해결:

```r
df_target$nh[df_target$bsym=='2011-11'] <- df_target$nh[df_target$bsym=='2011-11'] - 1
```

이제 (비례배분을 통해) 주어진 표본배분안에 따라 `bsym`별 30% 정도의 표본을 뽑아보자:
```r
set.seed(123)
ord <- unique(df_target$bsym)
units <- sampling::strata(df_all, stratanames = "bsym", size = df_target$nh[1:24], method="srswor")

df_all_sample <- df_all %>% 
  slice(units$ID_unit)
df_all_sample %>% dim()
```
```
[1] 7495 3
```

다음은 층화추출로 뽑은 표본 데이터 `df_all_sample`이 기존 모집단 `df_all`의 `bsym`별 `target` 비율을 유지하는지 확인한 표를 나타낸다.

> [!note]- code fold
> ```r
> library(gt)
> library(gtExtras)
> df_tmp <- df_all_sample %>% 
>   group_by(bsym) %>% 
>   reframe(sum_y = sum(target),
>           nh = n())
> 
> df_tmp2 <- df_tmp %>% 
>   mutate(target_ratio = sum_y/nh,
>          target_ratio_pop = df_target$ratio[1:24]) %>% 
>   pivot_longer(cols=c(target_ratio, target_ratio_pop), names_to = 'name') %>% 
>   group_by(bsym) %>% 
>   reframe(target_ratio = list(value))
> df_tmp %>% left_join(df_tmp2, by='bsym') %>% 
>   gt() %>% 
>   gt_theme_nytimes() %>% 
>   tab_header(title = "bsym별 target 비율") %>% 
>   gt_plt_bar_stack(column = target_ratio, labels = c("target_ratio", "target_ratio_pop"), palette = c("skyblue", "hotpink"))
> ```

![[Pasted image 20240219200917.png|center]]

```r
#### 표본 전체 target ratio: 38.1%
#### 모집단 target ratio: 37.5%
df_all_sample$target %>% mean()
```
```
[1] 0.3813209
```
```r
df_all$target %>% mean()
```
```
[1] 0.3752151
```

이렇게 구성한 `df_all_sample` 표본 데이터를 가지고 분석에 사용할 Data Mart를 만들어보자.

## 03. Data Mart & Feature Engineering

### 03-01. Data Mart 기획 및 설계

![[data-mart.png|center]]

```
> df_all %>% dim()
[1] 24983     3
> df_origin %>% dim()
[1] 779494      9
> df_all_sample %>% dim()
[1] 7495    3
```

고객 24,983명(`df_all`)의 거래 이력이 총 779,494건(`df_origin`)이고, 이 고객 데이터셋을 층화추출법을 통해 undersampling하여 추출한 분석 대상이 7,495명(`df_all_sample`)의 고객 데이터이었다. 이를 이용해 고객별 구매 이력에 대한 Data Mart를 구성하고자 한다. 

여러가지 가설을 세워 재구매 여부 `target`에 영향을 미치는 여러 가지 변수(features)를 만들어보자. 아래는 Data Mart를 구성하는 여러 변수들에 대한 설명과 로직이 작성되어 있는 Data Mart 기획서이다. 

![[data-mart-desc.png|center]]

### 03-02. Data 추출 및 Mart 개발

앞서 기본전인 전처리가 끝난 후의 원본 데이터를 `df_orgin`으로 저장해두었다. 샘플링한 표본 데이터 `df_all_sample`을 이용해 Data Mart를 구성하기 위해선 우선 `df_origin`과 `df_all_sample`의 (`bsym`, `CustomerID`)를 key로 하여 병합해야 한다. 

> [!example]- Key 변수 생성 및 데이터 병합 코드
> ```r
> df_origin %>% head()
> ```
> ```
> # A tibble: 6 × 9
>   InvoiceNo StockCode Description                           Quantity InvoiceDate         UnitPrice CustomerID Country        bsym   
>   [chr]     [chr]     [chr]                                    [int] [dttm]                  [dbl] [chr]      [chr]          [chr]  
> 1 489434    85048     "15CM CHRISTMAS GLASS BALL 20 LIGHTS"       12 2009-12-01 07:45:00      6.95 13085      United Kingdom 2009-12
> 2 489434    79323P    "PINK CHERRY LIGHTS"                        12 2009-12-01 07:45:00      6.75 13085      United Kingdom 2009-12
> 3 489434    79323W    "WHITE CHERRY LIGHTS"                       12 2009-12-01 07:45:00      6.75 13085      United Kingdom 2009-12
> 4 489434    22041     "RECORD FRAME 7\" SINGLE SIZE"              48 2009-12-01 07:45:00      2.1  13085      United Kingdom 2009-12
> 5 489434    21232     "STRAWBERRY CERAMIC TRINKET BOX"            24 2009-12-01 07:45:00      1.25 13085      United Kingdom 2009-12
> 6 489434    22064     "PINK DOUGHNUT TRINKET POT"                 24 2009-12-01 07:45:00      1.65 13085      United Kingdom 2009-12
> ```
> ```r
> df_all_sample %>% head()
> ```
> ```
> # A tibble: 6 × 3
>   bsym    CustomerID target
>   [chr]   [chr]       [dbl]
> 1 2009-12 12682           1
> 2 2009-12 15413           1
> 3 2009-12 16321           0
> 4 2009-12 15712           1
> 5 2009-12 17700           1
> 6 2009-12 14911           1
> ```
> 
> ```r
> #### df_origin에 key 변수 생성
> df_origin <- df_origin %>% 
>   mutate(key = str_c(bsym, CustomerID))
> df_origin %>% 
>   reframe(n_key = n_distinct(key))
> ```
> ```
> # A tibble: 1 × 1
>   n_key
>   [int]
> 1 25598
> ```
> 
> ```r
> #### df_all_sample에 key 변수 생성
> df_all_sample <- df_all_sample %>% 
>   mutate(key = str_c(bsym, CustomerID))
> df_all_sample %>% 
>   reframe(n_key = n_distinct(key))
> ```
> ```
> # A tibble: 1 × 1
>   n_key
>   [int]
> 1  7495
> ```
> 
> - 이제 `df_origin`의 `key`(`bsym` & `CustomerID`)를 이용해서 `df_all_sample`에 존재하는 행들만 가져와 `df_origin_sample`이라는 데이터로 저장:
> ```r
> df_origin_sample <- df_origin %>% 
>   filter(key %in% df_all_sample$key)
> 
> # df_origin과 df_origin_sample의 비율: 대략 30% 정도
> nrow(df_origin_sample)/nrow(df_origin)
> ```
> ```
> [1] 0.2947758
> ```

#### Mart 구성

> [!example]- Mart 구성 코드 
> 1. **구매금액**: 월별 구매금액에 따라 다음 달 재구매 확률이 다를 수 있음
> 	- Mart 기획서의 첫 번째 변수인 **구매금액** 관련 3개 변수를 만들기 위해 `StockCode`당 구매금액을 나타내는 `amt`를 `UnitPrice * Quantity`로 정의했다.
> ```r
> ### 1. 구매금액 amt 관련 변수 total_amt, max_amt, min_amt
> #### 1) total_amt: 당월 총 구매금액
> df_mart <- df_origin_sample %>%
>   mutate(amt = UnitPrice * Quantity) %>% 
>   group_by(bsym, CustomerID) %>% 
>   reframe(total_amt = sum(amt, na.rm = T))
> 
> #### 2) max_amt, min_amt: 당월 송장당 최대/최소 구매금액
> df_mart <- df_mart %>% left_join(
>   df_origin_sample %>% 
>     mutate(amt = UnitPrice * Quantity) %>% 
>     group_by(bsym, CustomerID, InvoiceNo) %>% 
>     reframe(amt = sum(amt, na.rm = T)) %>% 
>     group_by(bsym, CustomerID) %>% 
>     reframe(max_amt = max(amt, na.rm = T),
>             min_amt = min(amt, na.rm = T)),
>   by = c("bsym", "CustomerID")
> )
> 
> df_mart %>% head()
> ```
> ```
> # A tibble: 6 × 5
>   bsym    CustomerID total_amt max_amt min_amt
>   [chr]   [chr]          [dbl]   [dbl]   [dbl]
> 1 2009-12 12437         578.28  578.28  578.28
> 2 2009-12 12523          74.56   74.56   74.56
> 3 2009-12 12539        5149.06 2583.23 2565.83
> 4 2009-12 12557        1952.64 1952.64 1952.64
> 5 2009-12 12664         549.08  549.08  549.08
> 6 2009-12 12681        1014.54 1014.54 1014.54
> ```
> 
> 2. **구매건수**: 월별 구매건수에 따라 다음 달 재구매 확률이 다를 수 있음
> 	- 이제 구매건수(`cnt`)와 관련된 3가지 변수를 정의하자. 월별 총 구매건수는 `total_cnt`로, 월별 송장(`InvoiceNo`)별 구매 품목 수의 최대/최소는 `max_cnt`, `min_cnt`로 정의했다.
> ```r
> ### 2. 구매건수 cnt 관련 변수
> #### 1) total_cnt: 당월 총 구매건수
> df_mart <- df_mart %>% left_join(
>   df_origin_sample %>% 
>     group_by(bsym, CustomerID) %>% 
>     reframe(total_cnt = n_distinct(InvoiceNo)),
>   by = c("bsym", "CustomerID")
> )
> #### 2) max_cnt, min_cnt: InvoiceNo별 최대/최소 구매 품목 수
> df_mart <- df_mart %>% 
>   left_join(
>     df_origin_sample %>% 
>       group_by(bsym, CustomerID, InvoiceNo) %>% 
>       reframe(cnt = n_distinct(StockCode)) %>% 
>       group_by(bsym, CustomerID) %>% 
>       reframe(max_cnt = max(cnt, na.rm = T),
>               min_cnt = min(cnt, na.rm = T)),
>     by = c("bsym", "CustomerID")
>   )
> 
> 
> df_mart %>% head()
> ```
> ```
> # A tibble: 6 × 8
>   bsym    CustomerID total_amt max_amt min_amt total_cnt max_cnt min_cnt
>   [chr]   [chr]          [dbl]   [dbl]   [dbl]     [int]   [int]   [int]
> 1 2009-12 12437         578.28  578.28  578.28         1      27      27
> 2 2009-12 12523          74.56   74.56   74.56         1       4       4
> 3 2009-12 12539        5149.06 2583.23 2565.83         2     104     103
> 4 2009-12 12557        1952.64 1952.64 1952.64         1       3       3
> 5 2009-12 12664         549.08  549.08  549.08         1       4       4
> 6 2009-12 12681        1014.54 1014.54 1014.54         1      46      46
> ```
> 
> 3. **구매수량**: 월별 구매수량에 따라 다음 달 재구매 확률이 다를 수 있음
> ```r
> ### 3. 구매수량 qty 관련 변수
> #### 1) total_qty: 당월 총 구매수량
> #### 2) min/max_qty: 당월 최소/최대 구매수량
> df_mart <- df_mart %>% left_join(
>   df_origin_sample %>% 
>     group_by(bsym, CustomerID) %>% 
>     reframe(total_qty = sum(Quantity, na.rm = T),
>             max_qty = max(Quantity, na.rm = T),
>             min_qty = min(Quantity, na.rm = T)),
>   by = c("bsym", "CustomerID")
> )
> 
> df_mart %>% head()
> ```
> ```
> # A tibble: 6 × 11
>   bsym    CustomerID total_amt max_amt min_amt total_cnt max_cnt min_cnt total_qty max_qty min_qty
>   [chr]   [chr]          [dbl]   [dbl]   [dbl]     [int]   [int]   [int]     [int]   [int]   [int]
> 1 2009-12 12437         578.28  578.28  578.28         1      27      27       263      24       3
> 2 2009-12 12523          74.56   74.56   74.56         1       4       4        62      36       4
> 3 2009-12 12539        5149.06 2583.23 2565.83         2     104     103      2128      48       2
> 4 2009-12 12557        1952.64 1952.64 1952.64         1       3       3       576     216     144
> 5 2009-12 12664         549.08  549.08  549.08         1       4       4       134      72       2
> 6 2009-12 12681        1014.54 1014.54 1014.54         1      46      46       650      72       1
> ```
> 
> 4. **국적**: 국적에 따라 재구매 확률이 다를 수 있음
> ```r
> df_mart <- df_mart %>% left_join(
>   df_origin_sample %>% 
>     group_by(bsym, CustomerID) %>% 
>     reframe(Country = first(Country)),
>   by = c("bsym", "CustomerID")
> )
> df_mart %>% head()
> ```
> ```
> # A tibble: 6 × 12
>   bsym  CustomerID total_amt max_amt min_amt total_cnt max_cnt min_cnt total_qty max_qty min_qty Country
>   [chr] [chr]          [dbl]   [dbl]   [dbl]     [int]   [int]   [int]     [int]   [int]   [int] [chr]  
> 1 2009… 12437         578.28  578.28  578.28         1      27      27       263      24       3 France 
> 2 2009… 12523          74.56   74.56   74.56         1       4       4        62      36       4 France 
> 3 2009… 12539        5149.06 2583.23 2565.83         2     104     103      2128      48       2 Spain  
> 4 2009… 12557        1952.64 1952.64 1952.64         1       3       3       576     216     144 Spain  
> 5 2009… 12664         549.08  549.08  549.08         1       4       4       134      72       2 Finland
> 6 2009… 12681        1014.54 1014.54 1014.54         1      46      46       650      72       1 France
> ```
> 
> 5. **구매시간대**: 구매 시간대(아침, 점심, 저녁, 밤)에 따라 재구매 확률이 다를 수 있음
> 	- 월별 고객별 아침, 점심, 저녁, 밤에 따른 구매 빈도를 계산한 후 이 값이 가장 높은 시간대를 `peak_time`이라는 변수로 정의
> ```r
> ### 5. 구매 시간대(아침, 점심, 저녁, 밤)
> #### 아침: 6~12시, 점심: 12~18시, 저녁: 18~24시, 밤: 0~6시
> #### 시간대별 구매 빈도 계산
> df_mart <- df_mart %>% left_join(
>   df_origin_sample %>% 
>     mutate(hour = hour(InvoiceDate),
>            peak_time = case_when(
>              hour >= 6  & hour < 12 ~ "Morning",
>              hour >= 12 & hour < 18 ~ "Afternoon",
>              hour >= 18 & hour < 24 ~ "Evening",
>              TRUE ~ "Night"
>            )) %>% 
>     group_by(bsym, CustomerID, peak_time) %>% 
>     reframe(purchase_cnt = n()) %>% 
>     group_by(bsym, CustomerID) %>% 
>     slice_max(purchase_cnt, n = 1, with_ties = FALSE) %>% 
>     select(-purchase_cnt) %>% 
>     ungroup(),
>   by = c("bsym", "CustomerID")
> )
> df_mart %>% head()
> ```
> ```
> # A tibble: 6 × 14
>   bsym    CustomerID total_amt max_amt min_amt total_cnt max_cnt min_cnt total_qty max_qty min_qty Country peak_time
>   [chr]   [chr]          [dbl]   [dbl]   [dbl]     [int]   [int]   [int]     [int]   [int]   [int] [chr]   [chr]       
> 1 2009-12 12437         578.28  578.28  578.28         1      27      27       263      24       3 France  Afternoon    
> 2 2009-12 12523          74.56   74.56   74.56         1       4       4        62      36       4 France  Afternoon    
> 3 2009-12 12539        5149.06 2583.23 2565.83         2     104     103      2128      48       2 Spain   Afternoon    
> 4 2009-12 12557        1952.64 1952.64 1952.64         1       3       3       576     216     144 Spain   Morning        
> 5 2009-12 12664         549.08  549.08  549.08         1       4       4       134      72       2 Finland Morning        
> 6 2009-12 12681        1014.54 1014.54 1014.54         1      46      46       650      72       1 France  Afternoon  
> ```
> 
> 6. **계절**: 계절에 따라 재구매 확률이 다를 수 있음
> ```r
> ### 6. 계절 변수 추가
> df_mart <- df_mart %>% left_join(
>   df_origin_sample %>% 
>     mutate(month = month(InvoiceDate),
>            season = case_when(
>              month %in% c(3,4,5) ~ "Spring",
>              month %in% c(6,7,8) ~ "Summer",
>              month %in% c(9,10,11) ~ "Autumn",
>              TRUE ~ "Winter"
>            )) %>% 
>     group_by(bsym, CustomerID) %>% 
>     reframe(season = first(season)),
>   by = c("bsym", "CustomerID")
> )
> df_mart %>% head()
> ```
> ```
> # A tibble: 6 × 15
>   bsym    CustomerID total_amt max_amt min_amt total_cnt max_cnt min_cnt total_qty max_qty min_qty Country peak_time  season
>   [chr]   [chr]          [dbl]   [dbl]   [dbl]     [int]   [int]   [int]     [int]   [int]   [int] [chr]   [chr]      [chr] 
> 1 2009-12 12437         578.28  578.28  578.28         1      27      27       263      24       3 France  Afternoon   Winter
> 2 2009-12 12523          74.56   74.56   74.56         1       4       4        62      36       4 France  Afternoon   Winter
> 3 2009-12 12539        5149.06 2583.23 2565.83         2     104     103      2128      48       2 Spain   Afternoon   Winter
> 4 2009-12 12557        1952.64 1952.64 1952.64         1       3       3       576     216     144 Spain   Morning     Winter
> 5 2009-12 12664         549.08  549.08  549.08         1       4       4       134      72       2 Finland Morning     Winter
> 6 2009-12 12681        1014.54 1014.54 1014.54         1      46      46       650      72       1 France  Afternoon   Winter
> ```
> 
> 7. **구매 빈도**: 구매 빈도가 높은 고객은 재구매 확률이 높을 수 있음
> ```r
> df_mart <- df_mart %>% left_join(
>   df_origin_sample %>% 
>     group_by(bsym, CustomerID) %>% 
>     reframe(cnt = n_distinct(InvoiceNo)) %>% 
>     mutate(
>       tmp_date = as.Date(paste0(bsym, "-01")),
>       days = as.integer(day(floor_date(tmp_date + months(1), "month") - 1)),
>       freq = cnt / days
>     ) %>% 
>     select(-c(cnt, tmp_date, days)),
>   by = c("bsym", "CustomerID")
> )
> 
> df_mart %>% head()
> ```
> ```
> # A tibble: 6 × 16
>   bsym    CustomerID total_amt max_amt min_amt total_cnt max_cnt min_cnt total_qty max_qty min_qty Country peak_time season      freq
>   [chr]   [chr]          [dbl]   [dbl]   [dbl]     [int]   [int]   [int]     [int]   [int]   [int] [chr]   [chr]       [chr]      [dbl]
> 1 2009-12 12437         578.28  578.28  578.28         1      27      27       263      24       3 France  Afternoon   Winter 0.0322581
> 2 2009-12 12523          74.56   74.56   74.56         1       4       4        62      36       4 France  Afternoon   Winter 0.0322581
> 3 2009-12 12539        5149.06 2583.23 2565.83         2     104     103      2128      48       2 Spain   Afternoon   Winter 0.0645161
> 4 2009-12 12557        1952.64 1952.64 1952.64         1       3       3       576     216     144 Spain   Morning     Winter 0.0322581
> 5 2009-12 12664         549.08  549.08  549.08         1       4       4       134      72       2 Finland Morning     Winter 0.0322581
> 6 2009-12 12681        1014.54 1014.54 1014.54         1      46      46       650      72       1 France  Afternoon   Winter 0.0322581
> ```
> 
> 8. **평균 구매금액**: 평균 구매 금액에 따라 재구매 확률이 다를 수 있음
> 	- 이 변수는 `total_amt`를 `total_cnt`로 단순히 나눈 값
> ```r
> df_mart <- df_mart %>% 
>   mutate(avg_amt = total_amt / total_cnt)
> 
> df_mart %>% head()
> ```
> ```
> # A tibble: 6 × 17
>   bsym    CustomerID total_amt max_amt min_amt total_cnt max_cnt min_cnt total_qty max_qty min_qty Country peak_time season      freq avg_amt
>   [chr]   [chr]          [dbl]   [dbl]   [dbl]     [int]   [int]   [int]     [int]   [int]   [int] [chr]   [chr]       [chr]      [dbl]   [dbl]
> 1 2009-12 12437         578.28  578.28  578.28         1      27      27       263      24       3 France  Afternoon   Winter 0.0322581  578.28
> 2 2009-12 12523          74.56   74.56   74.56         1       4       4        62      36       4 France  Afternoon   Winter 0.0322581   74.56
> 3 2009-12 12539        5149.06 2583.23 2565.83         2     104     103      2128      48       2 Spain   Afternoon   Winter 0.0645161 2574.53
> 4 2009-12 12557        1952.64 1952.64 1952.64         1       3       3       576     216     144 Spain   Morning       Winter 0.0322581 1952.64
> 5 2009-12 12664         549.08  549.08  549.08         1       4       4       134      72       2 Finland Morning       Winter 0.0322581  549.08
> 6 2009-12 12681        1014.54 1014.54 1014.54         1      46      46       650      72       1 France  Afternoon    Winter 0.0322581 1014.54
> ```

이렇게 만든 Mart에 key(`bsym` & `CustomerID`)를 이용해 `target` 변수와 병합해 저장하자.
```r
df_mart <- df_mart %>% left_join( df_all_sample %>% select(-key), by = [c](https://rdrr.io/r/base/c.html)("bsym", "CustomerID") )
df_mart %>% dim()
```
```
[1] 7495 17
```

## 04. Build the recipe & workflow

### 04-01. Pre-processing data with recipes

Tidymodels 프레임워크에서 모델링을 하려면 우선 모델링 대상이 되는 데이터(여기선 `df_mart`)가 모델별 적절한 recipe(모델링 전에 데이터에 적용되는 feature engineering 및 전처리)를 가져야 하고, 모델의 명세/사양(specification; spec)이 정의된 workflow가 필요하다. 이를 위해 우선 앞서 구축한 Data Mart인 `df_mart`를 살펴보고 적절한 recipe를 만들어보자.

> [!example]- Libraries load
> ```r
> suppressPackageStartupMessages({
>   library(tidyverse)  # contains dplyr, ggplot and our data set
>   library(vip)        # variable importance plots
>   library(finetune)   # package for more advanced hyperparameter tuning  
>   library(doParallel) # parallelisation package   
>   library(tictoc)     # measure how long processes take
>   library(tidymodels) # the main tidymodels packages all in one place - loaded last to overwrite any conflicts
>   library(ggsci)
>   library(patchwork)
>   library(moments)
>   library(GGally)
>   library(corrplot)
>   library(bonsai)
>   library(workflowsets)
> })
> tidymodels_prefer()
> ```


- 우선 국적 `Country`에 대해 살펴보면 대부분이 영국임을 알 수 있다. 따라서 영국을 제외한 나머지 값들을 기타(`'ETC'`) 국가로 변경하자.

> [!note]- code fold
> ```r
> df_mart %>% 
>   ggplot(aes(x=Country, fill = factor(target))) +
>   geom_bar(position = "dodge") +
>   theme_minimal() +
>   coord_flip() +
>   labs(fill = "Target", x = "Country", y = "Count")
> ```

![[Pasted image 20240220143008.png|center]]

```r
df_mart %>% 
  count(Country) %>% 
  arrange(desc(n)) %>% 
  mutate(ratio = n/sum(n))
```
```
# A tibble: 33 × 3
   Country            n      ratio
   <fct>          <int>      <dbl>
 1 United Kingdom  6871 0.916744  
 2 Germany          150 0.0200133 
 3 France           144 0.0192128 
 4 Belgium           38 0.00507005
 5 Spain             33 0.00440294
 6 Australia         25 0.00333556
 7 Italy             25 0.00333556
 8 Netherlands       23 0.00306871
 9 Switzerland       23 0.00306871
10 Portugal          22 0.00293529
# ℹ 23 more rows
# ℹ Use `print(n = ...)` to see more rows
```
```r
# UK 이외의 기타 국가로 처리
df_mart <- df_mart %>% 
  mutate(Country = ifelse(Country=="United Kingdom", "UK", "ETC")) 
```


 - 문자형 변수 3개 `factor`형으로 변환해주기
 ```r
 df_mart <- df_mart %>% 
 mutate(across(c(Country, peak_time, season), factor))
 ```

#### Splitting our data

주어진 데이터 `df_mart`의 `target` 값을 층(`strata`)으로 지정해서 test set의 size를 30%로 하여 데이터를 분할 하자:

```r
set.seed(123)

#### Create train-test set
split <- initial_split(df_mart, prop = 0.7, strata = target)
#### Create folds to for resampling on the Train set
train <- training(split)
test <- testing(split)
split
```
```
<Training/Testing/Total>
<5245/2250/7495>
```

- training set과 test set의 `target` 비율을 살펴보면 동일한 것을 확인할 수 있음:
```
# Confirm distribution is the same
> summary(train$target)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
 0.0000  0.0000  0.0000  0.3813  1.0000  1.0000 
> summary(test$target)
   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
 0.0000  0.0000  0.0000  0.3813  1.0000  1.0000
```

#### Explore the data

이제 `train` 데이터를 이용해 적절한 pre-processing 방법을 찾아보자. 이때 `skimr` 패키지의 `skim()` 함수를 이용해 변수별 간략한 정보를 확인할 수 있다:

> [!example]- `skim()`을 통한 변수별 정보
> ```
> > skimr::skim(train %>% select(-c(bsym, CustomerID)))
> ── Data Summary ────────────────────────
>                            Values                      
> Name                       train %>% select(-c(bsym,...
> Number of rows             5245                        
> Number of columns          15                          
> _______________________                                
> Column type frequency:                                 
>   factor                   3                           
>   numeric                  12                          
> ________________________                               
> Group variables            None                        
> 
> ── Variable type: factor ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
>   skim_variable n_missing complete_rate ordered n_unique top_counts                               
> 1 Country               0             1 FALSE          2 UK: 4799, ETC: 446
> 2 peak_time             0             1 FALSE          3 Aft: 3408, Mor: 1737, Eve: 100           
> 3 season                0             1 FALSE          4 Aut: 1790, Sum: 1231, Spr: 1229, Win: 995
> 
> ── Variable type: numeric ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
>    skim_variable n_missing complete_rate        mean           sd        p0         p25         p50         p75         p100 hist 
>  1 total_amt             0             1 679.238     1804.28      0         211.35      346.45      624.58      51031.7      ▇▁▁▁▁
>  2 max_amt               0             1 517.340     1258.98      0         196.93      319.19      517.53      44051.6      ▇▁▁▁▁
>  3 min_amt               0             1 352.274      407.216     0         141.7       272.95      405.3        7544.91     ▇▁▁▁▁
>  4 total_cnt             0             1   1.46806      1.25913   1           1           1           2            19        ▇▁▁▁▁ 
>  5 max_cnt               0             1  24.6982      23.1251    1          10          18          31           250        ▇▁▁▁▁ 
>  6 min_cnt               0             1  19.7607      20.7346    1           6          15          25           220        ▇▁▁▁▁ 
>  7 total_qty             0             1 389.856      943.241     1         102         194         368         26687        ▇▁▁▁▁
>  8 max_qty               0             1  62.6633     173.947     1          16          25          48          7128        ▇▁▁▁▁
>  9 min_qty               0             1   8.33136     60.1779    1           1           2           3          2000        ▇▁▁▁▁
> 10 freq                  0             1   0.0482555    0.0413473 0.0322581   0.0322581   0.0333333   0.0645161     0.633333 ▇▁▁▁▁
> 11 avg_amt               0             1 418.745      553.660     0         180.93      306         458.343     13305.5      ▇▁▁▁▁
> 12 target                0             1   0.381316     0.485756  0           0           0           1             1        ▇▁▁▁▅
> ```

- 범주형 변수는 3개이고, 수치형 변수는 `target` 제외 11개가 있다. 데이터를 보면 수치형 변수들이 모두 오른쪽 꼬리(right-skewed)를 가지고 있다. Recipe를 구성하기 전에 우선 범주형 변수의 각 수준별로 `target`의 비율이 어떻게 다른지 살펴보자:

> [!note]- code fold
> ```r
> categorical_vars <- train %>% 
>   select(-c(bsym, CustomerID)) %>% 
>   select_if(is.factor) %>% 
>   names()
> ## Categorical variables
> plot_categorical <- function(cat){
>   cat <- ensym(cat)
>   train %>%
>     group_by(!!cat) %>%
>     reframe(prop = mean(target, na.rm = T)) %>% 
>     mutate(across(!!cat, ~reorder(.x, prop))) %>% 
>     ggplot(aes(x = !!cat, y = prop, fill = !!cat)) +
>     geom_col(color = 'black', alpha = 0.6) +
>     geom_text(aes(label = scales::percent(prop, accuracy = 0.01)), vjust  = 1.5) +
>     scale_color_nejm() +
>     scale_fill_nejm() + 
>     ggtitle(paste0("Target rate by level of ", as.character(cat))) +
>     guides(fill = "none") +
>     theme_light()
> }
> cat_p <- map(categorical_vars, plot_categorical)
> cat_p[[1]] + cat_p[[2]] + cat_p[[3]]
> ```

![[Pasted image 20240220143554.png|center|500]]
`skim()`을 통해 보았을 때 범주형 변수의 각 수준별 도수에는 큰 차이를 보였으나, 각 수준별 재구매(`target`)의 비율은 약간의 차이만 보이고 있다. 하지만 수준별로 어느 정도 비율에서 차이를 보이기 때문에 범주형 변수가 `target`의 영향을 미칠 수 있을 것으로 판단된다. 

이제 수치형 변수에 대해 살펴보자. 앞서 `skim()`을 통해 11개의 수치형 변수들의 간단한 히스토그램과 사분위수를 보아 상당히 right-skewed 되어 있으며, 값이 굉장히 큰 이상치가 많았다. 

- 각 수치형 변수에 대해 왜도를 계산해보자:
	- 월별 최소 구매수량 `min_qty`가 가장 긴 우측 꼬리를 가지고 있음
> [!note]- code fold
> ```r
> #### 수치형 변수에 대해 왜도 계산
> numeric_vars <- train %>% 
>   select(-c(bsym, CustomerID)) %>% 
>   select_if(is.numeric) %>% 
>   select(-target) %>% 
>   names()
> 
> num_skew <- map(numeric_vars, ~ skewness(train %>% pull(.x))) %>% 
>   unlist()
> names(num_skew) <- numeric_vars
> ```
```
> num_skew

total_amt   max_amt   min_amt total_cnt   max_cnt   min_cnt total_qty   max_qty   min_qty      freq   avg_amt 
15.568941 20.722585  5.659813  6.242186  2.523002  2.640569 11.841334 19.602134 21.010613  6.224407  9.399795
```

- 아래와 같이 수치형 변수의 대략적인 분포를 보면 음의 값을 가지는 않지만 최소값으로 0을 갖는 변수들이 존재하므로 Box-Cox 변환 대신 [Yeo-Johnson 변환](https://en.wikipedia.org/wiki/Power_transform)을 적용하는 것이 좋아보임
```r
map_dfr(numeric_vars, ~summary(train %>% pull(.x))) %>% 
  mutate(var = numeric_vars, .before = Min.)
```
```
# A tibble: 11 × 7
   var       Min.        `1st Qu.`    Median       Mean         `3rd Qu.`    Max.        
   <chr>     <table[1d]> <table[1d]>  <table[1d]>  <table[1d]>  <table[1d]>  <table[1d]> 
 1 total_amt 0.00000000  211.35000000 346.45000000 679.23801849 624.58000000 5.103173e+04 
 2 max_amt   0.00000000  196.93000000 319.19000000 517.34002212 517.53000000 4.405160e+04 
 3 min_amt   0.00000000  141.70000000 272.95000000 352.27419333 405.30000000 7.544910e+03 
 4 total_cnt 1.00000000    1.00000000   1.00000000   1.46806482   2.00000000 1.900000e+01 
 5 max_cnt   1.00000000   10.00000000  18.00000000  24.69818875  31.00000000 2.500000e+02 
 6 min_cnt   1.00000000    6.00000000  15.00000000  19.76072450  25.00000000 2.200000e+02 
 7 total_qty 1.00000000  102.00000000 194.00000000 389.85605338 368.00000000 2.668700e+04 
 8 max_qty   1.00000000   16.00000000  25.00000000  62.66329838  48.00000000 7.128000e+03 
 9 min_qty   1.00000000    1.00000000   2.00000000   8.33136320   3.00000000 2.000000e+03
10 freq      0.03225806    0.03225806   0.03333333   0.04825553   0.06451613 6.333333e-01
11 avg_amt   0.00000000  180.93000000 306.00000000 418.74534443 458.34285714 1.330550e+04
```

우선 변수 변환 등의 전처리 작업을 하기 전에 모델의 formula만을 가지고 있는 basic recipe(`basic_rec`)을 정의하자.

- `basic_rec`에 `data_mart` 구성 시 key 변수의 역할을 했던 `bsym`과 `CustomerID`를 `"ID"` 역할로 지정함으로써 모델링 때 이 변수들을 제거하지 않고 진행할 수 있음
```r
basic_rec <- recipe(formula = target ~ ., data = train) 
basic_rec <- basic_rec %>% 
	update_role(bsym, CustomerID, new_role = "ID")
```
```
> basic_rec

── Recipe ──────────────────────────────────────────────────────────────────────────────────────────────────────

── Inputs 
Number of variables by role
outcome:    1
predictor: 14
ID:         2
```

- Yeo-Johnson 변환 전후의 분포 비교:

> [!note]- code fold
> ```r
> before_trans <- function(num){
>   num <- ensym(num)
>   basic_rec %>% 
>     prep(train) %>% 
>     bake(train) %>% 
>     ggplot(aes(x = !!num)) + 
>     geom_histogram(color = 'black', fill = 'skyblue') +
>     theme_light() +
>     theme(axis.title.y = element_blank(),
>           axis.text.y = element_blank(),
>           axis.ticks.y = element_blank())
> }
> before_trans_plot <- map(numeric_vars, before_trans)
> before_trans_plot[[1]]+before_trans_plot[[2]]+before_trans_plot[[3]]+
>   before_trans_plot[[4]]+before_trans_plot[[5]]+before_trans_plot[[6]]+
>   before_trans_plot[[7]]+before_trans_plot[[8]]+before_trans_plot[[9]]+
>   before_trans_plot[[10]]+before_trans_plot[[11]] +
>   plot_layout(ncol = 3) +
>   plot_annotation(title = "Before Transformation",
>                   theme = theme(plot.title = element_text(hjust = 0.5))) 
> ```

![[Pasted image 20240220153122.png|center]]

> [!note]- code fold
> ```r
> after_trans <- function(num){
>   num <- ensym(num)
>   basic_rec %>% 
>     step_YeoJohnson(!!num) %>% # Yeo-Johnson 변환
>     prep(train) %>% 
>     bake(train) %>% 
>     ggplot(aes(x = !!num)) + 
>     geom_histogram(color = 'black', fill = 'skyblue') +
>     theme_light() +
>     theme(axis.title.y = element_blank(),
>           axis.text.y = element_blank(),
>           axis.ticks.y = element_blank())
> }
> after_trans_plot <- map(numeric_vars, after_trans)
> after_trans_plot[[1]]+after_trans_plot[[2]]+after_trans_plot[[3]]+
>   after_trans_plot[[4]]+after_trans_plot[[5]]+after_trans_plot[[6]]+
>   after_trans_plot[[7]]+after_trans_plot[[8]]+after_trans_plot[[9]]+
>   after_trans_plot[[10]]+after_trans_plot[[11]] +
>   plot_layout(ncol = 3) +
>   plot_annotation(title = "After Yeo-Johnson Transformation",
>                   theme = theme(plot.title = element_text(hjust = 0.5)))
> ```

![[Pasted image 20240220153554.png|center]]

그런 다음 Yeo-Johnson 변환 후의 수치형 변수들에 대해 `target`과 어떠한 관계가 있는지 boxplot with violin plot을 살펴보자. 
- 수치형 변수들과 재구매 `target`의 관계를 아래와 같이 단순하게 살펴보았을 때는 눈에 띄는 차이는 파악되지 않음
- 하지만 각 변수별로 이상치로 판단되는 값이 많기 때문에 이에 대해 추가적으로 고려할 필요가 있음
> [!note]- code fold
> ```r
> #### Yeo-Johnson 변환 후 target 변수와의 관계 확인 - boxplot with violin plot
> plot_numerical <- function(num){
>   num <- ensym(num)
>   basic_rec %>% 
>     step_YeoJohnson(all_numeric_predictors()) %>% 
>     prep(train) %>% 
>     bake(train) %>% 
>     mutate(target = factor(target)) %>% 
>     ggplot(aes(x = target, y = !!num, fill = target)) +
>     geom_violin(width=1) +
>     geom_boxplot(width = 0.3, alpha=0.7) +
>     # scale_fill_viridis(discrete = TRUE) +
>     coord_flip() + 
>     theme_light() +
>     guides(color = 'none') +
>     theme(legend.position = "none")
> }
> num_p <- map(numeric_vars, plot_numerical)
> num_p[[1]]+num_p[[2]]+num_p[[3]]+
>   num_p[[4]]+num_p[[5]]+num_p[[6]]+
>   num_p[[7]]+num_p[[8]]+num_p[[9]]+
>   num_p[[10]]+num_p[[11]] + 
>   plot_layout(ncol = 3)
> ```

![[Pasted image 20240220155325.png|center]]

이제 변수 간 상관관계를 살펴보자. 데이터에는 범주형 변수도 있으므로 피어슨 상관계수만을 사용할 수 없다. 대신 다른 변수 중 하나를 입력으로 해서 수치형 변수 하나를 예측하는 선형 모델을 훈련시킨 다음 설명 분산을 측정해 그것의 제곱근을 구하는 방식을 적용할 수 있다. 이는 숫자형 변수를 입력으로 할 때 피어슨 상관계수의 절대값과 동일하다. 중요한 것은 입력으로 범주형 변수도 사용할 수 있다는 것이다.
- 구매빈도 `freq`는 구매 건수 `total_cnt`를 월별 일수로 단순히 나누어 계산한 값이므로 상관계수가 거의 1로 계산됨
- `total_cnt`는 제거하고 진행
> [!note]- code fold
> ```r
> #### 변수별 상관관계 - ANOVA 설명 분산량
> mycor <- function(cnames, dat){
>   x.num <- dat %>% pull(cnames[1])
>   x.cat <- dat %>% pull(cnames[2])
>   
>   suppressWarnings({
>     av <- anova(lm(x.num ~ x.cat))
>   })
>   sqrt(av$`Sum Sq`[1] / sum(av$`Sum Sq`))
> }
> 
> cnames <- basic_rec %>% 
>   step_YeoJohnson(all_numeric_predictors()) %>% 
>   prep(train) %>% 
>   bake(train) %>% 
>   mutate(target = factor(target)) %>% 
>   select_if(is.numeric) %>% names()
> combs <- expand.grid(y = cnames, 
>                      x = setdiff(names(train %>% select(-c(bsym, CustomerID))), "target"))
> combs$cor <- apply(combs, 1, mycor, 
>                   dat = basic_rec %>% 
>                     step_YeoJohnson(all_numeric_predictors()) %>% 
>                     prep(train) %>% 
>                     bake(train) )
> combs$lab <- sprintf("%.2f", combs$cor)
> forder <- c(cnames, setdiff(unique(combs$x), cnames))
> 
> combs <- combs %>% 
>   mutate(x = factor(x, levels = forder),
>          y = factor(y, levels = rev(cnames)))
> 
> combs %>% 
>   ggplot(aes(x = x, y = y, fill = cor, label = lab)) +
>   geom_tile() +
>   geom_label(fill = "white", size = 3) +
>   theme_light() +
>   theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
>   scale_x_discrete("") +
>   scale_y_discrete("") +
>   scale_fill_viridis("Variance\nexplained", begin = 0.2)
> ```

![[Pasted image 20240220161852.png|center]]

```r
#### total_cnt 제거
basic_rec <- basic_rec %>% 
  step_rm(total_cnt)
```


#### Dummy encoding

