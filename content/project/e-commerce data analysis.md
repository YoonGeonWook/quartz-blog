---
sticker: emoji//1f525
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


<iframe src="table1.html"></iframe>