library(tidyverse)

# 02. Data Readiness Check & Sampling
## 02-01. Data info Check 
df <- read_csv("../../data/e-commerce/Online Retail.csv", col_type = "ccciTdcc")
# df %>% head()

# df %>% glimpse()

### 결측치 확인
# df %>% is.na() %>% colSums()

df <- df %>% drop_na("CustomerID")
# df %>% is.na() %>% colSums()

### Outlier 확인
# df %>% 
#   select_if(is.numeric) %>% 
#   summary()

### Quantity가 음수인 데이터 확인: 반품/회수 물량일 수 있음
# df %>% filter(Quantity < 0)
### Quantity 음수값 제거
df <- df %>% 
  filter(Quantity > 0)

### 중복 데이터 확인
# df %>% 
#   mutate(duplicated = duplicated(.)) %>% 
#   count(duplicated)

# df %>% 
#   filter(duplicated(df) | duplicated(df, fromLast = T))

### 중복 삭제 후 재확인
df <- df %>% distinct()
# df %>% 
#   mutate(duplicated = duplicated(.)) %>% 
#   count(duplicated)
  
### 1,067,370 → 779,494
# df %>% dim()

## 02-02. Data Readiness Check
library(lubridate)

### 기준년월 변수 생성: bsym - %Y-%m 형식
df <- df %>% 
  mutate(bsym = format(InvoiceDate, "%Y-%m"))
### 원본 데이터 저장: df_origin
df_origin <- df
# 데이터 적재 기간 확인: 2009-12-01 ~ 2011-12-09 (약 2년)
min(df$InvoiceDate); max(df$InvoiceDate)

df <- df %>% 
  distinct(bsym, CustomerID)
# df %>% dim()
# df %>% head()

# 주어진 bsym에 구매 고객이 내월 구매 고객일 경우 1, 그렇지 않으면 0 값을 갖는 target 변수 생성: process_bsym()
process_bsym <- function(bsym_value, df) {
  df_left <- filter(df, bsym == bsym_value)
  bsym_1m <- ymd(paste0(bsym_value, "-01")) %m+% months(1) %>% format("%Y-%m")
  df_right <- df %>% 
    filter(bsym == bsym_1m) %>% 
    distinct(CustomerID) %>% 
    mutate(target = 1)
  
  df_merge <- left_join(df_left, df_right, by = "CustomerID") %>% 
    mutate(target = ifelse(is.na(target), 0, target))
  
  return(df_merge)
}

# 모든 bsym 값에 대해 process_bsym 함수 적용
df_all <- map_df(unique(df$bsym), ~process_bsym(.x, df))
# df_all %>% head()


### (2) Target Ratio 확인
#### bsym 기준 target ratio:
options(pillar.sigfig = 6)
df_target <- df_all %>% 
  group_by(bsym) %>% 
  reframe(total_y = sum(target),
          count_y = n(),
          ratio = total_y/count_y)
# df_target %>% 
#   print(n = nrow(.))

#### 2011-12에 해당하는 데이터 제외
df_target <- df_target %>% filter(bsym != '2011-12')
df_all <- df_all %>% filter(bsym != '2011-12')
# df_target$ratio %>% summary()

# df_all$target %>% mean()

#### 마지막 행에 합계 추가
df_target <- df_target %>% 
  bind_rows(
    tibble(bsym = "total",
           total_y = sum(.$total_y),
           count_y = sum(.$count_y),
           ratio = 1)  
  )

## 02-03. Data Sampling
# df_all %>% dim()
#### 전체 데이터의 30%: 7,495개
# round(nrow(df_all) * 0.3)


#### 모집단 층 크기: Nh, 표본 층 크기: nh
df_target <- df_target %>% 
  mutate(N = nrow(df_all)) %>% 
  mutate(n = round(nrow(df_all) * 0.3)) %>% 
  relocate(count_y, .after = n) %>% 
  rename(Nh = count_y) %>% 
  mutate(nh = round(n*Nh/N)) %>% 
  group_by(bsym)

# df_target %>% print(n=25)
# df_target$nh[-25] %>% sum()

# 반올림 때문에 표본 층 크기가 n=7495 보다 1개 더 크므로 
# 2011년 11월 표본 크기를 한 개 줄여주자
df_target$nh[df_target$bsym=='2011-11'] <- df_target$nh[df_target$bsym=='2011-11'] - 1

set.seed(123)
ord <- unique(df_target$bsym)
units <- sampling::strata(df_all, stratanames = "bsym", size = df_target$nh[1:24], method="srswor")

df_all_sample <- df_all %>% 
  slice(units$ID_unit)
# df_all_sample %>% dim()

library(gt)
library(gtExtras)
df_tmp <- df_all_sample %>% 
  group_by(bsym) %>% 
  reframe(sum_y = sum(target),
          nh = n())

df_tmp2 <- df_tmp %>% 
  mutate(target_ratio = sum_y/nh,
         target_ratio_pop = df_target$ratio[1:24]) %>% 
  pivot_longer(cols=c(target_ratio, target_ratio_pop), names_to = 'name') %>% 
  group_by(bsym) %>% 
  reframe(target_ratio = list(value))
# df_tmp %>% left_join(df_tmp2, by='bsym') %>% 
#   gt() %>% 
#   gt_theme_nytimes() %>% 
#   tab_header(title = "bsym별 target 비율") %>% 
#   gt_plt_bar_stack(column = target_ratio, labels = c("target_ratio", "target_ratio_pop"), palette = c("skyblue", "hotpink"))
# 
# df_all_sample$target %>% mean()
# df_all$target %>% mean()

# 03. Data Mart & Feature Engineering
## 03-01. Data Mart 기획 및 설계
# df_origin %>% head()
# df_all_sample %>% head()

#### df_origin에 key 변수 생성
df_origin <- df_origin %>% 
  mutate(key = str_c(bsym, CustomerID))
# df_origin %>% 
#   reframe(n_key = n_distinct(key))

#### df_all_sample에 key 변수 생성
df_all_sample <- df_all_sample %>% 
  mutate(key = str_c(bsym, CustomerID))
# df_all_sample %>% 
#   reframe(n_key = n_distinct(key))


df_origin_sample <- df_origin %>% 
  filter(key %in% df_all_sample$key)

# df_origin과 df_origin_sample의 비율: 대략 30% 정도
# nrow(df_origin_sample)/nrow(df_origin)

## Mart구성
### 구매금액
### 1. 구매금액 amt 관련 변수 total_amt, max_amt, min_amt
#### 1) total_amt: 당월 총 구매금액
df_mart <- df_origin_sample %>%
  mutate(amt = UnitPrice * Quantity) %>% 
  group_by(bsym, CustomerID) %>% 
  reframe(total_amt = sum(amt, na.rm = T))

#### 2) max_amt, min_amt: 당월 송장당 최대/최소 구매금액
df_mart <- df_mart %>% left_join(
  df_origin_sample %>% 
    mutate(amt = UnitPrice * Quantity) %>% 
    group_by(bsym, CustomerID, InvoiceNo) %>% 
    reframe(amt = sum(amt, na.rm = T)) %>% 
    group_by(bsym, CustomerID) %>% 
    reframe(max_amt = max(amt, na.rm = T),
            min_amt = min(amt, na.rm = T)),
  by = c("bsym", "CustomerID")
)

# df_mart %>% head()

### 2. 구매건수 cnt 관련 변수
#### 1) total_cnt: 당월 총 구매건수
df_mart <- df_mart %>% left_join(
  df_origin_sample %>% 
    group_by(bsym, CustomerID) %>% 
    reframe(total_cnt = n_distinct(InvoiceNo)),
  by = c("bsym", "CustomerID")
)
#### 2) max_cnt, min_cnt: InvoiceNo별 최대/최소 구매 품목 수
df_mart <- df_mart %>% 
  left_join(
    df_origin_sample %>% 
      group_by(bsym, CustomerID, InvoiceNo) %>% 
      reframe(cnt = n_distinct(StockCode)) %>% 
      group_by(bsym, CustomerID) %>% 
      reframe(max_cnt = max(cnt, na.rm = T),
              min_cnt = min(cnt, na.rm = T)),
    by = c("bsym", "CustomerID")
  )


# df_mart %>% head()

### 3. 구매수량 qty 관련 변수
#### 1) total_qty: 당월 총 구매수량
#### 2) min/max_qty: 당월 최소/최대 구매수량
df_mart <- df_mart %>% left_join(
  df_origin_sample %>% 
    group_by(bsym, CustomerID) %>% 
    reframe(total_qty = sum(Quantity, na.rm = T),
            max_qty = max(Quantity, na.rm = T),
            min_qty = min(Quantity, na.rm = T)),
  by = c("bsym", "CustomerID")
)

# df_mart %>% head()

### 4. 국적 변수 생성
df_mart <- df_mart %>% left_join(
  df_origin_sample %>% 
    group_by(bsym, CustomerID) %>% 
    reframe(Country = first(Country)),
  by = c("bsym", "CustomerID")
)
# df_mart %>% head()

### 5. 구매 시간대(아침, 점심, 저녁, 밤)
#### 아침: 6~12시, 점심: 12~18시, 저녁: 18~24시, 밤: 0~6시
#### 시간대별 구매 빈도 계산
df_mart <- df_mart %>% left_join(
  df_origin_sample %>% 
    mutate(hour = hour(InvoiceDate),
           peak_time = case_when(
             hour >= 6  & hour < 12 ~ "Morning",
             hour >= 12 & hour < 18 ~ "Afternoon",
             hour >= 18 & hour < 24 ~ "Evening",
             TRUE ~ "Night"
           )) %>% 
    group_by(bsym, CustomerID, peak_time) %>% 
    reframe(purchase_cnt = n()) %>% 
    group_by(bsym, CustomerID) %>% 
    slice_max(purchase_cnt, n = 1, with_ties = FALSE) %>% 
    select(-purchase_cnt) %>% 
    ungroup(),
  by = c("bsym", "CustomerID")
)
# df_mart %>% head()

### 6. 계절 변수 추가
df_mart <- df_mart %>% left_join(
  df_origin_sample %>% 
    mutate(month = month(InvoiceDate),
           season = case_when(
             month %in% c(3,4,5) ~ "Spring",
             month %in% c(6,7,8) ~ "Summer",
             month %in% c(9,10,11) ~ "Autumn",
             TRUE ~ "Winter"
           )) %>% 
    group_by(bsym, CustomerID) %>% 
    reframe(season = first(season)),
  by = c("bsym", "CustomerID")
)
# df_mart %>% head()

### 7. 당월 구매 빈도
#### freq = count(InvoiceNo) / # num of days in month
df_mart <- df_mart %>% left_join(
  df_origin_sample %>% 
    group_by(bsym, CustomerID) %>% 
    reframe(cnt = n_distinct(InvoiceNo)) %>% 
    mutate(
      tmp_date = as.Date(paste0(bsym, "-01")),
      days = as.integer(day(floor_date(tmp_date + months(1), "month") - 1)),
      freq = cnt / days
    ) %>% 
    select(-c(cnt, tmp_date, days)),
  by = c("bsym", "CustomerID")
)

# df_mart %>% head()

### 8. 평균 구매금액: avg_amt
#### 송장당 평균 구매금액
df_mart <- df_mart %>% 
  mutate(avg_amt = total_amt / total_cnt)

df_mart %>% head()

df_mart <- df_mart %>% left_join(
  df_all_sample %>% select(-key),
  by = c("bsym", "CustomerID")
)

# df_mart %>% head()
# df_mart %>% dim()


