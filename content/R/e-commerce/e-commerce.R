library(tidyverse)

# 02. Data Readiness Check & Sampling
## 02-01. Data info Check 
df <- read_csv("../../data/e-commerce/Online Retail.csv", col_type = "ccciTdcc")
df %>% head()

df %>% glimpse()

### 결측치 확인
df %>% is.na() %>% colSums()

df <- df %>% drop_na("CustomerID")
df %>% is.na() %>% colSums()

### Outlier 확인
df %>% 
  select_if(is.numeric) %>% 
  summary()

### Quantity가 음수인 데이터 확인: 반품/회수 물량일 수 있음
df %>% filter(Quantity < 0)
### Quantity 음수값 제거
df <- df %>% 
  filter(Quantity > 0)

### 중복 데이터 확인
df %>% 
  mutate(duplicated = duplicated(.)) %>% 
  count(duplicated)

df %>% 
  filter(duplicated(df) | duplicated(df, fromLast = T))

### 중복 삭제 후 재확인
df <- df %>% distinct()
df %>% 
  mutate(duplicated = duplicated(.)) %>% 
  count(duplicated)
  
### 1,067,370 → 779,494
df %>% dim()

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
df %>% dim()
df %>% head()

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
df_all %>% head()


### (2) Target Ratio 확인
#### bsym 기준 target ratio:
options(pillar.sigfig = 6)
df_target <- df_all %>% 
  group_by(bsym) %>% 
  reframe(total_y = sum(target),
          count_y = n(),
          ratio = total_y/count_y)
df_target %>% 
  print(n = nrow(.))

#### 2011-12에 해당하는 데이터 제외
df_target <- df_target %>% filter(bsym != '2011-12')
df_all <- df_all %>% filter(bsym != '2011-12')
df_target$ratio %>% summary()

df_all$target %>% mean()

#### 마지막 행에 합계 추가
df_target <- df_target %>% 
  bind_rows(
    tibble(bsym = "total",
           total_y = sum(.$total_y),
           count_y = sum(.$count_y),
           ratio = 1)  
  )

## 02-03. Data Sampling
df_all %>% dim()
#### 전체 데이터의 30%: 7,495개
round(nrow(df_all) * 0.3)


#### 모집단 층 크기: Nh, 표본 층 크기: nh
df_target <- df_target %>% 
  mutate(N = nrow(df_all)) %>% 
  mutate(n = round(nrow(df_all) * 0.3)) %>% 
  relocate(count_y, .after = n) %>% 
  rename(Nh = count_y) %>% 
  mutate(nh = round(n*Nh/N)) %>% 
  group_by(bsym)

df_target %>% print(n=25)
df_target$nh[-25] %>% sum()

# 반올림 때문에 표본 층 크기가 n=7495 보다 1개 더 크므로 
# 2011년 11월 표본 크기를 한 개 줄여주자
df_target$nh[df_target$bsym=='2011-11'] <- df_target$nh[df_target$bsym=='2011-11'] - 1

set.seed(123)
ord <- unique(df_target$bsym)
units <- sampling::strata(df_all, stratanames = "bsym", size = df_target$nh[1:24], method="srswor")

df_all_sample <- df_all %>% 
  slice(units$ID_unit)
df_all_sample %>% dim()

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
tt <- df_tmp %>% left_join(df_tmp2, by='bsym') %>% 
  gt() %>% 
  gt_theme_nytimes() %>% 
  tab_header(title = "bsym별 target 비율") %>% 
  gt_plt_bar_stack(column = target_ratio, labels = c("target_ratio", "target_ratio_pop"), palette = c("skyblue", "hotpink"))
tt %>% str()

