# 4.1 Bike Rental

# Assign libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(mlr)
})

day_diff <- function(date1, date2){
  as.numeric(difftime(as.Date(date1), as.Date(date2), units = 'days'))
}

year_diff <- function(date1, date2){
  day_diff(date1, date2) / 365.25
}

bike <- read.csv("./data/bike-sharing-daily.csv", stringsAsFactors = F)

bike <- bike %>% 
  mutate(weekday = factor(weekday, levels = 0:6, labels = c('SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT')),
         holiday = factor(holiday, levels = c(0,1), labels = c('NO HOLIDAY', 'HOLIDAY')),
         workingday = factor(workingday, levels = c(0,1), labels = c('NO WORKING DAY', 'WORKING DAY')),
         season = factor(season, levels = 1:4, labels = c('WINTER', 'SPRING', 'SUMMER', 'FALL')),
         weathersit = factor(weathersit, levels = 1:3, labels = c('GOOD', 'MISTY', 'RAIN/SNOW/STORM')),
         mnth = factor(mnth, levels = 1:12, labels = c('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC')),
         yr = ifelse(yr == 0, 2011, 2012),
         days_since_2011 = day_diff(dteday, min(as.Date(dteday)))) %>% 
  # Denormalize weather features:
  # temp : Normalized temperature in Celsius.
  # The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
  mutate(temp = temp * (39 - (-8)) + (-8)) %>% 
  # atemp: Normalized feeling temperature in Celsius.
  # The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
  mutate(atemp = atemp * (50-(16)) + (16)) %>% 
  # windspeed: Normalized wind speed. 
  # The values are divided to max 67.
  mutate(windspeed = windspeed * 67) %>% 
  # hum: Normalized humidity. 
  # The values are divided to 100 (max)
  mutate(hum = hum * 100) %>% 
  select(-c(instant, dteday, registered, casual))

# get.bike.task = function(data_dir){
#   mlr::makeRegrTask(id='bike', data=get.bike.data(data_dir), target = 'cnt')
# }

bike.features.of.interest = c('season','holiday', 'workingday', 'weathersit', 'temp', 'hum', 'windspeed', 'days_since_2011')

bike %>% head()


# 4.2 YouTube Spam data
load("./data/ycomments.RData")
ycomments %>% str()
ycomments$CONTENT %>% nchar() %>% summary()

tube <- read.csv("./data/TubeSpam.csv")
tube %>% str()

# 4.3 Risk Factors for Cervical Cancer data
cervical <- read.csv("./data/risk_factors_cervical_cancer.csv", na.strings = c("?"), stringsAsFactors = F)
cervical <- cervical %>% 
  select(-c(Citology, Schiller, Hinselmann)) %>% 
  mutate(Biopsy = factor(Biopsy, levels = c(0,1), labels = c("Healthy", "Cancer"))) %>% 
  # subset variables to the ones that should be used in the book
  select(Age, Number.of.sexual.partners, First.sexual.intercourse,
         Num.of.pregnancies, Smokes, Smokes..years., Hormonal.Contraceptives, Hormonal.Contraceptives..years.,
         IUD, IUD..years., STDs, STDs..number., STDs..Number.of.diagnosis, STDs..Time.since.first.diagnosis,
         STDs..Time.since.last.diagnosis, Biopsy)

## NA imputation
imputer = mlr::imputeMode()

cervical_impute <- cervical %>% 
  mlr::impute(classes = list(numeric = imputeMode()))
cervical <- cervical_impute$data

cervical %>% str()
