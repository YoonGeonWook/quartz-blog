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
})
tidymodels_prefer()

df_mart <- df_mart %>% 
  mutate(across(c(Country, peak_time, season), factor))