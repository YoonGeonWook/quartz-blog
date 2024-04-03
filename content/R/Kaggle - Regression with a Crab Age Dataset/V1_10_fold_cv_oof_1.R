source("./V1.R")
cat("\n")
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

results = list(gbm_cv_scores = gbm_cv_scores, gbm_preds = gbm_preds, 
               xgb_cv_scores = xgb_cv_scores, xgb_preds = xgb_preds, 
               lgbm_cv_scores = lgbm_cv_scores, lgbm_preds = lgbm_preds,
               cb_cv_scores = cb_cv_scores, cb_preds = cb_preds, 
               ens_cv_scores = ens_cv_scores, ens_preds = ens_preds) 
write_rds(results, file = "oof_mae_1.rds")