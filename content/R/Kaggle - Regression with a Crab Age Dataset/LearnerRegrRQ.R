#' @title Quantile Regression Learner
#' 
#' @name mlr_learners_regr.rq
#' 
#' @description
#' Quantile regression
#' Calls [quantreg::rq()].
#' 
#' @templateVar id regr.rq
#' @template learner
#' 
#' @export
#' @template seealso_learner
#' @template example
LearnerRegrRQ = R6::R6Class("LearnerRegrRQ",
  inherit = mlr3::LearnerRegr,
  
  public = list(
    #' @description
        #' Create a new instance of this [R6][R6::R6Class] class.
    initialize = function() {
      ps = ps(
        tau           = p_dbl(default = 0.5, lower = 0, upper = 1, tags = "train"),
        model         = p_lgl(default = TRUE, tags = "train"),
        x             = p_lgl(default = FALSE, tags = "train"),
        y             = p_lgl(default = FALSE, tags = "train"),
        contrasts     = p_uty(tags = "train"),
        method        = p_fct(c("br", "fn", "pfn", "bdp", "lasso"), default = "br", tags = "train"),
        se            = p_fct(c("iid", "nid", "ker", "boot"), default = "iid", tags = "predict"),
        verbose       = p_lgl(default = FALSE, tags = "predict")
      )
      
      super$initialize(
        id = "regr.rq",
        param_set = ps,
        predict_types = c("response", "se"),
        feature_types = c("logical", "integer", "numeric", "factor", "character"),
        properties = c("weights"),
        packages = c("mlr3learners", "quantreg"),
        label = "Quantile Regression",
        man = "mlr3learners::mlr_learners_regr.rq"
      )
    } 
  ),
  
  private = list(
    .train = function(task) {
      pv = self$param_set$get_values(tags = "train")
      if ("weights" %in% task$properties) {
        pv$weights = task$weights$weight
      }
      
      # method 인자가 필요하지 않거나 올바른 값이 설정되었는지 확인
      args_list = list(formula = task$formula(), data = task$data(), tau = pv$tau)
      if (!is.null(pv$method) && pv$method %in% c("fn", "fnb", "br")) {
        args_list$method = pv$method
      }
      if (!is.null(pv$model)) {
        args_list$model = pv$model
      }
      if (!is.null(pv$x)) {
        args_list$x = pv$x
      }
      if (!is.null(pv$y)) {
        args_list$y = pv$y
      }
      
      # quantreg::rq() 함수에 필요한 인자들만 전달
      self$model = do.call(quantreg::rq, args_list)
    },
    
    .predict = function(task) {
      pv = self$param_set$get_values(tags = "predict")
      newdata = ordered_features(task, self)
      se_fit = self$predict_type == "se"
      prediction = invoke(predict, object = self$model, newdata = newdata, se.fit = se_fit, .args = pv)
      
      if (se_fit) {
        list(response = unname(prediction$fit), se = unname(prediction$se.fit))
      } else {
        list(response = unname(prediction))
      }
    }
  )
)

ordered_features = function(task, learner) {
  cols = names(learner$state$data_prototype) %??% learner$state$feature_names
  task$data(cols = intersect(cols, task$feature_names))
}


