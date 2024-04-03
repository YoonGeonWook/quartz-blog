#' Custom {skimr} summary stats
#'
#' `my_skim` stands for _summary_. Returns different summary stats for different classes, distributions (unicode) for numeric class, raw data of first and last parts, etc.
#'
#' @details `my_skim` and `my_skim2` are the same except that `my_skim2` provides extra _missingness_ functions `skimr::n_missing` and `skimr::complete_rate`.
#'
#' @param n_bins Number of histogram bars
#' @param digits Integer indicating the number of decimal places (round) or significant digits (signif) to be used. Negative values are allowed.
#' @param n_head_tail Number of `head` and `tail` of raw data.
#' @param base `skimr::sfl` that sets skimmers for all column types.
#' @param ... (none)
#'
#' @name custom_summary

#' @rdname custom_summary
#' @export
my_skim <- function(..., n_bins = 10, digits = 2, n_head_tail = 3, base = skimr::sfl()) {
  
  func <- skimr::skim_with(append = FALSE, base = base,
                           numeric = skimr::sfl(
                             mean  = function(x) round(mean(x, na.rm = TRUE), digits),
                             sd    = function(x) round(sd(x,   na.rm = TRUE), digits),
                             med   = function(x) round(median(x, na.rm = TRUE), digits),
                             min   = function(x) round(min(x, na.rm = TRUE), digits),
                             `5%`  = function(x) quantile(x, .05, na.rm = TRUE, names = FALSE),
                             `95%` = function(x) quantile(x, .95, na.rm = TRUE, names = FALSE),
                             max   = function(x) round(max(x, na.rm = TRUE), digits),
                             dist  = function(x) skimr::inline_hist(x , n_bins = n_bins),
                             head_tail = function(x)
                               paste(c(round(head(x, n_head_tail), digits), round(tail(x, n_head_tail), digits)), collapse = ' ')
                           ),
                           factor    = skim_default_headtail('factor',    n_head_tail),
                           character = skim_default_headtail('character', n_head_tail),
                           Date      = skim_default_headtail('Date',      n_head_tail),
                           list      = skim_default_headtail('list',      n_head_tail),
                           logical   = skim_default_headtail('logical',   n_head_tail),
                           AsIs      = skim_default_headtail('AsIs',      n_head_tail),
                           complex   = skim_default_headtail('complex',   n_head_tail),
                           difftime  = skim_default_headtail('difftime',  n_head_tail),
                           POSIXct   = skim_default_headtail('POSIXct',   n_head_tail),
                           ts        = skim_default_headtail('ts',        n_head_tail),
  )
  func(...)
}

#' @rdname custom_summary
#' @export
my_skim2 <- function(..., n_bins = 10, digits = 2, n_head_tail = 3,
                     base = skimr::sfl(missing     = skimr:::n_missing)) {
  my_skim(..., n_bins = n_bins, digits = digits, n_head_tail = n_head_tail, base=base)
}

# Internal func
skim_default_headtail <- function(skim_type, n_head_tail) {
  skim_default  <- skimr::get_default_skimmers(skim_type)
  skim_headtail <- skimr::sfl(head_tail = function(x) {
    # Workaround issues of `c` combining factors makes integers
    if (skim_type == 'factor') x <- as.character(x)
    paste(c(head(x, n_head_tail), tail(x, n_head_tail)), collapse = ' ')
  })
  
  do.call(skimr::sfl, c(skim_default[[1]], skim_headtail$funs))
}