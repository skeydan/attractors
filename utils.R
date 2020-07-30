gen_timesteps <- function(x, n_timesteps) {
  do.call(rbind,
          purrr::map(seq_along(x),
                     function(i) {
                       start <- i
                       end <- i + n_timesteps - 1
                       out <- x[start:end]
                       out
                     })) %>%
    na.omit()
}

calc_mse <- function(df, y_true, y_pred) {
  (sum((df[[y_true]] - df[[y_pred]]) ^ 2)) / nrow(df)
}

get_mse <- function(test_batch, prediction, use_actual_y = TRUE) {
  y <- if (use_actual_y)
    test_batch[[2]]
  else
    test_batch[[4]]
  comp_df <- data.frame(y[, , 1] %>%
                          as.array()) %>%
    rename_with(function(name)
      paste0(name, "_true")) %>%
    bind_cols(data.frame(prediction[, , 1] %>% as.array()) %>%
                rename_with(function(name)
                  paste0(name, "_pred")))
  mse <- purrr::map(1:dim(prediction)[2],
                    function(varno)
                      calc_mse(
                        comp_df,
                        paste0("X", varno, "_true"),
                        paste0("X", varno, "_pred")
                      )) %>%
    unlist()
  
  mse
  
}
