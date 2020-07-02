gen_timesteps <- function(x, n_timesteps) {
  do.call(rbind,
          purrr::map(seq_along(x),
                     function(i) {
                       start <- i
                       end <- i + n_timesteps - 1
                       out <- x[start:end]
                       out
                     })
  ) %>%
    na.omit()
}

