encoder_model <- function(n_timesteps,
                          n_features,
                          n_latent,
                          name = NULL) {
  
  keras_model_custom(name = name, function(self) {
    
    self$noise <- layer_gaussian_noise(stddev = 0.5)
    self$lstm <-  layer_lstm(
      units = n_latent,
      input_shape = c(n_timesteps, n_features),
      return_sequences = FALSE
    ) 
    self$batchnorm <- layer_batch_normalization()
    
    function (x, mask = NULL) {
      x %>%
        self$noise() %>%
        self$lstm() %>%
        self$batchnorm() 
    }
  })
}

decoder_model <- function(n_timesteps,
                          n_features,
                          n_latent,
                          name = NULL) {
  
  keras_model_custom(name = name, function(self) {
    
    self$repeat_vector <- layer_repeat_vector(n = n_timesteps)
    self$noise <- layer_gaussian_noise(stddev = 0.5)
    self$lstm <- layer_lstm(
      units = n_timesteps,
      return_sequences = TRUE,
      go_backwards = TRUE
    ) 
    self$batchnorm <- layer_batch_normalization()
    self$elu <- layer_activation_elu() 
    self$time_distributed <- time_distributed(layer = layer_dense(units = n_features))
    
    function (x, mask = NULL) {
      x %>%
        self$repeat_vector() %>%
        self$noise() %>%
        self$lstm() %>%
        self$batchnorm() %>%
        self$elu() %>%
        self$time_distributed()
    }
  })
}

lstm <- function(n_latent, n_timesteps, n_features, dropout, recurrent_dropout,
                 optimizer = optimizer_adam(lr =  1e-3)) {
  model <- keras_model_sequential() %>%
    layer_lstm(
      units = n_latent,
      input_shape = c(n_timesteps, n_features),
      dropout = dropout, 
      recurrent_dropout = recurrent_dropout,
      return_sequences = TRUE
    ) %>% 
    layer_lstm(
      units = n_latent,
      dropout = dropout,
      recurrent_dropout = recurrent_dropout,
      return_sequences = TRUE
    ) %>% 
    time_distributed(layer_dense(units = 1))
  
  model %>%
    compile(
      loss = "mse",
      optimizer = optimizer
    )
  model
  
}
