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
    
    self$repeat_vector <- layer_repeat_vector(n = n_timesteps - 1)
    self$noise <- layer_gaussian_noise(stddev = 0.5)
    self$lstm <- layer_lstm(
      units = n_timesteps - 1,
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

