encoder_model <- function(n_timesteps,
                          n_features,
                          n_recurrent,
                          n_latent,
                          name = NULL) {
  keras_model_custom(name = name, function(self) {
    self$noise <- layer_gaussian_noise(stddev = 0.5)
    self$lstm1 <-  layer_lstm(
      units = n_recurrent,
      input_shape = c(n_timesteps, n_features),
      return_sequences = TRUE
    )
    self$batchnorm1 <- layer_batch_normalization()
    self$lstm2 <-  layer_lstm(units = n_latent,
                              return_sequences = FALSE)
    self$batchnorm2 <- layer_batch_normalization()
    
    function (x, mask = NULL) {
      x %>%
        self$noise() %>%
        self$lstm1() %>%
        self$batchnorm1() %>%
        self$lstm2() %>%
        self$batchnorm2()
    }
  })
}

decoder_model <- function(n_timesteps,
                          n_features,
                          n_recurrent,
                          n_latent,
                          name = NULL) {
  keras_model_custom(name = name, function(self) {
    self$repeat_vector <- layer_repeat_vector(n = n_timesteps)
    self$noise <- layer_gaussian_noise(stddev = 0.5)
    self$lstm <- layer_lstm(
      units = n_recurrent,
      return_sequences = TRUE,
      go_backwards = TRUE
    )
    self$batchnorm <- layer_batch_normalization()
    self$elu <- layer_activation_elu()
    self$time_distributed <-
      time_distributed(layer = layer_dense(units = n_features))
    
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

lstm <-
  function(n_latent,
           n_timesteps,
           n_features,
           n_recurrent,
           dropout,
           recurrent_dropout,
           optimizer = optimizer_adam(lr =  1e-3)) {
    model <- keras_model_sequential() %>%
      layer_lstm(
        units = n_recurrent,
        input_shape = c(n_timesteps, n_features),
        dropout = dropout,
        recurrent_dropout = recurrent_dropout,
        return_sequences = TRUE
      ) %>%
      layer_lstm(
        units = n_recurrent,
        dropout = dropout,
        recurrent_dropout = recurrent_dropout,
        return_sequences = TRUE
      ) %>%
      time_distributed(layer_dense(units = 1))
    
    model %>%
      compile(loss = "mse",
              optimizer = optimizer)
    model
    
  }

conv_encoder_model <- function(n_timesteps,
                               n_features,
                               n_latent,
                               name = NULL) {
  keras_model_custom(name = name, function(self) {
    self$noise <- layer_gaussian_noise(stddev = 0.5)
    self$conv1 <- layer_conv_1d(kernel_size = 3,
                                filters = 16,
                                strides = 2)
    self$act1 <- layer_activation_leaky_relu()
    self$batchnorm1 <- layer_batch_normalization()
    self$conv2 <- layer_conv_1d(kernel_size = 7,
                                filters = 32,
                                strides = 2)
    self$act2 <- layer_activation_leaky_relu()
    self$batchnorm2 <- layer_batch_normalization()
    self$conv3 <- layer_conv_1d(kernel_size = 9,
                                filters = 64,
                                strides = 2)
    self$act3 <- layer_activation_leaky_relu()
    self$batchnorm3 <- layer_batch_normalization()
    self$conv4 <- layer_conv_1d(
      kernel_size = 9,
      filters = n_latent,
      strides = 2,
      activation = "linear" ## ?
    )
    self$batchnorm4 <- layer_batch_normalization()
    self$flat <- layer_flatten()
    
    function (x, mask = NULL) {
      x %>%
        self$noise() %>%
        self$conv1() %>%
        self$act1() %>%
        self$batchnorm1() %>%
        self$conv2() %>%
        self$act2() %>%
        self$batchnorm2() %>%
        self$conv3() %>%
        self$act3() %>%
        self$batchnorm3() %>%
        self$conv4() %>%
        self$batchnorm4() %>%
        self$flat()
    }
  })
}

conv_decoder_model <- function(n_timesteps,
                               n_features,
                               n_latent,
                               name = NULL) {
  keras_model_custom(name = name, function(self) {
    self$reshape <- layer_reshape(target_shape = c(1, n_latent))
    self$noise <- layer_gaussian_noise(stddev = 0.5)
    self$conv1 <- layer_conv_1d_transpose(kernel_size = 15,
                                          filters = 64,
                                          strides = 3)
    self$act1 <- layer_activation_leaky_relu()
    self$batchnorm1 <- layer_batch_normalization()
    self$conv2 <- layer_conv_1d_transpose(kernel_size = 11,
                                          filters = 32,
                                          strides = 3)
    self$act2 <- layer_activation_leaky_relu()
    self$batchnorm2 <- layer_batch_normalization()
    self$conv3 <- layer_conv_1d_transpose(
      kernel_size = 9,
      filters = 16,
      strides = 2,
      output_padding = 1
    )
    self$act3 <- layer_activation_leaky_relu()
    self$batchnorm3 <- layer_batch_normalization()
    self$conv4 <- layer_conv_1d_transpose(
      kernel_size = 7,
      filters = 1,
      strides = 1,
      activation = "linear" ## ?
    )
    self$batchnorm4 <- layer_batch_normalization()
    
    function (x, mask = NULL) {
      x %>%
        self$reshape() %>%
        self$noise() %>%
        self$conv1() %>%
        self$act1() %>%
        self$batchnorm1() %>%
        self$conv2() %>%
        self$act2() %>%
        self$batchnorm2() %>%
        self$conv3() %>%
        self$act3() %>%
        self$batchnorm3() %>%
        self$conv4() %>%
        self$batchnorm4()
        
    }
  })
}

vae_encoder_model <- function(n_timesteps,
                               n_features,
                               n_latent,
                               name = NULL) {
  keras_model_custom(name = name, function(self) {
    self$conv1 <- layer_conv_1d(kernel_size = 3,
                                filters = 16,
                                strides = 2)
    self$act1 <- layer_activation_leaky_relu()
    self$batchnorm1 <- layer_batch_normalization()
    self$conv2 <- layer_conv_1d(kernel_size = 7,
                                filters = 32,
                                strides = 2)
    self$act2 <- layer_activation_leaky_relu()
    self$batchnorm2 <- layer_batch_normalization()
    self$conv3 <- layer_conv_1d(kernel_size = 9,
                                filters = 64,
                                strides = 2)
    self$act3 <- layer_activation_leaky_relu()
    self$batchnorm3 <- layer_batch_normalization()
    self$conv4 <- layer_conv_1d(
      kernel_size = 9,
      filters = n_latent,
      strides = 2,
      activation = "linear" ## ?
    )
    self$batchnorm4 <- layer_batch_normalization()
    self$flat <- layer_flatten()
    
    function (x, mask = NULL) {
      x %>%
        self$conv1() %>%
        self$act1() %>%
        self$batchnorm1() %>%
        self$conv2() %>%
        self$act2() %>%
        self$batchnorm2() %>%
        self$conv3() %>%
        self$act3() %>%
        self$batchnorm3() %>%
        self$conv4() %>%
        self$batchnorm4() %>%
        self$flat()
    }
  })
}

vae_decoder_model <- function(n_timesteps,
                               n_features,
                               n_latent,
                               name = NULL) {
  keras_model_custom(name = name, function(self) {
    self$reshape <- layer_reshape(target_shape = c(1, n_latent))
    self$conv1 <- layer_conv_1d_transpose(kernel_size = 15,
                                          filters = 64,
                                          strides = 3)
    self$act1 <- layer_activation_leaky_relu()
    self$batchnorm1 <- layer_batch_normalization()
    self$conv2 <- layer_conv_1d_transpose(kernel_size = 11,
                                          filters = 32,
                                          strides = 3)
    self$act2 <- layer_activation_leaky_relu()
    self$batchnorm2 <- layer_batch_normalization()
    self$conv3 <- layer_conv_1d_transpose(
      kernel_size = 9,
      filters = 16,
      strides = 2,
      output_padding = 1
    )
    self$act3 <- layer_activation_leaky_relu()
    self$batchnorm3 <- layer_batch_normalization()
    self$conv4 <- layer_conv_1d_transpose(
      kernel_size = 7,
      filters = 1,
      strides = 1,
      activation = "linear"
    )
    self$batchnorm4 <- layer_batch_normalization()
    
    function (x, mask = NULL) {
      x %>%
        self$reshape() %>%
        self$conv1() %>%
        self$act1() %>%
        self$batchnorm1() %>%
        self$conv2() %>%
        self$act2() %>%
        self$batchnorm2() %>%
        self$conv3() %>%
        self$act3() %>%
        self$batchnorm3() %>%
        self$conv4() %>%
        self$batchnorm4()
      
    }
  })
}

