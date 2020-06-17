library(keras)
library(tfdatasets)
library(tfautograph)
library(reticulate)
library(purrr)


source("lorenz.R")



# Preprocess --------------------------------------------------------------

obs <- obs %>% mutate(
  x = scale(x)
)

obs[1:10,]

n <- nrow(obs)
n_timesteps <- 10

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


# train with start of time series, test with end of time series 
x_train <- gen_timesteps(as.matrix(obs$x)[1:(n/2)], n_timesteps)
x_test <- gen_timesteps(as.matrix(obs$x)[(n/2):n], n_timesteps) 

x_train[1:10, ]
x_test[1:10, ]

dim(x_train) <- c(dim(x_train), 1)
dim(x_test) <- c(dim(x_test), 1)

dim(x_train)
dim(x_test)

batch_size <- 100

ds_train <- tensor_slices_dataset(x_train) %>%
  dataset_batch(batch_size)

ds_test <- tensor_slices_dataset(x_test) %>%
  dataset_batch(nrow(x_test))


# model -------------------------------------------------------------------

n_latent <- 10L
n_features <- 1
fnn_weight <- 20


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
        units = n_latent,
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


encoder <- encoder_model(n_timesteps,
                              n_features,
                              n_latent)

decoder <- decoder_model(n_timesteps,
                         n_features,
                         n_latent)


optimizer <- optimizer_adam(lr = 1e-3)

mse_loss <- tf$keras$losses$MeanSquaredError(reduction = tf$keras$losses$Reduction$SUM)

loss_false_nn <- function(x) {
  
  
  # changing these parameters is equivalent to
  # changing the strength of the regularizer, so we keep these fixed (these values
  # correspond to the original values used in Kennel et al 1992).
  rtol <- 20
  atol <- 2
  k_frac <- 0.01
  
  k <- max(1, floor(k_frac * batch_size))
  
  ## Vectorized version of distance matrix calculation
  tri_mask <-
    tf$linalg$band_part(
      tf$ones(
        shape = c(n_latent, n_latent),
        dtype = tf$float32
      ),
      num_lower = -1L,
      num_upper = 0L
    )
  
  # latent x batch_size x latent
  batch_masked <-
    tf$multiply(tri_mask[, tf$newaxis,], x[tf$newaxis, reticulate::py_ellipsis()])
  
  # latent x batch_size x 1
  x_squared <-
    tf$reduce_sum(batch_masked * batch_masked,
                  axis = 2L,
                  keepdims = TRUE)
  
  # latent x batch_size x batch_size
  pdist_vector <- x_squared + tf$transpose(x_squared, perm = c(0L, 2L, 1L)) -
    2 * tf$matmul(batch_masked, tf$transpose(batch_masked, perm = c(0L, 2L, 1L)))
  
  #(latent, batch_size, batch_size)
  all_dists <- pdist_vector
  # latent
  all_ra <-
    tf$sqrt((1 / (
      batch_size * tf$range(1, 1 + n_latent, dtype = tf$float32)
    )) *
      tf$reduce_sum(tf$square(
        batch_masked - tf$reduce_mean(batch_masked, axis = 1L, keepdims = TRUE)
      ), axis = c(1L, 2L)))
  
  # Avoid singularity in the case of zeros
  #(latent, batch_size, batch_size)
  all_dists <-
    tf$clip_by_value(all_dists, 1e-14, tf$reduce_max(all_dists))
  
  #inds = tf.argsort(all_dists, axis=-1)
  top_k <- tf$math$top_k(-all_dists, tf$cast(k + 1, tf$int32))
  # (#(latent, batch_size, batch_size)
  top_indices <- top_k[[1]]
  
  #(latent, batch_size, batch_size)
  neighbor_dists_d <-
    tf$gather(all_dists, top_indices, batch_dims = -1L)
  #(latent - 1, batch_size, batch_size)
  neighbor_new_dists <-
    tf$gather(all_dists[2:-1, , ],
              top_indices[1:-2, , ],
              batch_dims = -1L)
  
  # Eq. 4 of Kennel et al.
  #(latent - 1, batch_size, batch_size)
  scaled_dist <- tf$sqrt((
    tf$square(neighbor_new_dists) -
      # (9, 8, 2)
      tf$square(neighbor_dists_d[1:-2, , ])) /
      # (9, 8, 2)
      tf$square(neighbor_dists_d[1:-2, , ])
  )
  
  # Kennel condition #1
  #(latent - 1, batch_size, batch_size)
  is_false_change <- (scaled_dist > rtol)
  # Kennel condition 2
  #(latent - 1, batch_size, batch_size)
  is_large_jump <-
    (neighbor_new_dists > atol * all_ra[1:-2, tf$newaxis, tf$newaxis])
  
  is_false_neighbor <-
    tf$math$logical_or(is_false_change, is_large_jump)
  #(latent - 1, batch_size, 1)
  total_false_neighbors <-
    tf$cast(is_false_neighbor, tf$int32)[reticulate::py_ellipsis(), 2:(k + 2)]
  
  # Pad zero to match dimensionality of latent space
  # (latent - 1)
  reg_weights <-
    1 - tf$reduce_mean(tf$cast(total_false_neighbors, tf$float32), axis = c(1L, 2L))
  # (latent,)
  temp = reg_weights
  reg_weights <- tf$pad(reg_weights, list(list(1L, 0L)))
  
  # Find batch average activity
  
  # L2 Activity regularization
  activations_batch_averaged <-
    tf$sqrt(tf$reduce_mean(tf$square(x), axis = 0L))
  
  loss <- tf$reduce_sum(tf$multiply(reg_weights, activations_batch_averaged))
  loss
  
}


train_loss <- tf$keras$metrics$Mean(name='train_loss')
train_fnn <- tf$keras$metrics$Mean(name='train_fnn')
train_mse <-  tf$keras$metrics$Mean(name='train_mse')


train_step <- function(batch) {
  
  with (tf$GradientTape(persistent = TRUE) %as% tape, {
    code <- encoder(batch)
    reconstructed <- decoder(code)
    
    l_mse <- mse_loss(batch, reconstructed)
    l_fnn <- loss_false_nn(code)
    loss <- l_mse + fnn_weight * l_fnn
  })
  
  encoder_gradients <- tape$gradient(loss, encoder$trainable_variables)
  decoder_gradients <- tape$gradient(loss, decoder$trainable_variables)
  
  optimizer$apply_gradients(
    purrr::transpose(list(encoder_gradients, encoder$trainable_variables))
  )
  optimizer$apply_gradients(
    purrr::transpose(list(decoder_gradients, decoder$trainable_variables))
  )
  
  train_loss(loss)
  train_mse(l_mse)
  train_fnn(l_fnn)
}

training_loop <- tf_function(autograph(function(ds_train) {
  
  for (batch in ds_train) {
    train_step(batch)
  }
  
  tf$print("Loss: ", train_loss$result())
  tf$print("MSE: ", train_mse$result())
  tf$print("FNN loss: ", train_fnn$result())
  
  train_loss$reset_states()
  train_mse$reset_states()
  train_fnn$reset_states()
  
}))

for (epoch in 1:200) {
  cat("Epoch: ", epoch, " -----------\n")
  training_loop(ds_train)  
}


# check variances -------------------------------------------------------------


test_batch <- as_iterator(ds_test) %>% iter_next()
predicted <- encoder(test_batch) %>%
  as.array(predicted) %>%
  as_tibble()

predicted %>% summarise_all(var)

# plot attractors on test set ---------------------------------------------------


v1_2 <- ggplot(predicted,
               aes(V1, V2)) +
  geom_path(size = 0.2) +
  theme_classic() +
  theme(aspect.ratio = 1)

v1_3 <- ggplot(predicted ,
               aes(V1, V3)) +
  geom_path(size = 0.2) +
  theme_classic() +
  theme(aspect.ratio = 1)

v2_3 <- ggplot(predicted,
               aes(V2, V3)) +
  geom_path(size = 0.2) +
  theme_classic() +
  theme(aspect.ratio = 1)

plot_grid(v1_2, v1_3, v2_3, ncol = 3)



