library(keras)
library(tfdatasets)
library(reticulate)

source("lorenz-data.R")
source("models.R")

n_latent <- 10L
n_features <- 1
fnn_weight <- 10

encoder <- encoder_model(n_timesteps,
                         n_features,
                         n_latent)

decoder <- decoder_model(n_timesteps,
                         n_features,
                         n_latent)

encoder %>% load_model_weights_tf("lorenz_encoder")
decoder %>% load_model_weights_tf("lorenz_decoder")

n_train <- nrow(x_train)
ds_train <- tensor_slices_dataset(list(x_train, y_train)) %>%
  dataset_batch(n_train)

train_batch <- as_iterator(ds_train) %>% iter_next()
train_attractor <- encoder(train_batch[[1]]) %>%
  as.array() %>%
  as_tibble()

ds_test <- tensor_slices_dataset(list(x_test, y_test)) %>%
  dataset_batch(1)
test_batch <- as_iterator(ds_test) %>% iter_next()

test_encoded <- encoder(test_batch[[1]]) %>%
  as.array() %>%
  as_tibble()


