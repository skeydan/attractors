source("lorenz-data.R")

n_latent <- as.integer(n_timesteps)
n_features <- 1

model <- lstm(n_latent, n_timesteps, n_features, dropout = 0, recurrent_dropout = 0)

history <- model %>% fit(
  ds_train,
  validation_data = ds_test,
  epochs = 200)

model %>% save_model_hdf5("lorenz-lstm.hdf5")

prediction <- model %>% predict(ds_test)

test_batch <- as_iterator(ds_test) %>% iter_next()

mse_lstm <- get_mse(test_batch, prediction)
