library(keras)
library(tfdatasets)
library(reticulate)

source("lorenz-data.R")
source("models.R")
source("utils.R")

n_latent <- as.integer(n_timesteps)
n_features <- 1

encoder <- encoder_model(n_timesteps,
                         n_features,
                         n_latent)

decoder <- decoder_model(n_timesteps,
                         n_features,
                         n_latent)

encoder %>% load_model_weights_tf("lorenz_encoder")
decoder %>% load_model_weights_tf("lorenz_decoder")


test_batch <- as_iterator(ds_test) %>% iter_next()
prediction <- decoder(encoder(test_batch[[1]]))

prediction[1:10, , 1] %>% as.array()

test_batch[[2]][1:10, , 1] %>% as.array()

test_batch[[1]][1:10, , 1] %>% as.array()


mse_fnn <- get_mse(test_batch, prediction)





