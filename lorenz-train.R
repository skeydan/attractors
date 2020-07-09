library(tensorflow)
library(keras)
library(tfdatasets)
library(tfautograph)
library(reticulate)
library(purrr)


source("lorenz-data.R")
source("models.R")
source("loss.R")
source("training.R")


# autoencoder -------------------------------------------------------------------

n_latent <- as.integer(n_timesteps)
n_features <- 1

encoder <- encoder_model(n_timesteps,
                         n_features,
                         n_latent)

decoder <- decoder_model(n_timesteps,
                         n_features,
                         n_latent)


mse_loss <-
  tf$keras$losses$MeanSquaredError(reduction = tf$keras$losses$Reduction$SUM)


train_loss <- tf$keras$metrics$Mean(name = 'train_loss')
train_fnn <- tf$keras$metrics$Mean(name = 'train_fnn')
train_mse <-  tf$keras$metrics$Mean(name = 'train_mse')



fnn_weight <- 10

optimizer <- optimizer_adam(lr = 1e-3)

for (epoch in 1:200) {
  cat("Epoch: ", epoch, " -----------\n")
  training_loop(ds_train)
}

encoder %>% save_model_weights_tf("lorenz_encoder")
decoder %>% save_model_weights_tf("lorenz_decoder")

# check variances -------------------------------------------------------------


test_batch <- as_iterator(ds_test) %>% iter_next()
predicted <- encoder(test_batch[[1]]) %>%
  as.array() %>%
  as_tibble()

predicted %>% summarise_all(var)


# plot attractors on test set ---------------------------------------------------

v1_2 <- ggplot(predicted, aes(V1, V2)) +
  geom_path(size = 0.1, color = "darkgrey") +
  theme_classic() +
  theme(aspect.ratio = 1)

v1_3 <- ggplot(predicted, aes(V1, V4)) +
  geom_path(size = 0.1, color = "darkgrey") +
  theme_classic() +
  theme(aspect.ratio = 1)

v2_3 <- ggplot(predicted, aes(V2, V4)) +
  geom_path(size = 0.1, color = "darkgrey") +
  theme_classic() +
  theme(aspect.ratio = 1)

plot_grid(v1_2, v1_3, v2_3, ncol = 3)
