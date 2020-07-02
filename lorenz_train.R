library(tensorflow)
library(keras)
library(tfdatasets)
library(tfautograph)
library(reticulate)
library(purrr)


source("lorenz-data.R")
source("models.R")
source("loss.R")


# model -------------------------------------------------------------------

n_latent <- 10L
n_features <- 1
fnn_weight <- 10

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


train_step <- function(batch) {
  with (tf$GradientTape(persistent = TRUE) %as% tape, {
    code <- encoder(batch[[1]])
    reconstructed <- decoder(code)
    
    l_mse <- mse_loss(batch[[2]], reconstructed)
    l_fnn <- loss_false_nn(code)
    loss <- l_mse + fnn_weight * l_fnn
  })
  
  encoder_gradients <-
    tape$gradient(loss, encoder$trainable_variables)
  decoder_gradients <-
    tape$gradient(loss, decoder$trainable_variables)
  
  optimizer$apply_gradients(purrr::transpose(list(
    encoder_gradients, encoder$trainable_variables
  )))
  optimizer$apply_gradients(purrr::transpose(list(
    decoder_gradients, decoder$trainable_variables
  )))
  
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

#predicted %>% summarise_all(var)


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

#plot_grid(v1_2, v1_3, v2_3, ncol = 3)
