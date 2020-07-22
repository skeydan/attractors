library(tensorflow)
library(keras)
library(tfdatasets)
library(tfautograph)
library(reticulate)
library(purrr)

source("loss.R")


train_step <- function(batch) {
  with (tf$GradientTape(persistent = TRUE) %as% tape, {
    code <- encoder(batch[[1]])
    prediction <- decoder(code)
    
    l_mse <- mse_loss(batch[[2]], prediction)
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

reparameterize <- function(mean, logvar = 0) {
  eps <- k_random_normal(shape = n_latent)
  eps * k_exp(logvar * 0.5) + mean
}

train_step_vae <- function(batch) {
  with (tf$GradientTape(persistent = TRUE) %as% tape, {
    code <- encoder(batch[[1]])
    z <- reparameterize(code)
    prediction <- decoder(z)
    
    l_mse <- mse_loss(batch[[2]], prediction)
    l_fnn <- loss_false_nn(code)
    l_kl <- -0.5 * k_mean(1 - k_square(z))
    loss <- l_mse + fnn_weight * l_fnn + kl_weight * l_kl
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
  train_kl(l_kl)
}

training_loop_vae <- tf_function(autograph(function(ds_train) {
  for (batch in ds_train) {
    train_step_vae(batch) 
  }
  
  tf$print("Loss: ", train_loss$result())
  tf$print("MSE: ", train_mse$result())
  tf$print("FNN loss: ", train_fnn$result())
  tf$print("KL loss: ", train_kl$result())
  
  train_loss$reset_states()
  train_mse$reset_states()
  train_fnn$reset_states()
  train_kl$reset_states()
  
}))
