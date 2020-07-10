library(reticulate)
source("models.R")
source("training.R")
source("utils.R")
library(tidyverse)
library(cowplot)



mouse <- py_load_object("data/mouse.pkl")

n <- 10000
mouse <- mouse[1:(2 * n)]

ggplot(data.frame(kw = mouse[1:2000]), aes(x = 1:2000, y = kw)) + 
  geom_line() +
  theme_classic() +
  theme(
    axis.title.x=element_blank(),
    axis.ticks.x=element_blank())

mouse <- scale(mouse)

n_timesteps <- 8
batch_size <- 100

train <- gen_timesteps(mouse[1:(n/2)], 2 * n_timesteps)
test <- gen_timesteps(mouse[(n/2):n], 2 * n_timesteps) 

dim(train) <- c(dim(train), 1)
dim(test) <- c(dim(test), 1)

x_train <- train[ , 1:8, , drop = FALSE]
y_train <- train[ , 9:16, , drop = FALSE]
x_train[1:8, , 1]
y_train[1:8, , 1]

ds_train <- tensor_slices_dataset(list(x_train, y_train)) %>%
  dataset_shuffle(nrow(x_train)) %>%
  dataset_batch(batch_size)

x_test <- test[ , 1:8, , drop = FALSE]
y_test <- test[ , 9:16, , drop = FALSE]

ds_test <- tensor_slices_dataset(list(x_test, y_test)) %>%
  dataset_batch(nrow(x_test))


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


fnn_multiplier <- 0.5
fnn_weight <- fnn_multiplier * nrow(x_train)/batch_size

optimizer <- optimizer_adam(lr = 1e-3)

for (epoch in 1:200) {
  cat("Epoch: ", epoch, " -----------\n")
  training_loop(ds_train)
  
  test_batch <- as_iterator(ds_test) %>% iter_next()
  encoded <- encoder(test_batch[[1]]) 
  test_var <- tf$math$reduce_variance(encoded, axis = 0L)
  print(test_var %>% as.numeric() %>% round(5))
}

encoder %>% save_model_weights_tf(paste0("mouse_encoder_", fnn_multiplier))
decoder %>% save_model_weights_tf(paste0("mouse_decoder_", fnn_multiplier))

# check variances -------------------------------------------------------------


test_batch <- as_iterator(ds_test) %>% iter_next()
encoded <- encoder(test_batch[[1]]) %>%
  as.array() %>%
  as_tibble()

encoded %>% summarise_all(var)


# plot attractors on test set ---------------------------------------------------

a1 <- ggplot(encoded, aes(V1, V2)) +
  geom_path(size = 0.1, color = "darkgrey") +
  theme_classic() +
  theme(aspect.ratio = 1)

a2 <- ggplot(encoded, aes(V1, V3)) +
  geom_path(size = 0.1, color = "darkgrey") +
  theme_classic() +
  theme(aspect.ratio = 1)

a3 <- ggplot(encoded, aes(V2, V3)) +
  geom_path(size = 0.1, color = "darkgrey") +
  theme_classic() +
  theme(aspect.ratio = 1)

plot_grid(a1, a2, a3, ncol = 3)



# predict -----------------------------------------------------------------


prediction_fnn <- decoder(encoder(test_batch[[1]]))

prediction_fnn[1:10, , 1] %>% as.array()

test_batch[[2]][1:10, , 1] %>% as.array()

test_batch[[1]][1:10, , 1] %>% as.array()


mse_fnn <- get_mse(test_batch, prediction_fnn)



# lstm --------------------------------------------------------------------

model <- lstm(n_latent, n_timesteps, n_features, dropout = 0, recurrent_dropout = 0)

history <- model %>% fit(
  ds_train,
  validation_data = ds_test,
  epochs = 200)

model %>% save_model_hdf5("mouse-lstm.hdf5")

prediction_lstm <- model %>% predict(ds_test)

mse_lstm <- get_mse(test_batch, prediction_lstm)

#

# > mse_fnn
# [1] 0.007859814 0.013104261 0.016770325 0.021908303 0.029646938 0.038758405 0.050011331 0.064407731
# > mse_lstm
# [1] 0.30758243 0.16506516 0.12208551 0.09304894 0.07583015 0.06916708 0.06846981 0.06907555

