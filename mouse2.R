library(reticulate)
source("models.R")
source("training.R")
source("utils.R")
library(tidyverse)
library(cowplot)

n <- 10000
mouse <-
  read_csv("data/mouse.csv", col_names = FALSE) %>% select(X1) %>% pull() %>% unclass()

n_timesteps <- 120
batch_size <- 32

mouse <- scale(mouse)

train <- gen_timesteps(mouse[1:(n/2)], 2 * n_timesteps)
test <- gen_timesteps(mouse[(n/2):n], 2 * n_timesteps) 

dim(train) <- c(dim(train), 1)
dim(test) <- c(dim(test), 1)

x_train <- train[ , 1:n_timesteps, , drop = FALSE]
y_train <- train[ , (n_timesteps + 1):(2*n_timesteps), , drop = FALSE]

ds_train <- tensor_slices_dataset(list(x_train, y_train)) %>%
  dataset_shuffle(nrow(x_train)) %>%
  dataset_batch(batch_size)

x_test <- test[ , 1:n_timesteps, , drop = FALSE]
y_test <- test[ , (n_timesteps + 1):(2*n_timesteps), , drop = FALSE]

ds_test <- tensor_slices_dataset(list(x_test, y_test)) %>%
  dataset_batch(nrow(x_test))


# autoencoder -------------------------------------------------------------------

n_latent <- 10L
n_features <- 1

encoder <- conv_encoder_model(n_timesteps,
                         n_features,
                         n_latent)

decoder <- conv_decoder_model(n_timesteps,
                         n_features,
                         n_latent)
#batch <- as_iterator(ds_train) %>% iter_next()
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

encoder %>% save_model_weights_tf(paste0("mouse_conv_encoder_", fnn_multiplier))
decoder %>% save_model_weights_tf(paste0("mouse_conv_decoder_", fnn_multiplier))

encoder %>% load_model_weights_tf(paste0("mouse_conv_encoder_", fnn_multiplier))
decoder %>% load_model_weights_tf(paste0("mouse_conv_decoder_", fnn_multiplier))

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

mse_fnn <- get_mse(test_batch, prediction_fnn)

mse_fnn


# plot --------------------------------------------------------------------


given <- data.frame(
  as.array(
    tf$concat(list(test_batch[[1]][ , , 1], test_batch[[2]][ , , 1]),
              axis = 1L)) %>% t()) %>% 
  add_column(type = "given") %>%
  add_column(num = 1:(2 * n_timesteps))

fnn <- data.frame(as.array(prediction_fnn[ , , 1]) %>% 
                    t()) %>%
  add_column(type = "fnn") %>%
  add_column(num = (n_timesteps  +1):(2 * n_timesteps))

compare_preds_df <- bind_rows(given, fnn)

plots <- purrr::map(sample(1: dim(compare_preds_df)[2], 16), 
                    function(v) ggplot(compare_preds_df, aes(num, .data[[paste0("X", v)]], color = type)) +
                      geom_line() + 
                      theme_classic() + 
                      theme(legend.position = "none", axis.title = element_blank()) +
                      scale_color_manual(values=c("#00008B", "#DB7093")))


plot_grid(plotlist = plots, ncol = 4)


