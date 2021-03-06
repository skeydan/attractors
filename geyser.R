library(reticulate)
source("models.R")
source("training.R")
source("utils.R")
library(tidyverse)
library(cowplot)

#geyser <- py_load_object("data/geyser_train_test.pkl")

n <- 10000
#geyser <- geyser[1:n]
#as.data.frame(geyser) %>% write_csv("data/geyser.csv", col_names = FALSE)
geyser <-
  read_csv("data/geyser.csv", col_names = FALSE) %>% select(X1) %>% pull() %>% unclass()

g1000 <- ggplot(data.frame(temp = geyser[1:1000]), aes(x = 1:1000, y = temp)) +
  geom_line() +
  theme_classic() +
  theme(axis.title = element_blank())

g200 <- ggplot(data.frame(temp = geyser[1:200]), aes(x = 1:200, y = temp)) +
  geom_line() +
  theme_classic() +
  theme(axis.title = element_blank())

plot_grid(g1000, g200, nrow = 2)

geyser <- scale(geyser)

n_timesteps <- 60
batch_size <- 32

train <- gen_timesteps(geyser[1:(n / 2)], 2 * n_timesteps)
test <- gen_timesteps(geyser[(n / 2):n], 2 * n_timesteps)

dim(train) <- c(dim(train), 1)
dim(test) <- c(dim(test), 1)

x_train <- train[, 1:n_timesteps, , drop = FALSE]
y_train <-
  train[, (n_timesteps + 1):(2 * n_timesteps), , drop = FALSE]

ds_train <- tensor_slices_dataset(list(x_train, y_train)) %>%
  dataset_shuffle(nrow(x_train)) %>%
  dataset_batch(batch_size)

x_test <- test[, 1:n_timesteps, , drop = FALSE]
y_test <- test[, (n_timesteps + 1):(2 * n_timesteps), , drop = FALSE]

ds_test <- tensor_slices_dataset(list(x_test, y_test)) %>%
  dataset_batch(nrow(x_test))


# autoencoder -------------------------------------------------------------------

n_latent <- 10L
n_features <- 1
n_hidden <- 32

encoder <- encoder_model(n_timesteps,
                         n_features,
                         n_hidden,
                         n_latent)

decoder <- decoder_model(n_timesteps,
                         n_features,
                         n_hidden,
                         n_latent)


mse_loss <-
  tf$keras$losses$MeanSquaredError(reduction = tf$keras$losses$Reduction$SUM)


train_loss <- tf$keras$metrics$Mean(name = 'train_loss')
train_fnn <- tf$keras$metrics$Mean(name = 'train_fnn')
train_mse <-  tf$keras$metrics$Mean(name = 'train_mse')


fnn_multiplier <- 0.7
fnn_weight <- fnn_multiplier * nrow(x_train) / batch_size

optimizer <- optimizer_adam(lr = 1e-3)

# for (epoch in 1:200) {
#   cat("Epoch: ", epoch, " -----------\n")
#   training_loop(ds_train)
#
#   test_batch <- as_iterator(ds_test) %>% iter_next()
#   encoded <- encoder(test_batch[[1]])
#   test_var <- tf$math$reduce_variance(encoded, axis = 0L)
#   print(test_var %>% as.numeric() %>% round(5))
# }
#
# encoder %>% save_model_weights_tf(paste0("geyser_encoder_", fnn_multiplier))
# decoder %>% save_model_weights_tf(paste0("geyser_decoder_", fnn_multiplier))


encoder %>% load_model_weights_tf(paste0("geyser_encoder_", fnn_multiplier))
decoder %>% load_model_weights_tf(paste0("geyser_decoder_", fnn_multiplier))

# check variances -------------------------------------------------------------


test_batch <- as_iterator(ds_test) %>% iter_next()
encoded <- encoder(test_batch[[1]]) %>%
  as.array() %>%
  as_tibble()

encoded %>% summarise_all(var)
# V1     V2       V3       V4       V5      V6      V7      V8      V9     V10
# 0.258 0.0262 0.0000627 0.000000600 0.000533 0.000362 0.000238 0.000121 0.000518 0.000365


# plot attractors on test set ---------------------------------------------------

a1 <- ggplot(encoded, aes(V1, V2)) +
  geom_path(size = 0.1, color = "darkgrey") +
  theme_classic() +
  theme(aspect.ratio = 1)

a2 <- ggplot(encoded, aes(V1, V5)) +
  geom_path(size = 0.1, color = "darkgrey") +
  theme_classic() +
  theme(aspect.ratio = 1)

a3 <- ggplot(encoded, aes(V2, V5)) +
  geom_path(size = 0.1, color = "darkgrey") +
  theme_classic() +
  theme(aspect.ratio = 1)

plot_grid(a1, a2, a3, ncol = 3)



# predict -----------------------------------------------------------------


prediction_fnn <- decoder(encoder(test_batch[[1]]))

mse_fnn <- get_mse(test_batch, prediction_fnn)
mse_fnn


# lstm --------------------------------------------------------------------

# model <- lstm(n_latent, n_timesteps, n_features, n_hidden, dropout = 0.2, recurrent_dropout = 0.2)
#
# history <- model %>% fit(
#   ds_train,
#   validation_data = ds_test,
#   epochs = 200)
#
# model %>% save_model_hdf5("geyser-lstm.hdf5")
model <- load_model_hdf5("geyser-lstm.hdf5")

test_batch <- as_iterator(ds_test) %>% iter_next()

prediction_lstm <- model %>% predict(ds_test)

mse_lstm <- get_mse(test_batch, prediction_lstm)


# compare errors ----------------------------------------------------------

mses <- data.frame(timestep = 1:n_timesteps, fnn = mse_fnn, lstm = mse_lstm) %>%
  gather(key = "type", value = "mse", -timestep)
ggplot(mses, aes(timestep, mse, color = type)) +
  geom_point() +
  scale_color_manual(values = c("#00008B", "#3CB371")) +
  theme_classic() +
  theme(legend.position = "none") 
                       

# plot predictions --------------------------------------------------------------------


given <- data.frame(as.array(tf$concat(list(
  test_batch[[1]][, , 1], test_batch[[2]][, , 1]
),
axis = 1L)) %>% t()) %>%
  add_column(type = "given") %>%
  add_column(num = 1:(2 * n_timesteps))

fnn <- data.frame(as.array(prediction_fnn[, , 1]) %>%
                    t()) %>%
  add_column(type = "fnn") %>%
  add_column(num = (n_timesteps  + 1):(2 * n_timesteps))

lstm <- data.frame(as.array(prediction_lstm[, , 1]) %>%
                     t()) %>%
  add_column(type = "lstm") %>%
  add_column(num = (n_timesteps + 1):(2 * n_timesteps))

compare_preds_df <- bind_rows(given, lstm, fnn)

plots <- purrr::map(sample(1:dim(compare_preds_df)[2], 16),
                    function(v)
                      ggplot(compare_preds_df, aes(num, .data[[paste0("X", v)]], color = type)) +
                      geom_line() +
                      theme_classic() +
                      theme(legend.position = "none", axis.title = element_blank()) +
                      scale_color_manual(values = c("#00008B", "#DB7093", "#3CB371")))


plot_grid(plotlist = plots, ncol = 4)
