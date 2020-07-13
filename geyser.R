library(reticulate)
source("models.R")
source("training.R")
source("utils.R")
library(tidyverse)
library(cowplot)



geyser <- py_load_object("data/geyser_train_test.pkl")

n <- 10000
geyser <- geyser[1:(2 * n)]

ggplot(data.frame(temp = geyser[1:1000]), aes(x = 1:1000, y = temp)) + 
  geom_line() +
  theme_classic() +
  theme(
    axis.title.x=element_blank(),
    axis.ticks.x=element_blank())

ggplot(data.frame(temp = geyser[1:200]), aes(x = 1:200, y = temp)) + 
  geom_line() +
  theme_classic() +
  theme(
    axis.title.x=element_blank(),
    axis.ticks.x=element_blank())
  

geyser <- scale(geyser)

n_timesteps <- 60
batch_size <- 32

train <- gen_timesteps(geyser[1:(n/2)], 2 * n_timesteps)
test <- gen_timesteps(geyser[(n/2):n], 2 * n_timesteps) 

dim(train) <- c(dim(train), 1)
dim(test) <- c(dim(test), 1)

x_train <- train[ , 1:n_timesteps, , drop = FALSE]
y_train <- train[ , (n_timesteps + 1):(2*n_timesteps), , drop = FALSE]
x_train[1:4, , 1]
y_train[1:4, , 1]

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
fnn_weight <- fnn_multiplier * nrow(x_train)/batch_size

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
# 1 0.265 0.0271  7.84e-5  5.70e-7 0.000543 3.42e-4 2.46e-4 1.29e-4 5.11e-4 3.62e-4


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



# lstm --------------------------------------------------------------------

# model <- lstm(n_latent, n_timesteps, n_features, dropout = 0, recurrent_dropout = 0)
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

#> mse_fnn
#[1] 0.1743249 0.4184542 0.5609179 0.6515022 0.7126671 0.7483752 0.7556070 0.7468666

# [1] 0.3814891 0.5110050 0.5736291 0.5890495 0.5946818 0.6051609 0.6250113
# [8] 0.6500638 0.6693160 0.6828012 0.6932985 0.7053073 0.7041866 0.6996050
# [15] 0.7037117 0.7072416 0.7069045 0.7080551 0.7145087 0.7092809 0.7045435
# [22] 0.7085868 0.7087260 0.7097822 0.7064175 0.7069261 0.7143834 0.7131030
# [29] 0.7111084 0.7146822 0.7199935 0.7171039 0.7107147 0.7066788 0.7022842
# [36] 0.7017120 0.7023662 0.7019490 0.6914956 0.6864950 0.6939680 0.7048042
# [43] 0.7228798 0.7762417 0.8176706 0.8338339 0.8177023 0.8306618 0.8666893
# [50] 0.8894805 0.9060865 0.9260449 0.9446564 0.9513241 0.9371954 0.9269431
# [57] 0.9251430 0.9323774 0.9403623 0.9486062


# > mse_lstm



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

lstm <- data.frame(as.array(prediction_lstm[ , , 1]) %>% 
                    t()) %>%
  add_column(type = "lstm") %>%
  add_column(num = (n_timesteps + 1):(2 * n_timesteps))

compare_preds_df <- bind_rows(given, lstm, fnn)

plots <- purrr::map(sample(1: dim(compare_preds_df)[2], 16), 
                    function(v) ggplot(compare_preds_df, aes(num, .data[[paste0("X", v)]], color = type)) +
                      geom_line() + 
                      theme_classic() + 
                      theme(legend.position = "none"))


plot_grid(plotlist = plots, ncol = 4)


