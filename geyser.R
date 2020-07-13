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

for (epoch in 1:200) {
  cat("Epoch: ", epoch, " -----------\n")
  training_loop(ds_train)
  
  test_batch <- as_iterator(ds_test) %>% iter_next()
  encoded <- encoder(test_batch[[1]]) 
  test_var <- tf$math$reduce_variance(encoded, axis = 0L)
  print(test_var %>% as.numeric() %>% round(5))
}

encoder %>% save_model_weights_tf(paste0("geyser_encoder_", fnn_multiplier))
decoder %>% save_model_weights_tf(paste0("geyser_decoder_", fnn_multiplier))


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

a2 <- ggplot(encoded, aes(V1, V7)) +
  geom_path(size = 0.1, color = "darkgrey") +
  theme_classic() +
  theme(aspect.ratio = 1)

a3 <- ggplot(encoded, aes(V2, V7)) +
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

model %>% save_model_hdf5("geyser-lstm.hdf5")
model <- load_model_hdf5("geyser-lstm.hdf5")

test_batch <- as_iterator(ds_test) %>% iter_next()

prediction_lstm <- model %>% predict(ds_test)

mse_lstm <- get_mse(test_batch, prediction_lstm)

# 0.5
# > mse_fnn
# [1] 0.1794896 0.4213111 0.5680545 0.6563896 0.7078483 0.7351122 0.7399738 0.7351620
# 0.7
#> mse_fnn
#[1] 0.1743249 0.4184542 0.5609179 0.6515022 0.7126671 0.7483752 0.7556070 0.7468666


# > mse_lstm
# [1] 0.9025139 0.8522131 0.8259575 0.8110787 0.7985898 0.7847978 0.7710731 0.7590278 0.7491193
# [10] 0.7407346 0.7330293 0.7253728 0.7152035 0.7041104 0.6950241 0.6877765 0.6808039 0.6736727
# [19] 0.6670824 0.6618595 0.6582018 0.6548377 0.6514161 0.6487431 0.6470793 0.6453608 0.6424494
# [28] 0.6387382 0.6350009 0.6318269 0.6298718 0.6298243 0.6314871 0.6337807 0.6351997 0.6350761
# [37] 0.6338936 0.6321963 0.6313722 0.6315157 0.6315492 0.6318302 0.6319601 0.6319010 0.6319674
# [46] 0.6325347 0.6330479 0.6348631 0.6362745 0.6374549 0.6383519 0.6389980 0.6392849 0.6391275
# [55] 0.6387466 0.6380682 0.6373467 0.6367899 0.6366458 0.6368802




  
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

compare_preds_df <- bind_rows(given, lstm)#, fnn


plots <- purrr::map(sample(1: dim(compare_preds_df)[2], 16), 
                    function(v) ggplot(compare_preds_df, aes(num, .data[[paste0("X", v)]], color = type)) + geom_line() )


plot_grid(plotlist = plots, ncol = 4)

plots[[2]]
