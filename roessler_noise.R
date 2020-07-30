
# conditions --------------------------------------------------------------

noise <- 1.5
fnn_multiplier <- 1
infix <- if (fnn_multiplier == 0) "nofnn_" else ""

# -------------------------------------------------------------------------


library(reticulate)
source("models.R")
source("training.R")
source("utils.R")
library(tidyverse)
library(cowplot)
library(deSolve)

parameters <- c(a = .2,
                b = .2,
                c = 5.7)
initial_state <-
  c(x = 1,
    y = 1,
    z = 1.05)

roessler <- function(t, state, parameters) {
  with(as.list(c(state, parameters)), {
    dx <- -y - z
    dy <- x + a * y
    dz = b + z * (x - c)
    
    list(c(dx, dy, dz))
  })
}

times <- seq(0, 2500, length.out = 20000)

roessler_ts <-
  ode(
    y = initial_state,
    times = times,
    func = roessler,
    parms = parameters,
    method = "lsoda"
  ) %>% unclass() %>% as_tibble()

# original variance
vars <- roessler_ts %>%
  summarise_all(var)

# attractors, projected to 2d
y_z <- ggplot(roessler_ts %>%
                select(y, z) %>%
                slice(1:10000),
              aes(y, z)) +
  geom_path(size = 0.1, color = "#c7aabc") +
  theme_void() +
  theme(aspect.ratio = 1)

x_z <- ggplot(roessler_ts %>%
                select(x, z) %>%
                slice(1:10000),
              aes(x, z)) +
  geom_path(size = 0.1, color = "#b8e3ea") +
  theme_void() +
  theme(aspect.ratio = 1)

x_y <- ggplot(roessler_ts %>%
                select(x, y) %>%
                slice(1:10000),
              aes(x, y)) +
  geom_path(size = 0.1, color = "#a9bcc6") +
  theme_void() +
  theme(aspect.ratio = 1)

plot_grid(x_y, y_z, x_z, ncol = 3)

n <- 10000
roessler <- roessler_ts$x[1:n]
n_timesteps <- 120
batch_size <- 32

roessler <- scale(roessler)
data.frame(x = roessler[1:1000]) %>%
  ggplot(aes(1:1000, x)) +
  geom_line() +
  theme_classic()

roessler_orig <- roessler
roessler <- roessler + rnorm(10000, mean = 0, sd = noise)

data.frame(x = roessler[1:1000]) %>%
  ggplot(aes(1:1000, x)) +
  geom_line() +
  theme_classic()

train <- gen_timesteps(roessler[1:(n/2)], 2 * n_timesteps)
test <- gen_timesteps(roessler[(n/2):n], 2 * n_timesteps) 
test_orig <- gen_timesteps(roessler_orig[(n/2):n], 2 * n_timesteps)

dim(train) <- c(dim(train), 1)
dim(test) <- c(dim(test), 1)
dim(test_orig) <- c(dim(test_orig), 1)

x_train <- train[ , 1:n_timesteps, , drop = FALSE]
y_train <- train[ , (n_timesteps + 1):(2*n_timesteps), , drop = FALSE]

ds_train <- tensor_slices_dataset(list(x_train, y_train)) %>%
  dataset_shuffle(nrow(x_train)) %>%
  dataset_batch(batch_size)

x_test <- test[ , 1:n_timesteps, , drop = FALSE]
y_test <- test[ , (n_timesteps + 1):(2*n_timesteps), , drop = FALSE]
x_test_orig <- test_orig[ , 1:n_timesteps, , drop = FALSE]
y_test_orig <- test_orig[ , (n_timesteps + 1):(2*n_timesteps), , drop = FALSE]

ds_test <- tensor_slices_dataset(list(x_test, y_test)) %>%
  dataset_batch(nrow(x_test))
ds_test_noise_and_orig <- tensor_slices_dataset(
  list(x_test, y_test, x_test_orig, y_test_orig)) %>%
  dataset_batch(nrow(x_test_orig))


# vae -------------------------------------------------------------------

n_latent <- 10L
n_features <- 1

encoder <- vae_encoder_model(n_timesteps,
                         n_features,
                         n_latent)

decoder <- vae_decoder_model(n_timesteps,
                         n_features,
                         n_latent)
mse_loss <-
  tf$keras$losses$MeanSquaredError(reduction = tf$keras$losses$Reduction$SUM)

train_loss <- tf$keras$metrics$Mean(name = 'train_loss')
train_fnn <- tf$keras$metrics$Mean(name = 'train_fnn')
train_mse <-  tf$keras$metrics$Mean(name = 'train_mse')
train_kl <-  tf$keras$metrics$Mean(name = 'train_kl')

fnn_weight <- fnn_multiplier * nrow(x_train)/batch_size

kl_weight <- 1

optimizer <- optimizer_adam(lr = 1e-3)


# for (epoch in 1:100) {
#    cat("Epoch: ", epoch, " -----------\n")
#    training_loop_vae(ds_train)
# 
#    test_batch <- as_iterator(ds_test) %>% iter_next()
#    encoded <- encoder(test_batch[[1]][1:1000])
#    test_var <- tf$math$reduce_variance(encoded, axis = 0L)
#    print(test_var %>% as.numeric() %>% round(5))
# }
# 
# encoder %>% save_model_weights_tf(paste0("roessler_encoder__nofnn_vae_", noise))
# decoder %>% save_model_weights_tf(paste0("roessler_decoder_nofnn_vae_", noise))

encoder %>% load_model_weights_tf(paste0("roessler_encoder_", infix, "vae_", noise))
decoder %>% load_model_weights_tf(paste0("roessler_decoder_", infix, "vae_", noise))


# check variances -------------------------------------------------------------

test_batch <- as_iterator(ds_test) %>% iter_next()
encoded <- encoder(test_batch[[1]]) %>%
  as.array() %>%
  as_tibble()

encoded %>% summarise_all(var)
# 0
# 224.  135. 0.00434 0.0000202 0.0195 0.00260  1.14  31.8 0.0172  20.9

#1
#  207.  39.3 2.96e-12 2.79e-10      1.70e-9 9.51e-11      1.58e-9 8.08e-12 6.29e-10 4.28e-10

# 1.5
#  142. 3.36e-11 5.46e-12      1.05e-9  18.5      3.33e-9 1.52e-10 9.80e-12 5.85e-10 9.21e-10

# 2
# 208. 4.04e-10      1.68e-9 3.15e-12      1.00e-9 1.29e-10  39.9 2.46e-10 3.55e-10 7.14e-10

#2.5
# 267. 8.20e-11 3.85e-12  51.3 9.84e-10 3.74e-10 2.93e-10 5.29e-10 0.00000000413 0.00000000125



# predict -----------------------------------------------------------------
test_batch_noise_and_orig <- as_iterator(ds_test_noise_and_orig) %>% iter_next()

prediction_vae <- decoder(encoder(test_batch_noise_and_orig[[1]]))

mse_vae_actual <- get_mse(test_batch_noise_and_orig, prediction_vae)
mse_vae_latent <- get_mse(test_batch_noise_and_orig, prediction_vae, FALSE)

mses <- data.frame(actual = mse_vae_actual, latent = mse_vae_latent)
saveRDS(mses, paste0("mses_vae_", noise, "_", infix, ".rds"))


# lstm encoder ------------------------------------------------------------


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


fnn_multiplier <- 1
fnn_weight <- fnn_multiplier * nrow(x_train)/batch_size

optimizer <- optimizer_adam(lr = 1e-3)

# for (epoch in 1:100) {
#   cat("Epoch: ", epoch, " -----------\n")
#   training_loop(ds_train)
# 
#   test_batch <- as_iterator(ds_test) %>% iter_next()
#   encoded <- encoder(test_batch[[1]][1:1000])
#   test_var <- tf$math$reduce_variance(encoded, axis = 0L)
#   print(test_var %>% as.numeric() %>% round(5))
# }

# encoder %>% save_model_weights_tf(paste0("roessler_encoder_nofnn_lstm_", noise))
# decoder %>% save_model_weights_tf(paste0("roessler_decoder_nofnn_lstm_", noise))


encoder %>% load_model_weights_tf(paste0("roessler_encoder_", infix, "lstm_", noise))
decoder %>% load_model_weights_tf(paste0("roessler_decoder_", infix, "lstm_", noise))


encoded <- encoder(test_batch_noise_and_orig[[1]]) %>%
  as.array() %>%
  as_tibble()

encoded %>% summarise_all(var)

#0
#0.148 0.00799 0.0000279    3.72e-8 0.000122    1.84e-9  9.29e-6 1.00e-12   8.27e-8 4.44e-10
#1
#0.108 0.0915 0.000344    6.83e-8    3.27e-8    1.32e-8 0.000110     5.89e-9 1.34e-4 1.66e-4
# 2
#0.288 0.00459   1.36e-8 2.57e-11 1.12e-10   4.97e-7   1.10e-8    2.38e-9  9.08e-7   5.35e-9
#2.5
#0.210 0.0186 3.72e-10 0.00000000187 6.06e-11 0.0000000153 2.79e-13 7.65e-13 6.85e-10 1.75e-10


prediction_lstm <- decoder(encoder(test_batch_noise_and_orig[[1]]))

mse_lstm_actual <- get_mse(test_batch_noise_and_orig, prediction_lstm)
mse_lstm_latent <- get_mse(test_batch_noise_and_orig, prediction_lstm, FALSE)

mses <- data.frame(actual = mse_lstm_actual, latent = mse_lstm_latent)
saveRDS(mses, paste0("mses_lstm_", noise, "_", infix, ".rds"))



# plot --------------------------------------------------------------------


given <- data.frame(
  as.array(
    tf$concat(list(test_batch_noise_and_orig[[1]][ , , 1], test_batch_noise_and_orig[[2]][ , , 1]),
              axis = 1L)) %>% t()) %>% 
  add_column(type = "given") %>%
  add_column(num = 1:(2 * n_timesteps))

orig <- data.frame(
  as.array(
    tf$concat(list(test_batch_noise_and_orig[[3]][ , , 1], test_batch_noise_and_orig[[4]][ , , 1]),
              axis = 1L)) %>% t()) %>% 
  add_column(type = "orig") %>%
  add_column(num = 1:(2 * n_timesteps))

lstm <- data.frame(as.array(prediction_lstm[ , , 1]) %>% 
                    t()) %>%
  add_column(type = "lstm") %>%
  add_column(num = (n_timesteps  +1):(2 * n_timesteps))

vae <- data.frame(as.array(prediction_vae[ , , 1]) %>% 
                     t()) %>%
  add_column(type = "vae") %>%
  add_column(num = (n_timesteps  +1):(2 * n_timesteps))

compare_preds_df <- bind_rows(given, orig, lstm, vae)

#indices <- sample(1: dim(compare_preds_df)[2], 16)
indices <- c(1405, 4336, 50, 538, 2013, 3459, 1998, 1906, 1795, 3960, 666, 1430, 2388, 2759, 2227, 4668)
plots <- purrr::map(indices,
                    function(v) ggplot(compare_preds_df, aes(num, .data[[paste0("X", v)]], color = type)) +
                      geom_line() + 
                      theme_classic() + 
                      theme(legend.position = "none", axis.title = element_blank()) +
                      scale_color_manual(values=c("#bbbbdd", "#FF7F00", "#00FF7F", "#593780")))


plot_grid(plotlist = plots, ncol = 4)


# mses ---------------------------------------------------------------------

plots_actual <- list()
plots_latent <- list()

noisevals <- c("1", "1.5", "2", "2.5")

for (n in seq_along(noisevals)) {
  
  noise <- noisevals[[n]]
  print(n)
  print(noisevals[[n]])
  
  vae_nofnn <-
    readRDS(paste0("mses_vae_", noise, "_", "nofnn_", ".rds"))
  vae_fnn <- readRDS(paste0("mses_vae_", noise, "_.rds"))
  lstm_nofnn <-
    readRDS(paste0("mses_lstm_", noise, "_", "nofnn_", ".rds"))
  lstm_fnn <- readRDS(paste0("mses_lstm_", noise, "_.rds"))

  msecmp_actual <- data.frame(
    timestep = 1:119,
    nofnn_vae = vae_nofnn$actual[-120],
    fnn_vae = vae_fnn$actual[-120],
    nofnn_lstm = lstm_nofnn$actual[-120],
    fnn_lstm = lstm_fnn$actual[-120]
  )

  plots_actual[[n]] <- msecmp_actual %>%
    gather(key = "type", value = "mse",-timestep) %>%
    mutate(mse = mse / max(mse)) %>%
    ggplot(aes(x = timestep, y = mse, color = type)) +
    geom_point(size = 0.5) +
    theme_classic() +
    theme(legend.position = "none", axis.title = element_blank()) +
    scale_color_manual(values = c("#00FF7F", "#593780", "#FF7F00", "#bbbbdd")) +
    coord_cartesian(ylim = c(0, 1))


  msecmp_latent <- data.frame(
    timestep = 1:119,
    nofnn_vae = vae_nofnn$latent[-120],
    fnn_vae = vae_fnn$latent[-120],
    nofnn_lstm = lstm_nofnn$latent[-120],
    fnn_lstm = lstm_fnn$latent[-120]
  )

  plots_latent[[n]] <- msecmp_latent %>%
    gather(key = "type", value = "mse",-timestep) %>%
    mutate(mse = mse / max(mse)) %>%
    ggplot(aes(x = timestep, y = mse, color = type)) +
    geom_point(size = 0.5) +
    theme_classic() +
    theme(legend.position = "none", axis.title = element_blank()) +
    scale_color_manual(values = c("#00FF7F", "#593780", "#FF7F00", "#bbbbdd")) +
    coord_cartesian(ylim = c(0, 1))

}

plots <- c(rbind(plots_actual, plots_latent))
do.call("plot_grid", c(plots, ncol = 2))
