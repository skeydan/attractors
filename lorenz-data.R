library(deSolve)
library(tidyverse)
library(tfdatasets)

source("utils.R")


# generate data -----------------------------------------------------------


parameters <- c(sigma = 10,
                rho = 28,
                beta = 8 / 3)
initial_state <-
  c(x = -8.60632853,
    y = -14.85273055,
    z = 15.53352487)

lorenz <- function(t, state, parameters) {
  with(as.list(c(state, parameters)), {
    dx <- sigma * (y - x)
    dy <- x * (rho - z) - y
    dz <- x * y - beta * z
    
    list(c(dx, dy, dz))
  })
}

times <- seq(0, 500, length.out = 125000)

lorenz_ts <-
  ode(
    y = initial_state,
    times = times,
    func = lorenz,
    parms = parameters,
    method = "lsoda"
  ) %>% as_tibble()

# original variance
vars <- lorenz_ts %>%
  summarise_all(var)



# Preprocess --------------------------------------------------------------

n_timesteps <- 8
batch_size <- 100

# every 10th x
obs <- lorenz_ts %>%
  select(time, x) %>%
  filter(row_number() %% 10 == 0)

obs <- obs %>% mutate(
  x = scale(x)
)

n <- nrow(obs)

train <- gen_timesteps(as.matrix(obs$x)[1:(n/2)], 2 * n_timesteps)
test <- gen_timesteps(as.matrix(obs$x)[(n/2):n], 2 * n_timesteps) 

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


