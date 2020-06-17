library(deSolve)
library(tidyverse)
library(cowplot)
library(gganimate)


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


lorenz_ts[1:10,]

# original variance
vars <- lorenz_ts %>%
  summarise_all(var)


# plot --------------------------------------------------------------------

# every 10th x
obs <- lorenz_ts %>%
  select(time, x) %>%
  filter(row_number() %% 10 == 0)

ggplot(obs, aes(time, x)) +
  geom_line() +
  coord_cartesian(xlim = c(0, 100)) +
  theme_classic()

# attractors, projected to 2d
y_z <- ggplot(lorenz_ts %>%
                select(y, z) %>%
                slice(1:10000),
              aes(y, z)) +
  geom_path(size = 0.2) +
  coord_cartesian(xlim = c(-25, 25), ylim = c(0, 50)) +
  theme_classic() +
  theme(aspect.ratio = 1)

x_z <- ggplot(lorenz_ts %>%
                select(x, z) %>%
                slice(1:10000),
              aes(x, z)) +
  geom_path(size = 0.2) +
  coord_cartesian(xlim = c(-25, 25), ylim = c(0, 50)) +
  theme_classic() +
  theme(aspect.ratio = 1)

x_y <- ggplot(lorenz_ts %>%
                select(x, y) %>%
                slice(1:10000),
              aes(x, y)) +
  geom_path(size = 0.2) +
  coord_cartesian(xlim = c(-25, 25), ylim = c(-25, 25)) +
  theme_classic() +
  theme(aspect.ratio = 1)

plot_grid(x_y, y_z, x_z, ncol = 3)



# animate -----------------------------------------------------------------


x_z_anim <- ggplot(lorenz_ts %>%
                     select(time, x, z) %>%
                     slice(1:10000),
                   aes(x, z)) +
  geom_path(
    data = lorenz_ts %>%
      select(x, z) %>%
      slice(1:10000),
    aes(x, z),
    size = 0.4,
    color = "darkgrey"
  ) +
  geom_point(size = 2, color = "violet") +
  theme_void()  +
  coord_equal() +
  transition_time(time = time)

# animate(x_z_anim, nframes = 10000, fps = 50, renderer = gifski_renderer())
# anim_save("x_z.gif")
