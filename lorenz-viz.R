
library(tidyverse)
library(cowplot)
library(gganimate)

source("lorenz-data.R")

# plot --------------------------------------------------------------------

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
  geom_point(size = 4, color = "#B57EDC") +
  theme_void()  +
  coord_equal() +
  transition_time(time = time)

#animate(x_z_anim, nframes = 10000, fps = 50, renderer = gifski_renderer())
#anim_save("x_z.gif")


