library(tidyverse)

logistic <- function(x, r) {
  r * x * (1 - x)
}


gen_trajectory <- function(n_it, x0, r) {
  res <- vector(length = n_it, mode = "numeric")
  x <- x0
  res[1] <- x0
  for (i in 2:n_it) {
    x <- logistic(x, r)
    res[i] <- x
  }
  res
}

n_it <- 100

df <- data.frame(
  r2.1_x0.3 = gen_trajectory(n_it, 0.3, 2.1),
  r2.1_x0.9 = gen_trajectory(n_it, 0.9, 2.1),
  r3.3_x0.3 = gen_trajectory(n_it, 0.3, 3.3),
  r3.3_x0.9 = gen_trajectory(n_it, 0.9, 3.3),
  r3.6_x0.3 = gen_trajectory(n_it, 0.3, 3.6),
  r3.6_x0.9 = gen_trajectory(n_it, 0.9, 3.6),
  r3.6_x0.301 = gen_trajectory(n_it, 0.301, 3.6),
  r3.6_x0.30000001 = gen_trajectory(n_it,
                                    0.30000001,
                                    3.6)) %>%
  add_column(iteration = factor(1:n_it))


df

df %>%
  slice(1:10) %>%
  select(iteration, r2.1_x0.9, r2.1_x0.3) %>%
  gather("condition", "x", -iteration) %>%
  ggplot(
    aes(
      iteration,
      x,
      color = condition)) +
  geom_point(size = 2) +
  theme_classic() +
  scale_color_manual(values = c("#40E0D0", "#B57EDC"))



df %>%
  slice(1:25) %>%
  select(iteration, r3.3_x0.9, r3.3_x0.3) %>%
  gather("condition", "x",-iteration) %>%
  ggplot(
    aes(
      iteration,
      x,
      color = condition)) +geom_point(size = 2) +
  theme_classic() +scale_color_manual(
    values = c(
      "#40E0D0",
      "#B57EDC"))


df %>%
  slice(1:100) %>%
  select(iteration, r3.6_x0.3, r3.6_x0.9) %>%
  gather("condition", "x",-iteration) %>%
  ggplot(
    aes(
      iteration,
      x,
      color = condition)) +geom_point(size = 2) +
  theme_classic() +scale_color_manual(
    values = c(
      "#40E0D0",
      "#B57EDC")) +scale_x_discrete(
        breaks = seq(
          0,
          100,
          10))


df %>%
  slice(1:25) %>%
  select(iteration, r3.6_x0.3, r3.6_x0.301) %>%
  gather("condition", "x",-iteration) %>%
  ggplot(
    aes(
      iteration,
      x,
      color = condition)) +geom_point(size = 2) +
  theme_classic() +scale_color_manual(
    values = c(
      "#40E0D0",
      "#B57EDC"))


df %>%
  slice(1:100) %>%
  select(iteration, r3.6_x0.3, r3.6_x0.30000001) %>%
  gather("condition", "x",-iteration) %>%
  ggplot(
    aes(
      iteration,
      x,
      color = condition)) +geom_point(size = 2, alpha = 0.5) +
  theme_classic() +scale_color_manual(
    values = c(
      "#40E0D0",
      "#B57EDC")) +scale_x_discrete(breaks = seq(0, 100, 10))
