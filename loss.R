loss_false_nn <- function(x) {
  
  # changing these parameters is equivalent to
  # changing the strength of the regularizer, so we keep these fixed (these values
  # correspond to the original values used in Kennel et al 1992).
  rtol <- 10 
  atol <- 2
  k_frac <- 0.01
  
  k <- max(1, floor(k_frac * batch_size))
  
  ## Vectorized version of distance matrix calculation
  tri_mask <-
    tf$linalg$band_part(
      tf$ones(
        shape = c(tf$cast(n_latent, tf$int32), tf$cast(n_latent, tf$int32)),
        dtype = tf$float32
      ),
      num_lower = -1L,
      num_upper = 0L
    )
  
  # latent x batch_size x latent
  batch_masked <-
    tf$multiply(tri_mask[, tf$newaxis,], x[tf$newaxis, reticulate::py_ellipsis()])
  
  # latent x batch_size x 1
  x_squared <-
    tf$reduce_sum(batch_masked * batch_masked,
                  axis = 2L,
                  keepdims = TRUE)
  
  # latent x batch_size x batch_size
  pdist_vector <- x_squared + tf$transpose(x_squared, perm = c(0L, 2L, 1L)) -
    2 * tf$matmul(batch_masked, tf$transpose(batch_masked, perm = c(0L, 2L, 1L)))
  
  #(latent, batch_size, batch_size)
  all_dists <- pdist_vector
  # latent
  all_ra <-
    tf$sqrt((1 / (
      batch_size * tf$range(1, 1 + n_latent, dtype = tf$float32)
    )) *
      tf$reduce_sum(tf$square(
        batch_masked - tf$reduce_mean(batch_masked, axis = 1L, keepdims = TRUE)
      ), axis = c(1L, 2L)))
  
  # Avoid singularity in the case of zeros
  #(latent, batch_size, batch_size)
  all_dists <-
    tf$clip_by_value(all_dists, 1e-14, tf$reduce_max(all_dists))
  
  #inds = tf.argsort(all_dists, axis=-1)
  top_k <- tf$math$top_k(-all_dists, tf$cast(k + 1, tf$int32))
  # (#(latent, batch_size, batch_size)
  top_indices <- top_k[[1]]
  
  #(latent, batch_size, batch_size)
  neighbor_dists_d <-
    tf$gather(all_dists, top_indices, batch_dims = -1L)
  #(latent - 1, batch_size, batch_size)
  neighbor_new_dists <-
    tf$gather(all_dists[2:-1, , ],
              top_indices[1:-2, , ],
              batch_dims = -1L)
  
  # Eq. 4 of Kennel et al.
  #(latent - 1, batch_size, batch_size)
  scaled_dist <- tf$sqrt((
    tf$square(neighbor_new_dists) -
      # (9, 8, 2)
      tf$square(neighbor_dists_d[1:-2, , ])) /
      # (9, 8, 2)
      tf$square(neighbor_dists_d[1:-2, , ])
  )
  
  # Kennel condition #1
  #(latent - 1, batch_size, batch_size)
  is_false_change <- (scaled_dist > rtol)
  # Kennel condition 2
  #(latent - 1, batch_size, batch_size)
  is_large_jump <-
    (neighbor_new_dists > atol * all_ra[1:-2, tf$newaxis, tf$newaxis])
  
  is_false_neighbor <-
    tf$math$logical_or(is_false_change, is_large_jump)
  #(latent - 1, batch_size, 1)
  total_false_neighbors <-
    tf$cast(is_false_neighbor, tf$int32)[reticulate::py_ellipsis(), 2:(k + 2)]
  
  # Pad zero to match dimensionality of latent space
  # (latent - 1)
  reg_weights <-
    1 - tf$reduce_mean(tf$cast(total_false_neighbors, tf$float32), axis = c(1L, 2L))
  # (latent,)
  reg_weights <- tf$pad(reg_weights, list(list(1L, 0L)))
  
  # Find batch average activity
  
  # L2 Activity regularization
  activations_batch_averaged <-
    tf$sqrt(tf$reduce_mean(tf$square(x), axis = 0L))
  
  loss <- tf$reduce_sum(tf$multiply(reg_weights, activations_batch_averaged))
  loss
  
}

