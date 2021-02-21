require(data.table)
require(tensorflow)
require(tfprobability)
require(keras)
require(ggplot2)

tf$compat$v1$enable_v2_behavior()

# Data generating function:

test1 <- function(x,sx=0.3) { 
  x <- x
  ifelse(x < 0.5, (pi**sx)*(1.2*exp(-(x-0.2)^2/sx^2+0.8*exp(-(x-0.7)^2/sx^2))),
         (pi**sx)*(1.2*exp(-(x-0.2)^2/sx^2+0.8*exp(-(x-0.7)^2/sx^2))) + 1.5)
}

# Number of samples for training data:
n <- 1000

samp <- data.frame(x = runif(n), y = 0)
samp$y <- test1(samp$x) + (rnorm(n)*0.2)
#samp[samp$x > 0.5, "y"] <- samp[samp$x > 0.5, "y"]+2
plot(samp)


# The true surface:
surf <- data.frame(x = seq(0,1,length=100), truth = 0)
surf$truth <- test1(surf$x)

lines(surf, col = "red")

#### BNN solution ####

time <- Sys.time()

bnn_samp <- samp
samp_mean <- mean(samp$y)
samp_sd <- sd(samp$y)
bnn_samp$y <- (samp$y-samp_mean)/samp_sd

x <- as.matrix(bnn_samp$x)
y <- samp$y

input <- layer_input(shape = c(1))

width = 256
droprate <- 1/8
act <- "relu"

kernel_ini <- initializer_he_uniform()
bias_ini <- initializer_random_uniform(minval = -3, maxval = +2)
#bias_ini <- initializer_random_normal(stddev = 1)


output <- input %>%
  layer_dense(units = width, activation = act, 
              kernel_initializer = kernel_ini, bias_initializer = bias_ini) %>%
  layer_dropout(rate = droprate) %>%
  layer_dense(units = width, activation = act,
              kernel_initializer = kernel_ini, bias_initializer = bias_ini) %>%
  layer_dropout(rate = droprate) %>%
  layer_dense(units = width, activation = act,
              kernel_initializer = kernel_ini, bias_initializer = bias_ini) %>%
  layer_dropout(rate = droprate) %>%
  layer_dense(units = 2, activation = "linear", 
              kernel_initializer = kernel_ini, bias_initializer = bias_ini) %>%
  layer_distribution_lambda(function(x)
    tfd_normal(loc = x[, 1, drop = FALSE],
               scale = 1e-3 + tf$math$softplus(0.01 * x[, 2, drop = FALSE])
    )
  )

bnnmodel <- keras_model(input, output)
bnnmodel %>% save_model_weights_hdf5("bnnweights.hdf5", overwrite = TRUE)

# Sample the initialised prior
n.sims <- 100

i <- 1
for(i in 1:n.sims){
  bnnmodel %>% load_model_weights_hdf5("bnnweights.hdf5", by_name = FALSE, skip_mismatch = FALSE, reshape = FALSE)
  weights <- bnnmodel %>% get_weights()
  
  dropweights <- weights
  #str(dropweights)
  
  maskl1 <- rbinom(dim(dropweights[[1]])[2], 1, 1-droprate) # generate dropout mask for kernel and bias of layer 1
  maskl2 <- rbinom(dim(dropweights[[3]])[2], 1, 1-droprate) # generate dropout mask for kernel and bias of layer 2
  maskl3 <- rbinom(dim(dropweights[[5]])[2], 1, 1-droprate) # generate dropout mask for kernel and bias of layer 3
  
  dropweights[[1]] <- t(t(dropweights[[1]]) * maskl1)*(1/(1-droprate))
  dropweights[[2]] <- (dropweights[[2]] * maskl1)*(1/(1-droprate))
  
  dropweights[[3]] <- t(t(dropweights[[3]]) * maskl2)*(1/(1-droprate))
  dropweights[[4]] <- (dropweights[[4]] * maskl2)*(1/(1-droprate))
  
  dropweights[[5]] <- t(t(dropweights[[5]]) * maskl3)*(1/(1-droprate))
  dropweights[[6]] <- (dropweights[[6]] * maskl3)*(1/(1-droprate))
  
  bnnmodel %>% set_weights(dropweights)
  
  preds <- as.data.table(data.frame(x = c(seq(-1.5,0, length.out = 100),surf$x,seq(1,2.5, length.out = 100)), draw = i))
  preds[, y := as.numeric(tfd_mean(bnnmodel(as.matrix(x))))]
  
  if(i == 1){
    sim <- copy(preds)
  }else{
    sim <- rbindlist(list(sim, preds), use.names = FALSE, fill = FALSE, idcol = NULL)
  }
}

bnnints <- copy(sim[, list(upper = quantile(y, 0.975), lower = quantile(y, 0.025)), by = x])

paste("BNN took", round(Sys.time() - time, 0), "seconds")

coverage.bnn <- copy(merge(sim, bnnints)[, list(cov = sum(y > lower & y < upper)/.N), by = x])
mean(coverage.bnn$cov)

# Plot the resulting prediction intervals
ggplot() + geom_point(data = samp, aes(x = x, y = y), shape = 16, alpha = 0.33) + 
  geom_line(data = surf, aes(x = x, y = truth), col = "red") +
  geom_line(data = sim, aes(x = x, y = y, group = draw, col = draw), alpha = 0.25) +
  theme_bw() +
  ggtitle("Dropout NN - prior mean function draws")
ggsave("Dropout NN - prior mean function draws.png", units = "mm", width = 160, height = 100, type = "cairo", scale = 1.5)


negloglik <- function(y, bnnmodel) - (bnnmodel %>% tfd_log_prob(y))
bnnmodel %>% compile(optimizer = optimizer_adam(lr = 0.001), loss = negloglik)
# bnnmodel %>% compile(optimizer = optimizer_adam(lr = 0.01), loss = "mse")
history <- fit(bnnmodel, x, y, epochs = 400,
               shuffle = TRUE, batch_size = 64,
               callbacks = callback_early_stopping(monitor = "loss", patience = 100, restore_best_weights = TRUE))
bnnmodel %>% save_model_weights_hdf5("bnnweights.hdf5", overwrite = TRUE)

round(min(history$metrics$loss), 5)

print(Sys.time() - time)

mean <- data.frame(x = c(seq(-1,0, length.out = 10),surf$x,seq(1,2, length.out = 10)),
                   y = as.numeric(tfd_mean(bnnmodel(as.matrix(c(seq(-1,0, length.out = 10),surf$x,seq(1,2, length.out = 10)))))))

ggplot(mean) + geom_point(data = samp, aes(x = x, y = y), shape = 16, alpha = 0.33) +
  geom_line(data = surf, aes(x = x, y = truth), col = "red") +
  geom_line(aes(x = x, y = y), col = "darkblue") +
  theme_bw() +
  ggtitle("Dropout NN - all nodes on")
#ggsave("Dropout NN - all nodes on.png", units = "mm", width = 160, height = 100, type = "cairo", scale = 1.5)

weights <- bnnmodel %>% get_weights()
str(weights)

# Predict from the BNN
n.sims <- 250

i <- 1
for(i in 1:n.sims){
  bnnmodel %>% load_model_weights_hdf5("bnnweights.hdf5", by_name = FALSE, skip_mismatch = FALSE, reshape = FALSE)
  weights <- bnnmodel %>% get_weights()
  
  dropweights <- weights
  #str(dropweights)
  
  maskl1 <- rbinom(dim(dropweights[[1]])[2], 1, 1-droprate) # generate dropout mask for kernel and bias of layer 1
  maskl2 <- rbinom(dim(dropweights[[3]])[2], 1, 1-droprate) # generate dropout mask for kernel and bias of layer 2
  maskl3 <- rbinom(dim(dropweights[[5]])[2], 1, 1-droprate) # generate dropout mask for kernel and bias of layer 3
  
  dropweights[[1]] <- t(t(dropweights[[1]]) * maskl1)*(1/(1-droprate))
  dropweights[[2]] <- (dropweights[[2]] * maskl1)*(1/(1-droprate))
  
  dropweights[[3]] <- t(t(dropweights[[3]]) * maskl2)*(1/(1-droprate))
  dropweights[[4]] <- (dropweights[[4]] * maskl2)*(1/(1-droprate))
  
  dropweights[[5]] <- t(t(dropweights[[5]]) * maskl3)*(1/(1-droprate))
  dropweights[[6]] <- (dropweights[[6]] * maskl3)*(1/(1-droprate))
  
  bnnmodel %>% set_weights(dropweights)
  
  preds <- as.data.table(data.frame(x = c(seq(-1.5,0, length.out = 100),surf$x,seq(1,2.5, length.out = 100)), draw = i))
  preds[, y := as.numeric(tfd_mean(bnnmodel(as.matrix(x))))]
  
  if(i == 1){
    sim <- copy(preds)
  }else{
    sim <- rbindlist(list(sim, preds), use.names = FALSE, fill = FALSE, idcol = NULL)
  }
}

bnnints <- copy(sim[, list(upper = quantile(y, 0.975), lower = quantile(y, 0.025)), by = x])

paste("BNN took", round(Sys.time() - time, 0), "seconds")

coverage.bnn <- copy(merge(sim, bnnints)[, list(cov = sum(y > lower & y < upper)/.N), by = x])
mean(coverage.bnn$cov)

sim[, y_mean := mean(y), by = "x"]

# Plot the resulting prediction intervals
ggplot() + geom_point(data = samp, aes(x = x, y = y), shape = 16, alpha = 0.33) + 
  geom_line(data = surf, aes(x = x, y = truth), col = "red") +
  geom_line(data = sim, aes(x = x, y = y, group = draw, col = draw), alpha = 0.25) +
  # geom_line(data = sim, aes(x = x, y = y_mean)) +
  # geom_line(data = mean, aes(x = x, y = y)) +
  theme_bw() +
  ggtitle("Dropout NN - posterior mean function draws")
ggsave("Dropout NN - posterior mean function draws.png", units = "mm", width = 160, height = 100, type = "cairo", scale = 1.5)
