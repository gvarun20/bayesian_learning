###### BAYESIAN LEARNING ######
############ LAB 3 ############
######### QUESTION 1 ##########

##### PART A) 
library(BayesLogit) # for Polya-Gamma sampling
library(mvtnorm)

# Load dataset: Disease.xlsx (from Lab 2)
data <- read.csv("Disease.csv")
X <- model.matrix(~ gender + age + duration_of_symptoms + dyspnoea + white_blood, data)
y <- as.numeric(data$class_of_diagnosis)

n <- nrow(X)
p <- ncol(X)
tau <- 3

# Gibbs Sampler 
gibbs_logistic_pg <- function(X, y, tau = 3, n_iter = 500) {
  n <- nrow(X)
  p <- ncol(X)
  
  beta_samples <- matrix(0, nrow = n_iter, ncol = p)
  beta <- rep(0, p)
  
  for (iter in 1:n_iter) {
    # Step 1: Sample omega_i ~ PG(1, x_i^T beta)
    eta <- X %*% beta
    omega <- rpg(n, h = 1, z = eta)
    Omega <- diag(omega)
    
    # Step 2: Sample beta from multivariate normal
    B_inv <- diag(1 / tau^2, p)
    V_inv <- t(X) %*% Omega %*% X + B_inv
    V <- solve(V_inv)
    kappa <- y - 0.5
    m <- V %*% (t(X) %*% kappa)
    
    beta <- as.vector(rmvnorm(1, mean = m, sigma = V))
    beta_samples[iter, ] <- beta
  }
  
  return(beta_samples)
}

samples <- gibbs_logistic_pg(X, y, n_iter = 500)

# Plot trajectories
par(mfrow = c(2, 4))
for (j in 1:ncol(samples)) {
  plot(samples[, j], type = 'l', main = paste("Beta", j))
}

# # Calculate inefficiency factors using coda
# library(coda)
# mcmc_obj <- as.mcmc(samples)
# ineff_factors <- apply(samples, 2, function(chain) effectiveSize(chain))
# ineff_factors

# Computing inefficiency factor (IF) for each parameter
ineff_factors <- apply(samples, 2, function(chain) {
  acf_vals <- acf(chain, plot = FALSE)$acf[-1]  # exclude lag 0
  1 + 2 * sum(acf_vals)
})
ineff_factors


##### PART B)
set_sizes <- c(10, 40, 80)
results <- list()

for (m in set_sizes) {
  X_sub <- X[1:m, ]
  y_sub <- y[1:m]
  samples_sub <- gibbs_logistic_pg(X_sub, y_sub, n_iter = 500)
  results[[as.character(m)]] <- samples_sub
}
results


