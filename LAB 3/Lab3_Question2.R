#### BAYESIAN LEARNING LAB 3 ####
######### QUESTION 2 ############

######## PART A ##########
library(mvtnorm)

# Load the data
data <- read.table("eBayNumberOfBidderData_2025.dat", header = TRUE)

# Fit the Poisson regression model (Note: Don't include "Const" as glm adds 
# an intercept)
model_mle <- glm(nBids ~ . - Const, data = data, family = poisson())

# Summary of the model
summary(model_mle)

## Observation 
# The features which are significant in this model are- "Sealed" and 
# "MinBidShare".


######## PART B #########
# Define log-posterior function for Poisson regression
log_posterior <- function(beta, X, y) {
  eta <- X %*% beta
  log_likelihood <- sum(y * eta - exp(eta))
  prior_precision <- solve(100 * solve(t(X) %*% X))
  log_prior <- -0.5 * t(beta) %*% prior_precision %*% beta
  return(- (log_likelihood + log_prior))  # Negate for minimization
}

# Prepare data
X <- as.matrix(data[, -which(names(data) == "nBids" | names(data) == "Const")])
X <- cbind(1, X)  # Add intercept manually
y <- data$nBids
p <- ncol(X)

# Initial guess
beta_init <- rep(0, p)

# Optimization to find posterior mode
opt_result <- optim(beta_init, log_posterior, X = X, y = y, method = "BFGS", hessian = TRUE)

# Posterior mode and covariance
posterior_mode <- opt_result$par
posterior_cov <- solve(opt_result$hessian)

# Output
cat("Posterior mode:\n")
print(posterior_mode)
cat("Posterior covariance matrix:\n")
print(posterior_cov)


####### PART C) ########
RWMSampler <- function(logPostFunc, theta0, nIter, proposalCov, c = 1, ...) {
  p <- length(theta0)
  samples <- matrix(NA, nrow = nIter, ncol = p)
  samples[1, ] <- theta0
  logPostOld <- logPostFunc(theta0, ...)
  
  for (i in 2:nIter) {
    # Propose new theta
    thetaProp <- rmvnorm(1, mean = samples[i - 1, ], sigma = c * proposalCov)
    
    # Compute log posterior for proposal
    logPostNew <- logPostFunc(as.numeric(thetaProp), ...)
    
    # Acceptance probability
    logAlpha <- logPostNew - logPostOld
    if (log(runif(1)) < logAlpha) {
      samples[i, ] <- thetaProp
      logPostOld <- logPostNew
    } else {
      samples[i, ] <- samples[i - 1, ]
    }
  }
  
  return(samples)
}

set.seed(123)
samples <- RWMSampler(log_posterior, theta0 = posterior_mode, nIter = 1000,
                      proposalCov = posterior_cov, c = 0.8, X = X, y = y)

# Assessing MCMC Convergence
par(mfrow = c(3, 3))
for (j in 1:p) {
  plot(samples[, j], type = "l", main = paste("Trace of beta", j))
}


##### PART D) #######
# Construct new covariate vector (match order of X)
# Includes intercept manually
x_new <- c(1, 1, 0, 1, 0, 1, 0, 1.3, 0.7)  # [Intercept, PowerSeller, ..., MinBidShare]

# Number of posterior samples
n_samples <- nrow(samples)

# Compute predicted lambda for each posterior sample
lambda_new <- numeric(n_samples)
for (i in 1:n_samples) {
  beta_i <- samples[i, ]
  lambda_new[i] <- exp(sum(x_new * beta_i))
}

# Simulate predictive y values
y_pred <- rpois(n_samples, lambda = lambda_new)

# Plot predictive distribution
hist(y_pred, breaks = 30, col = "skyblue", main = "Predictive Distribution",
     xlab = "Number of Bidders", ylab = "Frequency")

# Estimate P(y_new = 0)
p_zero <- mean(y_pred == 0)
cat("Estimated P(y_new = 0):", p_zero, "\n")

## Observation - 
# The probability of no bidders in this new model is 0.378. 

