#Time series models in Stan
#Write a function in R that simulates data from the AR(1)-process
#xt=μ+φ(xt−1−μ)+εt,εt∼N0 , for given values of μ, φ and σ2. 
#Start the process at x1 = μ and then simulate values for xt for t = 2, 3 . . . , T 
#and return the vector x1:T containing all time points. 
#Use μ = 13, σ2 = 3 and T = 300 and look at some different realizations (simulations) of x1:T 
#for values of φ between −1 and 1 (this is the interval of φ where the AR(1)-process is stationary). 
#Include a plot of at least one realization in the report. What effect does the value of φ have on x1:T ?
#==================PART A=============================
# Load the required packages
library(rstan)  # For Bayesian modeling using Stan

# Enable auto-writing of compiled Stan models to avoid recompilation
rstan_options(auto_write = TRUE)

# Use all available CPU cores to parallelize MCMC sampling
options(mc.cores = parallel::detectCores())

# ================================
# PART 1: Simulating AR(1) Processes
# ================================

# Long-term mean of the AR(1) process
long_term_mean <- 5

# Standard deviation of white noise (σ = sqrt(variance))
noise_sd <- sqrt(9)

# Number of time points in the simulation
num_time_points <- 300

# Initial value of the process (can be set to the mean for stability)
initial_value <- long_term_mean

# Define a set of φ (phi) values to analyze different autocorrelation behaviors
phi_values <- seq(-1, 1, 0.25)

# AR(1) simulation function with fixed μ, σ, and φ
simulate_ar1 <- function(mu, sigma, T, x0, phi) {
  series <- numeric(T)     # Preallocate series
  series[1] <- x0          # Set the first value
  
  for (t in 2:T) {
    noise <- rnorm(1, mean = 0, sd = sigma)  # White noise
    series[t] <- mu + phi * (series[t - 1] - mu) + noise
  }
  
  return(series)
}

# Matrix to hold the AR(1) simulations for each phi value
simulated_series_matrix <- matrix(0, num_time_points, length(phi_values))

# Simulate AR(1) series for each phi value and store in the matrix
for (i in seq_along(phi_values)) {
  simulated_series_matrix[, i] <- simulate_ar1(
    long_term_mean, noise_sd, num_time_points, initial_value, phi_values[i])
}

# Plotting a few representative simulations to visualize AR(1) behavior
par(mfrow = c(2, 2))  # 2x2 plotting window

# φ = -1: Strong negative autocorrelation → sharp oscillations
plot(1:num_time_points, simulated_series_matrix[, 1], type = "l",
     xlab = "Time", ylab = "x_t", main = paste("phi =", phi_values[1]))
# -> Series flips direction every step, forming a zig-zag pattern

# φ = -0.5: Mild negative autocorrelation → smoother oscillations
plot(1:num_time_points, simulated_series_matrix[, 3], type = "l",
     xlab = "Time", ylab = "x_t", main = paste("phi =", phi_values[3]))
# -> Still oscillates but with reduced amplitude and smoother transition

# φ = 0.5: Mild positive autocorrelation → smooth, trending process
plot(1:num_time_points, simulated_series_matrix[, 7], type = "l",
     xlab = "Time", ylab = "x_t", main = paste("phi =", phi_values[7]))
# -> Values gently follow previous values, remaining near the mean

# φ = 1: Strong positive autocorrelation → random walk-like behavior
plot(1:num_time_points, simulated_series_matrix[, 9], type = "l",
     xlab = "Time", ylab = "x_t", main = paste("phi =", phi_values[9]))
# -> Series drifts slowly, barely returning to the mean (non-stationary)

OBSERVATION:The AR(1) series exhibits a drastic decrease in oscillations as the autoregressive coefficient ϕ
increases. For example, the series exhibits visible oscillatory patterns with abrupt, alternating changes when ϕ = -1.
On the other hand, the series exhibits smoother and more regular trends with 
fewer oscillating patterns as ϕ moves towards  positive values, especially around 1.


# ==============================================
# PART  B
#Use your function from a) to simulate two AR(1)-processes, x1:T with φ = 0.2 and y1:T with φ = 0.95. 
#Now, treat your simulated vectors as synthetic data, and treat the values of μ, φ and σ2 as unknown parameters. 
#Implement Stan code that samples from the posterior of the three parameters, 
#using suitable non-informative priors of your choice.
#[Hint: Look at the time-series models examples in the Stan user's guide/reference manual, 
#and note the dierent parameterization used here.]

#i. Report the posterior mean, 95% credible intervals and the number of effective posterior samples
#for the three inferred parameters for each of the simulated AR(1)-process. Are you able to estimate the true values?
#ii. For each of the two data sets, evaluate the convergence of the samplers
#and plot the joint posterior of μ and φ.Comments?
# ==============================================



# Simulate two time series: one with phi = 0.4 and another with phi = 0.98
series_x <- simulate_ar1(long_term_mean, noise_sd, num_time_points, initial_value, 0.4)
series_y <- simulate_ar1(long_term_mean, noise_sd, num_time_points, initial_value, 0.98)

# Define a Stan model for AR(1) process
stan_code <- '
data {
  int<lower=0> N;       // Number of time points
  vector[N] z;          // Observed AR(1) time series
}
parameters {
  real mu;              // Long-term mean
  real phi;             // Autoregressive coefficient
  real<lower=0> sigma2; // Variance of noise
}
model {
  for (i in 2:N)
    z[i] ~ normal(mu + phi * (z[i-1] - mu), sqrt(sigma2));  // AR(1) likelihood
}
'

# Prepare data and fit model to series_x (phi = 0.4)
data_x <- list(N = num_time_points, z = series_x)
fit_x <- stan(model_code = stan_code, data = data_x)

# Prepare data and fit model to series_y (phi = 0.98)
data_y <- list(N = num_time_points, z = series_y)
fit_y <- stan(model_code = stan_code, data = data_y)

# Extract posterior samples from each fitted model
posterior_x <- extract(fit_x)
posterior_y <- extract(fit_y)

# -------------------------------------
# Posterior Summary: Series with phi=0.4
# -------------------------------------
mean_mu_x <- mean(posterior_x$mu)
ci_mu_x <- quantile(posterior_x$mu, probs = c(0.025, 0.975))
print(mean_mu_x); print(ci_mu_x)

mean_phi_x <- mean(posterior_x$phi)
ci_phi_x <- quantile(posterior_x$phi, probs = c(0.025, 0.975))
print(mean_phi_x); print(ci_phi_x)

mean_sigma2_x <- mean(posterior_x$sigma2)
ci_sigma2_x <- quantile(posterior_x$sigma2, probs = c(0.025, 0.975))
print(mean_sigma2_x); print(ci_sigma2_x)

# -------------------------------------
# Posterior Summary: Series with phi=0.98
# -------------------------------------
mean_mu_y <- mean(posterior_y$mu)
ci_mu_y <- quantile(posterior_y$mu, probs = c(0.025, 0.975))
print(mean_mu_y); print(ci_mu_y)

mean_phi_y <- mean(posterior_y$phi)
ci_phi_y <- quantile(posterior_y$phi, probs = c(0.025, 0.975))
print(mean_phi_y); print(ci_phi_y)

mean_sigma2_y <- mean(posterior_y$sigma2)
ci_sigma2_y <- quantile(posterior_y$sigma2, probs = c(0.025, 0.975))
print(mean_sigma2_y); print(ci_sigma2_y)

# Optional full printout of Stan summary for deeper inspection
print(fit_x)
print(fit_y)


OBSERVATION:When the autoregressive coefficient ϕ=0.2 is employed, the credible intervals for the AR(1) 
model parameters are noticeably smaller, suggesting a higher level of accuracy in estimating the true
parameter values. This smaller range, which reflects a more stable and well-behaved stationary process,
indicates more confidence in the posterior estimations. On the other hand, the credible intervals are
significantly broader when ϕ=0.95 is used, indicating greater variability and uncertainty in the parameter 
estimations. This greater dispersion reflects the difficulties in estimating parameters in a near-unit-root process,
where the series shows persistent, almost non-stationary behaviour, making it more challenging to precisely identify 
the genuine parameter values.

# =======================================
# Plotting Joint Posteriors (mu vs phi)
# =======================================

# Joint posterior for mu vs phi (Series X, φ = 0.4)
plot(posterior_x$mu[1000:2000], posterior_x$phi[1000:2000],
     xlab = "mu values", ylab = "phi values", col = "blue",
     main = "Joint Posterior: mu vs phi (X, phi = 0.4)")
# Interpretation:
# - Tight clustering of samples indicates strong convergence
# - Parameters are well-estimated due to relatively stable AR(1) process

# Joint posterior for mu vs phi (Series Y, φ = 0.98)
plot(posterior_y$mu[1000:2000], posterior_y$phi[1000:2000],
     xlab = "mu values", ylab = "phi values", col = "darkred",
     main = "Joint Posterior: mu vs phi (Y, phi = 0.98)")
# Interpretation:
# - Wider spread of points shows poor convergence and more uncertainty
# - The near-unit root (phi close to 1) makes estimation more difficult
# - Reflects the persistence and non-stationary behavior of the series


OBSERVATION:We are able to identify a convergence for the initial sampler.
For the second sampler, as seen in the picture, it is more challenging to 
discern a convergence. According to the earlier logic, the second sampler's
values encompass a larger range than the first sampler's.











