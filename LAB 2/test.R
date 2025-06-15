##########################################################################################
##1ST PART #####################################
#Consider the logistic regression model: Pr(y = 1|x ,beta)= (exp (x^T beta)/ 1+exp(x^T beta)) where 
# y equals 1 if the person has a disease and 0 if not. x is a 6-dimensional vector containing five features and a column of 
# 1's to include an intercept in the model. The values of the variables Age, Duration_of_symptoms, and White_blood in x need 
# to be standardized to mean 0 and variance 1. For each of these variables xj, calculate the standardized value for each observation 
# i by using the formula (x ij-(xj bar)/s xj)  where (xj bar )and s xj are the sample mean and standard deviation of xj, respectively. 
# The goal is to approximate the posterior distribution of the parameter vector beta  with a multivariate normal distribution 
# (beta|y,x~N(beta tilda,j^-1y(beta tilda))) where beta tilda is the posterior mode and J( beta tilda) is the negative of the observed Hessian 
# evaluated at the posterior mode.
# Load required package
library(mvtnorm)

# Load the dataset
disease_data <- read.csv("C:/Users/Dell/OneDrive - Linköpings universitet/732A73 Bayesian Learning/Labs/Lab2/Disease.csv")

standardize <- function(x, mean_val = NULL, sd_val = NULL) {
  if (is.null(mean_val)) mean_val <- mean(x, na.rm = TRUE)
  if (is.null(sd_val)) sd_val <- sd(x, na.rm = TRUE)
  
  (x - mean_val) / sd_val
}

# Prepare the response variable (using 'class_of_diagnosis' as outcome)
y <- disease_data$class_of_diagnosis  # Assuming 1 = has disease, 0 = healthy
# Calculate standardization parameters (must do this FIRST)
age_mean <- mean(disease_data$age)
age_sd <- sd(disease_data$age)
symptoms_mean <- mean(disease_data$duration_of_symptoms)
symptoms_sd <- sd(disease_data$duration_of_symptoms)
whiteblood_mean <- mean(disease_data$white_blood)
whiteblood_sd <- sd(disease_data$white_blood)
# Prepare predictors with standardization where needed
X <- cbind(# C binder means column-wise into a matrix or data frame.
  Intercept = 1,
  Age = (disease_data$age - age_mean)/age_sd,
  Gender = disease_data$gender,
  Symptoms = (disease_data$duration_of_symptoms - symptoms_mean)/symptoms_sd,
  Breathing = disease_data$dyspnoea,
  WhiteCells = (disease_data$white_blood - whiteblood_mean)/whiteblood_sd
)

# Convert to X is stored as a numeric matrix
X <- as.matrix(X)

#(A)Calculate both beta tilda and J(beta_tilda) by using the optim function in R.Use the prior beta~ N(0, tau^2I), where tau=2.

# Define log-posterior function with N(0, tau²I) prior (tau=2)
log_posterior <- function(beta, X, y) {
  eta <- X %*% beta #here it computes linear predictor
  prob <- plogis(eta)  # Logistic function or equal to 1/(1+exp(eta)) which is same as logistic expression
  log_lik <- sum(y * log(prob) + (1-y) * log(1-prob)) # log-likelihood is been applies on the above logistic regresssion model
  log_prior <- sum(dnorm(beta, mean=0, sd=2, log=TRUE))  # Prior N(0,4)# here we calculate log-prior for the bayesian model,and we sum up all the log-prior for all coefficients
  log_lik + log_prior
}

# Find posterior mode (beta_tilde) and Hessian (J)
optim_result <- optim(#optim function here is general optimixation function
  par = rep(0, ncol(X)),#It is the starting point for optimization,,it createsa vector of zeros,in logistic regression beta=(beta1,beta2,beta3....beta k)so here we assume all coefficient are zero
  fn = log_posterior,# this is the function that needs to be maximized
  X = X, y = y,
  method = "BFGS",
  control = list(fnscale=-1),  # Maximize
  hessian = TRUE  # it calculates the Hessian matrix(ie the second derivative )at the solution
)# why second derivative , the first derivate tells the direction of the gradient steep increase 
#the second derivative or the hessian tells about the curvature of the slope(ie how fast the slope is changing)

# Extract results
beta_tilde <- optim_result$par  # Posterior mode
J_beta_tilde <- -optim_result$hessian  # Negative Hessian

# (B) Present the numerical values of beta tilda and J^-1y(beta tilda )for the Disease data.
# Name the coefficients
names(beta_tilde) <- colnames(X)
rownames(J_beta_tilde) <- colnames(X)
colnames(J_beta_tilde) <- colnames(X)

# Print results
cat("(B) Posterior Mode (beta_tilde):\n")
print(beta_tilde)

cat("\n(B) Inverse Negative Hessian (J^-1(beta_tilde)):\n")
print(solve(J_beta_tilde))  # Posterior covariance matrix

# (C) Compute an approximate 95% equal tail posterior probability interval for the regression coeffcient to the variable Age. 

# Get Age coefficient index
age_index <- which(names(beta_tilde) == "Age")#IT IS MENTIONS IN THE QUESTION that we find the postirio mode and returns the coefficient of the age

# Calculate posterior standard deviation for Age
age_sd <- sqrt(solve(J_beta_tilde)[age_index, age_index])

# 95% equal-tailed credible interval
age_ci <- beta_tilde[age_index] + c(-1.96, 1.96) * age_sd

cat("\n(C) 95% Posterior Interval for Age coefficient:\n")
cat(sprintf("[%.4f, %.4f]\n", age_ci[1], age_ci[2]))

# Interpretation
if (prod(age_ci) > 0) {  # Both limits have same sign
  cat("Age is statistically significant (95% CI doesn't contain 0)\n")
  if (age_ci[1] > 0) {
    cat("Older age increases disease probability\n")
  } else {
    cat("Older age decreases disease probability\n")
  }
} else {
  cat("Age is not statistically significant (95% CI contains 0)\n")
}
#(D) verify that estimation results reasonable by comparing the posterior means tomaximum likelihood estimates given by:glmModel

# Optional: Compare with frequentist approach
cat("\nComparison with standard glm() results:\n")
glm_fit <- glm(y ~ Age + Gender + Symptoms + Breathing + WhiteCells, 
               data = as.data.frame(X[,-1]),  # Remove intercept
               family = binomial)
print(summary(glm_fit))
###############################################################################################
#2ND PART
# 1. Define the simulation function (improved version)
simulate_predictive <- function(age, gender, symptoms, breathing, whiteblood, n_sim = 10000) {
  # Standardize the continuous variables using the same scaling as before
  age_std <- (age - age_mean) / age_sd
  symptoms_std <- (symptoms - symptoms_mean) / symptoms_sd
  whiteblood_std <- (whiteblood - whiteblood_mean) / whiteblood_sd
  
  # Create the feature vector for this patient
  x_patient <- c(1, age_std, gender, symptoms_std, breathing, whiteblood_std)
  
  # Simulate beta draws from posterior normal approximation
  beta_draws <- rmvnorm(n_sim, mean = beta_tilde, sigma = post_cov)
  
  # Calculate Pr(y=1|x) for each draw
  prob_draws <- plogis(beta_draws %*% x_patient)
  
  return(prob_draws)
}

# 2. Simulate for our specific patient:
# 38-year-old woman (gender=1), 10 days symptoms, no labored breathing (0), 11000 white blood
library(ggplot2)
# 1. Define core functions
sigmoid <- function(z) 1 / (1 + exp(-z))

predict_prob <- function(beta_mean, beta_cov, new_x, n_samples = 1000) {
  beta_samples <- rmvnorm(n_samples, beta_mean, beta_cov)
  apply(beta_samples, 1, function(beta) sigmoid(sum(new_x * beta)))
}

new_patient <- c(
  1,  # Intercept
  standardize(38, means$age, sds$age),  # Age
  standardize(10, means$duration, sds$duration),  # Symptoms
  standardize(11000, means$white_blood, sds$white_blood),  # WhiteBlood
  1,  # Gender (1 = female)
  0   # Dyspnoea (0 = no)
)

probs <- predict_prob(beta_tilde, post_cov, new_patient)

# 5. Visualization (clean and simple)
par(mfrow = c(1, 2))

# Density plot
plot(density(probs), main = "Disease Probability Distribution",
     xlab = "P(Disease = 1)", col = "blue", lwd = 2)
abline(v = mean(probs), col = "red", lty = 2)
legend("topright", legend = c("Density", "Mean"), 
       col = c("blue", "red"), lty = c(1, 2), cex = 0.5)

# Histogram
hist(probs, breaks = 30, col = "skyblue", border = "white",
     main = "Posterior Predictive Distribution",
     xlab = "Probability of Disease")

par(mfrow = c(1, 1))

# 6. Print results
cat("\nPrediction Summary for 38yo Female Patient:\n")
cat(sprintf("Mean probability: %.3f\n", mean(probs)))
cat(sprintf("95%% Credible Interval: [%.3f, %.3f]\n", 
            quantile(probs, 0.025), quantile(probs, 0.975)))

















