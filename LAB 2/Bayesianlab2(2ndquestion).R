# Load required library
library(mvtnorm)

# ---------------------------------
# Helper: Standardize function
#What we are doing here is transforming x so that it has mean 0 and sd of 1.
#Here na.rm ignores missing values,when doing the calculation when kept to TRUE.
#using the equation given in the question 
# ---------------------------------
standardize <- function(x, mean_val = NULL, sd_val = NULL) {
  if (is.null(mean_val)) mean_val <- mean(x, na.rm = TRUE)
  if (is.null(sd_val)) sd_val <- sd(x, na.rm = TRUE)
  
  (x - mean_val) / sd_val
}

# ---------------------------------
# 1. Prepare Data
#There is a function called prepare_data,, we calculate mean of age ,duration_of_symtom,white_blood and we ignore the NA values.
#We repeat the same for standard deviation as well.after this we standardize 3 columns(age,duration,white_bood).
#Next we create a design matrix so turning all the selected variables into a clean table 
#for the machine learning model X as the variable and Y as the target.

# ---------------------------------
prepare_data <- function(data) {
  means <- list(
    age = mean(data$age, na.rm = TRUE),
    duration = mean(data$duration_of_symptoms, na.rm = TRUE),
    white_blood = mean(data$white_blood, na.rm = TRUE)
  )
  
  sds <- list(
    age = sd(data$age, na.rm = TRUE),
    duration = sd(data$duration_of_symptoms, na.rm = TRUE),
    white_blood = sd(data$white_blood, na.rm = TRUE)
  )
  
  # Standardize
  data$age_std <- standardize(data$age)
  data$duration_std <- standardize(data$duration_of_symptoms)
  data$white_blood_std <- standardize(data$white_blood)
  
  # Design matrix (no intercept, because model.matrix automatically adds it)
  X <- model.matrix(~ age_std + duration_std + white_blood_std + gender + dyspnoea, data)
  y <- as.numeric(data$class_of_diagnosis)
  
  list(X = X, y = y, data = data, means = means, sds = sds)
}

# ---------------------------------
# 2. Bayesian Logistic Regression
#We have 2 parts here log_posterior and fit_bayesian_logistic function.
#In log_posterior the inputs are beta,y output target, X input matrix,mu mean of prior beta, Sigma covariance matrixof prior beta.
#we next calculate linear predictor, calculation of log-likelihoodfor logistic regression,it measures how well the beta value predicts the target label.
#next we use calculate log-prior probability using multivariate normal density function ,we combine prior belifs with the likelihood data
#In fit_bayesian_logistic function inputs are the X datamatrix, target y, and the tau variance of the prior belifabout beta.We set the initial beta values and mean  to 0 . 
#Next is to find the best beta values by maximizing log_posterior,here hessian is set to TRUE means it computes the second derivative matrix.
#What we return as a list is best estimated beta values,posterior covariance matrix for beta and the optim output as fit.
#In overall  It finds the best beta coefficients using Bayesian logistic regression by maximizing the posterior probability.
# ---------------------------------

# Log-posterior
log_posterior <- function(beta, y, X, mu, Sigma) {
  lin_pred <- X %*% beta
  log_lik <- sum(lin_pred * y - log1p(exp(lin_pred)))
  log_prior <- dmvnorm(beta, mu, Sigma, log = TRUE)
  log_lik + log_prior
}

# Fit Bayesian model
fit_bayesian_logistic <- function(X, y, tau = 2) {
  p <- ncol(X)
  init_beta <- rep(0, p)
  mu <- rep(0, p)
  Sigma <- diag(tau^2, p)
  
  fit <- optim(
    init_beta,
    log_posterior,
    y = y,
    X = X,
    mu = mu,
    Sigma = Sigma,
    method = "BFGS",
    control = list(fnscale = -1),
    hessian = TRUE
  )
  
  list(
    beta_hat = fit$par,
    post_cov = solve(-fit$hessian),
    fit = fit
  )
}

# ---------------------------------
# 3. Prediction
#We have a sigmoid function here also called as logistic function.
#The predict_prob function where we pass new inputs , 
#we draw 1000 random samples from the multivariate normal distribution,
#like we take different beta values around the best beta, 
#so overall here we have a  vector of 1000 different predicted probabilities.
#Each one based on a slightly different beta


# ---------------------------------

# Sigmoid
sigmoid <- function(z) {
  1 / (1 + exp(-z))
}

# Predict
predict_prob <- function(model, new_x, n_samples = 1000) {
  beta_samples <- rmvnorm(n_samples, model$beta_hat, model$post_cov)
  apply(beta_samples, 1, function(beta) {
    sigmoid(sum(new_x * beta))
  })
}

# ---------------------------------
# 4. Main Workflow
# ---------------------------------

# Load your dataset
data <- load_and_validate_data("Disease.csv")  # Assuming this function exists

# Prepare data
prepared <- prepare_data(data)
X <- prepared$X
y <- prepared$y
means <- prepared$means
sds <- prepared$sds

# Fit Bayesian model
set.seed(12345)
bayes_model <- fit_bayesian_logistic(X, y)

# ---------------------------------
# NEW: Fit GLM for MLE estimation
# ---------------------------------

glm_model <- glm(class_of_diagnosis ~ age_std + duration_std + white_blood_std + gender + dyspnoea,
  data = prepared$data,
  family = binomial(link = "logit")
)

# ---------------------------------
# 5. Show Comparison
# ---------------------------------

cat("\nPosterior Coefficients (Bayesian):\n")
print(setNames(bayes_model$beta_hat, colnames(X)))

cat("\nGLM Coefficients (Maximum Likelihood Estimates):\n")
print(coef(glm_model))

# Combine into a comparison table
comparison <- data.frame(
  Variable = names(coef(glm_model)),
  Bayesian_Posterior_Mean = round(bayes_model$beta_hat, 4),
  MLE_GLM = round(coef(glm_model), 4)
)

cat("\nComparison Table:\n")
print(comparison)



# ---------------------------------
# UPDATED: Print Beta Bar and J(Beta)
# ---------------------------------
# ---------------------------------
# 6. Compute and print Beta Bar (tilde_beta) and J(beta_tilde)
# ---------------------------------

# Posterior mean (beta_tilde = beta_hat)
beta_tilde <- bayes_model$beta_hat
cat("\nBeta Bar (Posterior Mean, beta_tilde):\n")
print(beta_tilde)

# Prior parameters (tau = 2 as in fit_bayesian_logistic)
tau <- 2
mu_prior <- rep(0, length(beta_tilde))
Sigma_prior <- diag(tau^2, length(beta_tilde))

# Compute J(beta_tilde) = -log_posterior(beta_tilde)
log_post_at_tilde <- log_posterior(
  beta_tilde,
  y = y,
  X = X,
  mu = mu_prior,
  Sigma = Sigma_prior
)
J_beta_tilde <- -log_post_at_tilde

cat("\nJ(beta_tilde) = -log_posterior(beta_tilde):", J_beta_tilde, "\n")

# ---------------------------------
# 6. Predict for a new patient
# ---------------------------------

new_patient <- c(
  1,  # intercept
  standardize(38, means$age, sds$age),
  standardize(10, means$duration, sds$duration),
  standardize(11000, means$white_blood, sds$white_blood),
  1,  # female
  0   # no dyspnoea
)

probs <- predict_prob(bayes_model, new_patient)

# Visualization
par(mfrow = c(1, 2))
plot(density(probs), main = "Disease Probability", xlab = "P(y=1)")
abline(v = mean(probs), col = "blue", lty = 2)
barplot(table(rbinom(1000, 1, probs)) / 1000, names.arg = c("No Disease", "Disease"),
        main = "Predicted Outcomes")
par(mfrow = c(1, 1))

# Prediction summary
cat("\nPrediction Summary:\n")
cat("Mean probability:", mean(probs), "\n")
cat("95% CI:", quantile(probs, c(0.025, 0.975)), "\n")
