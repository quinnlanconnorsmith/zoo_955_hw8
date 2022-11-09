
# Maximum likelihood estimation
# ZOO 955 - Week 8
# Answer Key


# Background
# We're going to implement three ways of fitting a linear regression model: 
# (1) the analytical solution using the normal equation
# (2) a numerical optimization approach based on minimizing the sum of squared errors
# (3) a numerical optimization approach based on maximizing the likelihood.

# Q1: Simulate some data for a linear regression using 
# the simple linear regression model: yi = beta_0 + beta_1*xi + error 

# True parameters
b0 <- 0.1
b1 <- 0.8
sigma <- 2

# Simulate data from true parameters with error
x <- 1:25
y_obs <- b0 + b1*x + rnorm(length(x), mean=0, sd=sigma)
plot(y_obs ~ x)

# Fit a linear regression model using lm() 
lmfit <- lm(y_obs~x); summary(lmfit)
lmcoefs <- coef(lmfit)


# Q2: Analyze the data generated in Q1 using the normal equation (ne):
x1 <- cbind(rep(1, length(x)), x) # add column of 1s to calculate intercept
necoefs <- solve(t(x1)%*%x1)%*%(t(x1)%*%y_obs)

# Bonus Q1: Can you find an analytical solution 
# for the se and 95% CI for the model coefficients? 

# Need to do some googling for this one - see, for example:
# https://stats.stackexchange.com/questions/44838/how-are-the-standard-errors-of-coefficients-calculated-in-a-regression/44841#44841

# The standard errors are 
#from lm()
summary(lmfit)$coefficients[2, 2]

#from analytical solution
num <- length(x) * anova(lmfit)[[3]][2]
denom <- length(x) * sum(x^2) - sum(x)^2
sqrt(num / denom)



# Q3: Analyze the data generated in Q1 using a grid search 
# to minimize the sum of squared errors (no need to iterate more than twice):

# Function to calculate sum of squared errors (RSS)
calc_rss <- function(y_obs, y_pred){sum((y_pred-y_obs)^2)}

# Set grid search
b0_eval <- seq(0,0.5,0.01)
b1_eval <- seq(0.7,0.9,0.01)
rss_matrix <- matrix(data=NA, nrow=length(b0_eval), ncol=length(b1_eval), 
                     dimnames=list(b0_eval, b1_eval))
# Begin grid search: i <- 1; j <- 1
for(i in 1:length(b0_eval)){
  b0_curr <- b0_eval[i]
  for(j in 1:length(b1_eval)){
    b1_curr <- b1_eval[j]
    y_pred <- b0_curr + b1_curr*x
    rss <- calc_rss(y_obs, y_pred)
    rss_matrix[i,j] <- rss
  }
}
# Best params
min_rss_index <- which(rss_matrix==min(rss_matrix), arr.ind=T)
min_rss_coefs <- c(b0_eval[min_rss_index[1]], b1_eval[min_rss_index[2]])
# Visualize grid search
image(x=b0_eval, y=b1_eval, z=rss_matrix, las=1, xlab="B0", ylab="B1")
points(x=min_rss_coefs[1], y=min_rss_coefs[2], pch=16, cex=1)


# Q4: Analyze the data generated in Q1 using a grid search 
# to minimize the negative log likelihood (no need to iterate more than twice).
# Note, there is a third parameter that you will need to estimate here: sigma

# Function to calculate the negative log-likelihood
calc_nll <- function(y_obs, y_pred, sigma){
  -sum(dnorm(x=y_obs, mean=y_pred, sd=sigma, log=T))
}

# Set grid search
b0_eval <- seq(0,0.5,0.01)
b1_eval <- seq(0.7,0.9,0.01)
sigma_eval <- seq(1,3,0.1)
nll_array <- array(data=NA, dim=c(length(b0_eval), length(b1_eval), length(sigma_eval)), 
                   dimnames=list(b0_eval, b1_eval, sigma_eval))
# Begin grid search: i <- 1; j <- 1; k <- 1
for(i in 1:length(b0_eval)){
  b0_curr <- b0_eval[i]
  for(j in 1:length(b1_eval)){
    b1_curr <- b1_eval[j]
    for(k in 1:length(sigma_eval)){
      sigma_curr <- sigma_eval[k]
      y_pred <- b0_curr + b1_curr*x
      nll <- calc_nll(y_obs, y_pred, sigma)
      nll_array[i,j,k] <- nll
    }
  }
}
# Best params
min_nll_index <- which(nll_array==min(nll_array), arr.ind=T)
min_nll_coefs <- c(b0_eval[min_nll_index[1]], 
                   b1_eval[min_nll_index[2]],
                   sigma_eval[min_nll_index[2]])


# Q5: Analyze the data generated in Q1 using optim() to minimize the negative log likelihood.  

# Objective function
obj_func <- function(par){
  b0 <- par[1]
  b1 <- par[2]
  sigma <- par[3]
  y_pred <- b0 + b1*x
  nll <- calc_nll(y_obs, y_pred, sigma)
}

# Estimate parameters using optim()
optfit <- optim(par=c(0,1,2), fn=obj_func)
optcoefs <- optfit$par


# Q6: Plot a likelihood profile for the slope parameter 
# holding the intercept and sigma constant at their MLEs.

# Nicer function for calculating NLL
calc_nll_new <- function(b0, b1, sigma, y_obs){
  y_pred <- b0 + b1*x
  nll <- calc_nll(y_obs, y_pred, sigma)
}

# Params
b0_hold <- b0 * 1.05
sigma_hold <- sigma * 0.95

# Estimate NLL for varying slopes
b1s <- seq(0,2,0.01)
nlls <- sapply(1:length(b1s), function(x) calc_nll_new(b0_hold, b1s[x], sigma_hold, y_obs))
plot(nlls~b1s, type="l", bty="n", las=1, xlab="B1", ylab="NLL")


# Q7: Plot the joint likelihood surface for the intercept and slope parameters
# holding sigma constant at its MLE.  Is there evidence of confounding between
# these two parameters (i.e., a ridge rather than a mountain top)?

# Params
b1_eval <- seq(0,3,0.1)
b0_eval <- seq(-1,1,0.1)
sigma_hold <- sigma

# Setup grid search
nll_matrix <- matrix(data=NA, nrow=length(b0_eval), ncol=length(b1_eval), 
                     dimnames=list(b0_eval, b1_eval))
# Begin grid search: i <- 1; j <- 1
for(i in 1:length(b0_eval)){
  b0_curr <- b0_eval[i]
  for(j in 1:length(b1_eval)){
    b1_curr <- b1_eval[j]
    nll <- calc_nll_new(b0_curr, b1_curr, sigma_hold, y_obs)
    nll_matrix[i,j] <- nll
  }
}
# Visualize grid search
image(x=b0_eval, y=b1_eval, z=nll_matrix, las=1, xlab="B0", ylab="B1")
contour(x=b0_eval, y=b1_eval, z=nll_matrix, las=1, xlab="B0", ylab="B1")
persp(x=b0_eval, y=b1_eval, z=nll_matrix, theta=80, phi=30)


# Q8: How different are the estimated coefficients from Q1, Q2, and Q5 and 
# how do they compare to the true values?



# Bonus Q2: Calculate the standard errors of the intercept, slope, and sigma using the Hessian matrix.
# Standard errors are the square roots of the diagonal of the inverse Hessian matrix.  
# How do these standard errors compare to those from lm() in Q1?

optfit_hess <- optim(par=c(0,1,2), fn=obj_func, hessian=T)
SE_optim <- sqrt(diag(solve(optfit_hess$hessian)))

# Bonus Q3: How does the computational speed compare between using lm(), the normal equation, and optim() to estimate the coefficients?
# Note: no solution given here, but it should be apparent from the instructions below how to do this
# Record your system start time
start_time <- proc.time()

#your code goes here

# Subtract start time from current system time
proc.time() - start_time
