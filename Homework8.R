#homework 8
#Simulate some data for a linear regression using the simple linear regression 
#model: yi = beta_0 + beta_1*xi + error 
set.seed(123)
b0<-1
b1<-2
sigma<-4
x<-1:100

y_obs=b0+b1*x+rnorm(length(x), mean=0, sd=sigma)
plot(y~x)

model<-lm(y~x)

#plot(model)

ci<-confint(model, x, level=.95)


# Q2: Analyze the data generated in Q1 using the normal equation:
x1<-cbind(rep(1, length(x)), x)
b<-solve(t(x1)%*%x1) %*% (t(x1)%*%y)

# Q3: Analyze the data generated in Q1 using a grid search to minimize the sum of squared errors (no need to iterate more than twice):

# Function to calculate sum of squared errors (RSS)
calc_rss <- function(y_obs, y_pred){sum((y_pred-y_obs)^2)}

# Set grid search
b0_eval <- seq(-.2, .2, .001)
b1_eval <- seq(1.9, 2.1, .001)
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


# Q4: Analyze the data generated in Q1 using a grid search to minimize the negative log likelihood (no need to iterate more than twice).  Note, there is a third parameter that you will need to estimate here: sigma

calc_nll <- function(y_obs, y_pred, sigma){
  -sum(dnorm(x=y_obs, mean=y_pred, sd=sigma, log=T))
}

# Set grid search
b0_eval <- seq(0, 100, 1)
b1_eval <- seq(0, 100, 1)
sigma_eval <- seq(0,100,1)
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

# Q5: Analyze the data generated in Q1 using optim() to minimize the negative log likelihood.  Note, there is a third parameter that you will need to estimate here: sigma

# Objective function
obj_func <- function(par){
  b0 <- par[1]
  b1 <- par[2]
  sigma <- par[3]
  y_pred <- b0 + b1*x
  nll <- calc_nll(y_obs, y_pred, sigma)
}

# Estimate parameters using optim()
optfit <- optim(par=c(1,2,4), fn=obj_func)
optcoefs <- optfit$par

# Q6: Plot a likelihood profile for the slope parameter while estimating the conditional MLEs of the intercept and sigma for each plotted value of the slope parameter (see p. 173 of Hilborn and Mangel).
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

# Q7: Plot the joint likelihood surface for the intercept and slope parameters.  Is there evidence of confounding between these two parameters (i.e., a ridge rather than a mountain top)?

# Params
b1_eval <- seq(1,3,0.1)
b0_eval <- seq(0,2,0.1)
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

# Q8: How different are the estimated coefficients from Q1, Q2, and Q5 and how do they compare to the true values?

