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
b0_eval <- seq(.75, 1.25, .01)
b1_eval <- seq(1.75, 2.25, .01)
sigma_eval <- seq(2,5,0.1)
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



