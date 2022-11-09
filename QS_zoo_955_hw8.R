#Amanda Kerkhove and Quinn Smith and Adrianna Gorsky 
#Zoo 955 HW8 MLE

# Q1: Simulate some data for a linear regression using the simple linear regression model: yi = beta_0 + beta_1*xi + error 
# Where error~N(0,sigma).  

set.seed(123)
#50 points, roughly slope of 2, sigma of 4, intercept of 0
b0 <- 1
b1 <- 2
sig <- 4
x <- 1:100

y <- b0 + b1*x + rnorm(length(x), mean=0, sd=sig)
plot(y ~ x)

lm <-lm(y~x)

plot(lm)

summary(lm)

#Model assumptions look alright, qq plot is a little hazy but if you squint hard enough it's ok

lmcoef <-coefficients(lm)
lmcoef
#Pretty close to true values! 

confint(lm,x,level=0.95)
#(Intercept) -0.4646118 2.871600
#x            1.9644238 2.021779

# Q2: Analyze the data generated in Q1 using the normal equation:

x1 <- cbind(rep(1, length(x)), x)
b <- solve(t(x1)%*%x1)%*%(t(x1)%*%y)
b
#The estimates are pretty close to our values! Not as close as the first method though 

# Q3: Analyze the data generated in Q1 using a grid search to minimize the sum of squared errors (no need to iterate more than twice):

calc_rss <- function(y, y_pred){sum((y_pred-y)^2)}

b0_eval <- seq(7,9,0.01)
b1_eval <- seq(0,0.5,0.01)
rss_matrix <- matrix(data=NA, nrow=length(b0_eval), ncol=length(b1_eval), 
                     dimnames=list(b0_eval, b1_eval))

for(i in 1:length(b0_eval)){
  b0_curr <- b0_eval[i]
  for(j in 1:length(b1_eval)){
    b1_curr <- b1_eval[j]
    y_pred <- b0_curr + b1_curr*x
    rss <- calc_rss(y, y_pred)
    rss_matrix[i,j] <- rss
  }
}
min_rss_index <- which(rss_matrix==min(rss_matrix), arr.ind=T)
min_rss_coefs <- c(b0_eval[min_rss_index[1]], b1_eval[min_rss_index[2]])
# Visualize grid search
image(x=b0_eval, y=b1_eval, z=rss_matrix, las=1, xlab="B0", ylab="B1")
points(x=min_rss_coefs[1], y=min_rss_coefs[2], pch=16, cex=1)

#Something is mega wonky here 
#B0 is apparently 7.86
#B1 is apparently 0.05

# Q4: Analyze the data generated in Q1 using a grid search to minimize the negative log likelihood (no need to iterate more than twice).  Note, there is a third parameter that you will need to estimate here: sigma

# Function to calculate the negative log-likelihood
calc_nll <- function(y, y_pred, sig){
  -sum(dnorm(x=y, mean=y_pred, sd=sig, log=T))
}

# Set grid search
b0_eval <- seq(7,9,0.01)
b1_eval <- seq(0,0.5,0.01)
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
      nll <- calc_nll(y, y_pred, sig)
      nll_array[i,j,k] <- nll
    }
  }
}
# Best params
min_nll_index <- which(nll_array==min(nll_array), arr.ind=T)
min_nll_coefs <- c(b0_eval[min_nll_index[1]], 
                   b1_eval[min_nll_index[2]],
                   sigma_eval[min_nll_index[2]])
