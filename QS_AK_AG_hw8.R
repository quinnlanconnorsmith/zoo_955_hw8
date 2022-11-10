#Amanda Kerkhove and Quinn Smith and Adrianna Gorsky 
#Zoo 955 HW8 MLE

# Q1: Simulate some data for a linear regression using the simple linear regression model: yi = beta_0 + beta_1*xi + error 
# Where error~N(0,sig).  

#100 points, roughly slope of 2, sigma of 4, intercept of 1

set.seed(123)
b0<-1
b1<-2
sig<-4
x<-1:100

y=b0+b1*x+rnorm(length(x), mean=0, sd=sig)
plot(y~x)

model<-lm(y~x)

#plot(model)

ci<-confint(model, x, level=.95)
ci
#2.5 %   97.5 %
#  (Intercept) -0.6084649 2.317234
#x            1.9848956 2.035193
#Model assumptions look alright, qq plot is a little hazy but if you squint hard enough it's ok

# Q2: Analyze the data generated in Q1 using the normal equation:
x1<-cbind(rep(1, length(x)), x)
b<-solve(t(x1)%*%x1) %*% (t(x1)%*%y)
b
#0.8543846
#x 2.0100443
#The estimates are pretty close to our values! Not as close as the first method though 

# Q3: Analyze the data generated in Q1 using a grid search to minimize the sum of squared errors (no need to iterate more than twice):

rss_1 <- function(y, y_pred){sum((y_pred-y)^2)}

# Set grid search
b0_grid <- seq(.75, 1.25, .01)
b1_grid <- seq(1.75, 2.25, .01)
rss_matrix <- matrix(data=NA, nrow=length(b0_grid), ncol=length(b1_grid), 
                     dimnames=list(b0_grid, b1_grid))
# Begin grid search: i <- 1; j <- 1
for(i in 1:length(b0_grid)){
  b0_curr <- b0_grid[i]
  for(j in 1:length(b1_grid)){
    b1_curr <- b1_grid[j]
    y_pred <- b0_curr + b1_curr*x
    rss <- rss_1(y, y_pred)
    rss_matrix[i,j] <- rss
  }
}

rss_2 <- which(rss_matrix==min(rss_matrix), arr.ind=T)
rss_coef <- c(b0_grid[rss_2[1]], b1_grid[rss_2[2]])
# Visualize grid search
image(x=b0_grid, y=b1_grid, z=rss_matrix, las=1, xlab="B0 est", ylab="B1 est")
points(x=rss_coef[1], y=rss_coef[2], pch=20, cex=1)

rss_coef

#B0 is 0.86, which is kinda close 
#B1 is 2.01, which is very close 

# Q4: Analyze the data generated in Q1 using a grid search to minimize the negative log likelihood (no need to iterate more than twice).  Note, there is a third parameter that you will need to estimate here: sig

nll_1 <- function(y, y_pred, sig){
  -sum(dnorm(x=y, mean=y_pred, sd=sig, log=T))
}

#Grid search
b0_grid <- seq(.75, 1.25, .01)
b1_grid <- seq(1.75, 2.25, .01)
sig_eval <- seq(2,5,0.1)
nll_array <- array(data=NA, dim=c(length(b0_grid), length(b1_grid), length(sig_eval)), 
                   dimnames=list(b0_grid, b1_grid, sig_eval))
#Grid search: i <- 1; j <- 1; k <- 1
for(i in 1:length(b0_grid)){
  b0_curr <- b0_grid[i]
  for(j in 1:length(b1_grid)){
    b1_curr <- b1_grid[j]
    for(k in 1:length(sig_eval)){
      sig_curr <- sig_eval[k]
      y_pred <- b0_curr + b1_curr*x
      nll <- nll_1(y, y_pred, sig)
      nll_array[i,j,k] <- nll
    }
  }
}
#Evaluate best parameters 
nll_2 <- which(nll_array==min(nll_array), arr.ind=T)
nll_coef <- c(b0_grid[nll_2[1]], 
                   b1_grid[nll_2[2]],
                   sig_eval[nll_2[2]])
nll_coef

# Q5: Analyze the data generated in Q1 using optim() to minimize the negative log likelihood.  

fun_1 <- function(par){
  b0 <- par[1]
  b1 <- par[2]
  sig <- par[3]
  y_pred <- b0 + b1*x
  nll <- nll_1(y, y_pred, sig)
}

#optim() stuff 
fit <- optim(par=c(1,2,4), fn=fun_1)
optim_coef <- fit$par

optim_coef

#This did converge, the estimates are not on the bounds that we gave the function but within bounds 
#If we blow the bounds out we can see that this is the global maximum

# Q6: Plot a likelihood profile for the slope parameter while estimating the conditional MLEs of the intercept and sig for each plotted value of the slope parameter (see p. 173 of Hilborn and Mangel).

# Nicer function for calculating NLL
nll_1_new <- function(b0, b1, sig, y){
  y_pred <- b0 + b1*x
  nll <- nll_1(y, y_pred, sig)
}

# Parameters 
b0_hold <- b0 * 1.05
sig_hold <- sig * 0.95

# Estimate NLL over slopes
b1s <- seq(0,2.5,0.01)
nlls <- sapply(1:length(b1s), function(x) nll_1_new(b0_hold, b1s[x], sig_hold, y))
plot(nlls~b1s, type="l", bty="n", las=1, xlab="B1 est", ylab="Neg log likelihood")

#Lowest negative log likelihood at our estimated slope of 2, increases on each side! 

# Q7: Plot the joint likelihood surface for the intercept and slope parameters.  Is there evidence of confounding between these two parameters (i.e., a ridge rather than a mountain top)?

# Parameters 
b0_grid <- seq(.75, 1.25, .01)
b1_grid <- seq(1.75, 2.25, .01)
sig_hold <- sig

# Setup grid search
nll_matrix <- matrix(data=NA, nrow=length(b0_grid), ncol=length(b1_grid), 
                     dimnames=list(b0_grid, b1_grid))
# Begin grid search: i <- 1; j <- 1
for(i in 1:length(b0_grid)){
  b0_curr <- b0_grid[i]
  for(j in 1:length(b1_grid)){
    b1_curr <- b1_grid[j]
    nll <- nll_1_new(b0_curr, b1_curr, sig_hold, y)
    nll_matrix[i,j] <- nll
  }
}
# Visualize grid search
#This is wild 
image(x=b0_grid, y=b1_grid, z=nll_matrix, las=1, xlab="B0 est", ylab="B1 est")
contour(x=b0_grid, y=b1_grid, z=nll_matrix, las=1, xlab="B0 est", ylab="B1 est")
persp(x=b0_grid, y=b1_grid, z=nll_matrix, theta=270, phi=20)

#There is confounding between the parameters, we can see the graph has a ridge and looks like a taco 

# Q8: How different are the estimated coefficients from Q1, Q2, and Q5 and how do they compare to the true values?

#These values are slightly different from one another, the analytical solution was the "best" probably because this was 
#a simple simulated dataset. The optimization approach would probably be better for complex data 