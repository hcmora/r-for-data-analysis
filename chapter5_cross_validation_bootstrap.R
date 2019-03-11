############################################
### Cross-Validation
############################################

library(ISLR)
set.seed(1)
?sample
# We create a vector that has n/2 numbers selected randomly
train=sample(392,196)
# We proceed to fit the data using the subset created before
lm.fit = lm(mpg~horsepower,data=Auto,subset=train)
# Now we calculate the MSE of the predicted data
attach(Auto)
mean((mpg-predict(lm.fit,Auto))[-train]^2)
# The MSE of the linear model is 26.14, now we do the same for polynomial models
lm.fit2 = lm(mpg~poly(horsepower,2),data=Auto,subset=train)
mean((mpg-predict(lm.fit2,Auto))[-train]^2)
# MSE for a the cuadratic model is 19.82
lm.fit3 = lm(mpg~poly(horsepower,3),data=Auto,subset=train)
mean((mpg-predict(lm.fit3,Auto))[-train]^2)
# MSE for the cubic model is 19.78

## Now we'll check the same models with another random training data set
set.seed(2)
train=sample(392,196)
lm.fit = lm(mpg~horsepower,data=Auto,subset=train)
mean((mpg-predict(lm.fit,Auto))[-train]^2)
# Linear MSE = 23.30
lm.fit2 = lm(mpg~poly(horsepower,2),data=Auto,subset=train)
mean((mpg-predict(lm.fit2,Auto))[-train]^2)
# Cuadratic MSE = 18.90
lm.fit3 = lm(mpg~poly(horsepower,3),data=Auto,subset=train)
mean((mpg-predict(lm.fit3,Auto))[-train]^2)
# Cubic MSE = 19.25

# We concluded that the linear model is definitely not the best model for this data
# and that there's no need to use cubic instead of cuadratic, at least for these two
# tests

############################################
### Leave-One-Out Cross-Validation
############################################

# We'll use the cv.glm() function in combination with the glm function.
# The cv.glm() function is part of the boot library
library(boot)
glm.fit = glm(mpg~horsepower,data=Auto)
coef(glm.fit)
# lm.fit = lm(mpg~horsepower,data=Auto)
# coef(lm.fit)

?cv.glm
cv.err = cv.glm(Auto,glm.fit)
cv.err$delta

# We can check the least squares linear or polynomial regression formula to calculate this
loocv = function(fit){
  h = lm.influence(fit)$h
  mean((residuals(fit)/(1-h))^2)
}
loocv(glm.fit)


# We'll check out the error for polynomials up to 5th degree
cv.error = rep(0,5)
degree = 1:5
for (i in degree){
  glm.fit=glm(mpg~poly(horsepower,i),data=Auto)
  # cv.error[i] = cv.glm(Auto,glm.fit)$delta[1]
  cv.error[i] = loocv(glm.fit)
}
cv.error
plot(degree,cv.error,type = 'b')
# We check that the linear model is the worst fit. After the second degree, there's not much
# improvement in the model

############################################
### k-Fold Cross-Validation
############################################

# We can use the same cv.glm() function to implement the k-fold Cross-Validation

set.seed(17)
cv.error.10 = rep(0,5)
for (i in degree){
  glm.fit = glm(mpg~poly(horsepower,i),data=Auto)
  cv.error.10[i] = cv.glm(Auto,glm.fit,K=10)$delta[1]
}
cv.error.10

glm.fit = glm(mpg~poly(horsepower,2),data=Auto)
cv.error = cv.glm(Auto,glm.fit,K=10)
cv.error$delta
lines(degree,cv.error.10,type='b',col='red')

# We can see now that the value of the components of delta are different.
# The first value corresponds to the raw CV value, that is CV_k = 1/k*sum(MSE_i,i=1..k)
# The second value is a bias-corrected value, the adjustment is designed to compensate the 
# bias introduced for not using the leave-one-out cross-validation

############################################
### The Bootstrap
############################################

### Estimating the accuracy of a Statistic of Interest

# First, we create a function that estimates the value of alpha (reference the 5.7 Formula of the ISLR book)
# Alpha in this case is the fraction that minimize the variance between X and Y
# We'll use the Portfolio database, which has the columns X and Y
?Portfolio

alpha.fn = function(data,index) {
  X = data$X[index]
  Y = data$Y[index]
  return ((var(Y) - cov(X,Y))/(var(X)+var(Y)-2*cov(X,Y)))
}

# We could use the function sample() to pick the index randomly with repetition, and iterate this multiple
# times, saving the alpha estimate, to calculate the SD
?sample
set.seed(1)
alpha.fn(Portfolio,sample(100,100,replace=TRUE))

# Instead of iterating, we can use the boot function of the boot library to get a complete report
?boot
boot.out = boot(Portfolio,alpha.fn,R=1000)
boot.out
plot(boot.out)
### Estimating the accuracy of a Linear Regression Model

# Now we'll use the method to assess the variability of the coefficients beta_0 and beta_1 in a linear
# regression model applied to the Auto database

boot.fn = function(data,index){
  return(coef(lm(mpg~horsepower,data=data,subset=index)))
}
boot.fn(Auto,1:392)

# We can now create bootstrap estimates using the sample function
set.seed(1)
boot.fn(Auto,sample(392,392,replace = TRUE))
boot.fn(Auto,sample(392,392,replace = TRUE))

# Now we use bootstrap to estimate the SE of the coefficients
boot(Auto,boot.fn,R=1000)

# We compare it to the SE calculated using the formulas
summary(lm(mpg~horsepower,data=Auto))$coef

# We can see that the SE are different. This does not mean that the bootstrap method is flawed, on the
# contrary, it means that the assumptions made for the linear regression calculations, for example, 
# the estimation of the sigma^2 used in the linear model assumes that the linear model is correct. 
# We know that this is not true for the mpg~horsepower relationship, therefore, the SE given by
# the bootsrap method is most likely to be more accurate than the one given by the linear function

# Now we check the same thing with the quadratic expression
boot.fn = function(data,index) {
  coefficients(lm(mpg~horsepower+I(horsepower^2),data=data,subset=index))
}
set.seed(1)
boot(Auto,boot.fn,R=1000)
plot(boot(Auto,boot.fn,R=1000))
summary(lm(mpg~horsepower+I(horsepower^2),data=Auto))$coef

# We see that there's still difference between the coefficients, but they're more alike