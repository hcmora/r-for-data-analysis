#####################################
### Ridge Regression
#####################################

# For ridge regression and lasso, the glmnet package is used.
# The main function that'll be used to compute the regressions 
# is the glmnet() function, which has a different syntax ussage.
# For it to work, we must pass an x matrix as well as a y vector
# In this function, the y~x syntax is not used

library(ISLR)
Hitters = na.omit(Hitters)

# The [,-1] used when creating the matrix using model.matrix is used
# omit the intercept column that is automatically added by the function
# The model.matrix also automatically transform qualitative variables 
# into dummy variables, which is really useful

x = model.matrix(Salary~.,Hitters)[,-1]
y = Hitters$Salary

# The glmnet() function has an alpha argument that defines the type of
# model will be fit. 0 is for ridge regression and 1 is for lasso

# install.packages('glmnet')
library(glmnet)
# We'll model the ridge regression using lambda = 10^10 to 0.01
grid = 10^seq(from=10,to=-2,length=100)
ridge.mod = glmnet(x,y,alpha=0,lambda=grid)
plot(ridge.mod,xvar="lambda",label=TRUE)

# To access the coefficients for each lambda we can use the coef() function
dim(coef(ridge.mod))
# Each row is one of the variables and each column is a value of lambda

# We can check the coefficients for different lambdas and their l2 values.
# We expect that the lower the lambda value, the larger l2 will be

ridge.mod$lambda[50]
coef(ridge.mod)[,50]
sqrt(sum(coef(ridge.mod)[-1,50]^2))

ridge.mod$lambda[60]
coef(ridge.mod)[,60]
sqrt(sum(coef(ridge.mod)[-1,60]^2))

# We can see that for a much lower lambda (705 instead of 11498), the l2 value
# increases from 6.36 to 57.1

# We can also obtain the coefficients for a specific lambda using the predict function
predict(ridge.mod,s=50,type="coefficients")[1:20,]

# Now we'll split the data using another process. Instead of creating a True/False
# vector, we'll simply choose random rows of the x matrix using sample
set.seed(1)
train = sample(1:nrow(x),nrow(x)/2,replace = FALSE)
test = (-train)
y.test = y[test]

# Now we'll fit the ridge regression on the training set and evaluate its MSe using lambda = 4 with
# the predict function

ridge.mod = glmnet(x[train,],y[train],alpha=0,lambda=grid,thresh=1e-12)
ridge.pred = predict(ridge.mod,s=4,newx=x[test,])
mean((ridge.pred-y.test)^2)

# To compare it to how it would be to fit a model with just the intercept coefficient, we can do it 
# with two ways

mean((mean(y[train])-y.test)^2)
ridge.pred = predict(ridge.mod,s=1e10,newx=x[test,])
mean((ridge.pred-y.test)^2)

# So the model with lambda = 4 has better fit than one with just the coefficient (oh well...)
# Now we'll see if it improves against the case where lambda = 0 (least squares regression)

ridge.pred = predict(ridge.mod,s=0,newx=x[test,])
mean((ridge.pred-y.test)^2)

# Now, instead of choosing lambda = 4, we can use cross validation to choose the best lambda
# available. For this, we can use the function cv.glmnet()
set.seed(1)
cv.out = cv.glmnet(x[train,],y[train],alpha=0)
plot(cv.out)
bestlam = cv.out$lambda.min
bestlam

ridge.pred = predict(ridge.mod,s=bestlam,newx=x[test,])
mean((ridge.pred - y.test)^2)

# So, the MSE was reduced using this process! Now we use this lambda in our full dataset to check
# the coefficients

out = glmnet(x,y,alpha=0)
predict(out,type="coefficients",s=bestlam)[1:20,]

# It's important to note that none of the coefficients are zero, as ridge regression will never obtain this

#####################################
### The Lasso
#####################################

lasso.mod = glmnet(x[train,],y[train],alpha=1,lambda=grid)
plot(lasso.mod,xvar="lambda",label=TRUE)
# From the plot we can see that depending from the lambda parameter, some coefficients will be zero

# We can also check the deviance explained with the values of the coefficients
plot(lasso.mod,xvar="dev",label=TRUE)

set.seed(1)
cv.out = cv.glmnet(x[train,],y[train],alpha=1)
plot(cv.out)
bestlam = cv.out$lambda.min
lasso.pred = predict(lasso.mod,s=bestlam,newx=x[test,])
mean((lasso.pred-y.test)^2)

# The MSE obtained is lower than the least squares MSE and the intercept MSE.
# It's similar to the MSE obtained using ridge regression
# BUT, lasso produces coefficients with value 0, making the model more interpretable

out = glmnet(x,y,alpha=1,lambda=grid)
lasso.coef = predict(out,type="coefficients",s=bestlam)[1:20,]
lasso.coef

## Another way of doing this process is as it follows

lasso.tr = glmnet(x[train,],y[train])
pred = predict(lasso.tr,x[-train,])
rmse = sqrt(apply((y[-train]-pred)^2,2,mean))
plot(log(lasso.tr$lambda),rmse,type="b",xlab="Log(lambda)")
lam.best = lasso.tr$lambda[order(rmse)[1]]
lam.best
coef(lasso.tr,s=lam.best)
