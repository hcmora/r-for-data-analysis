#####################################
### Principal Components Regression
#####################################

# PCR can be performed with the pcr() function of the pls library

library(ISLR)
Hitters = na.omit(Hitters)

# install.packages('pls')
library(pls)
set.seed(2)
pcr.fit = pcr(Salary~.,data=Hitters,scale=TRUE,validation="CV")

# scale = TRUE standardize the data and validation="CV" performs cross-validation

# The summary of the pcr fit shows us the root mean squared error (to obtain MSE we have to square it)
# and how much of the variance is explained by each direction

summary(pcr.fit)
validationplot(pcr.fit,val.type="MSEP")

# From the plot, we see that the MSE can be lowered down using 1 component and 16.
# If we use M = 16, we would almost be doing a least square regression (which occurs when M = 19)
# and the improvement isn't significant! So we'll take the chance with the small number of components

# Now we'll evaluate how it performs using the training data
x = model.matrix(Salary~.,Hitters)[,-1]
y = Hitters$Salary

set.seed(1)
train = sample(1:nrow(x),nrow(x)/2,replace=FALSE)
test = (-train)
y.test = y[test]

set.seed(1)
pcr.fit = pcr(Salary~.,data=Hitters,subset=train,scale=TRUE,validation="CV")
validationplot(pcr.fit,val.type = "MSEP")

# The lowest cross validation occurs when M = 7
pcr.pred = predict(pcr.fit,x[test,],ncomp=7)
mean((pcr.pred-y.test)^2)

# We see that the PCR obtained has a MSE similar to the ones obtained with ridge regression and the lasso!
# But, the problem with it is that it's hard to interpret!

# We now fit the PCR with M=7 to the full database
pcr.fit = pcr(y~x,scale=TRUE,ncomp=7)
summary(pcr.fit)


#####################################
### Parcial Least Squares
#####################################

set.seed(1)
pls.fit = plsr(Salary~.,data=Hitters,subset=train,scale=TRUE,validation="CV")
summary(pls.fit)

validationplot(pls.fit,val.type = "MSEP")

# Now the lowest cross-validation error occurs when M = 2 partial least squares directions are used.
# Now we evaluate the test set

pls.pred = predict(pls.fit,x[test,],ncomp=2)
mean((pls.pred-y.test)^2)

# This test MSE is a little higher than the others.
# Now we fit the full data using M = 2

pls.fit = plsr(Salary~.,data=Hitters,scale=TRUE,ncomp=2)
summary(pls.fit)

# Notice that the variance of Salary explained with only two variables in PLS is similar to the one 
# with seven variables in PCR