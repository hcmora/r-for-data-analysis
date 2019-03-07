# Useful commands that can be used to analyze data
# To import libraries
library(MASS)

# If a library is not found, it can be installed using install.packages() method. For example: install.packages("ISRL")

# The command fix allows us to look the working database, in this case, it's the Boston database from the MASS library
fix(Boston)

# To look all the column names we use the names method
names(Boston)

# to define the column names of the database as variables we use the attach method
attach(Boston)

# To create a pair plot we use the pairs function
pairs(Boston)


# For simple linear regression, we use the lm command
# lm(y~x)

lm.fit = lm(medv~lstat)

# We can now check all the info about this linear regression
summary(lm.fit) # Gives a summary including coefficients, t values, p values, min, max, median, etc
coef(lm.fit) # Returns the coefficients of the regression

# To obtain a confidence interval of the coefficient estimates, we use the confint() command
confint(lm.fit)

# The predict() function can be used to obtain confidence and prediction intervals of medv for a given value of lstat
predict(lm.fit,data.frame(lstat=(c(5,10,15))),interval='confidence')

predict(lm.fit,data.frame(lstat=(c(5,10,15))),interval='prediction')

# Now we plot the scatter plot with the least square regression line
plot(lstat,medv,col='blue',pch=20) # col modifies color, pch modifies markers
abline(lm.fit,lwd=3,col='red')

# abline can be used to draw any kind of line, providing the intercept 'a' and slope 'b'
# for example abline(5,6)

# To analyze the residuals, it's often convenient to look at 4 plots at a time, to do this, we modify 
# the plot distribution with par() function
par(mfrow=c(2,2))
plot(lm.fit)

# par() modifies the plots distribution permanently, so a reset is necessary.
par(mfrow=c(1,1))
plot(lstat,medv)

# To analyze the residuals in other ways, the residuals() and rstudent() function works well
plot(predict(lm.fit),residuals(lm.fit))
plot(predict(lm.fit),rstudent(lm.fit))

#################################################
### Multiple Linear Regression
#################################################

# The syntax is lm(y~x1+x2+x3)
lm.fit = lm(medv~lstat+age,data=Boston)
summary(lm.fit)

# To use all the variables instantly for the linear regression
lm.fit = lm(medv~.,data=Boston)
summary(lm.fit)

# With the car library, you can calculate the VIF value, that is, the variance inflation factors
library(car)
vif(lm.fit)

#As it can be seen in the summary, the age variable has a high p-value, so we want to exclude it from the model
lm.fit1 = lm(medv~.-age,data=Boston)
summary(lm.fit1)

# Another good way of excluding variables is using the update function
lm.fit1 = update(lm.fit,~.-age-indus)

#################################################
### Interaction Terms
#################################################

# The * symbol tells R to include each variable and an interaction variable for the regression
# To add an interaction term manually, without including the independent variable you can use the : operator
# lstat:age
summary(lm(medv~lstat*age,data=Boston))

#################################################
### Non-linear transformation
#################################################

# To add a non linear predictor, I() is used
lm.fit2 = lm(medv~lstat+I(lstat^2))
summary(lm.fit2)

# To plot the new regression
plot(medv~lstat)
points(lstat,fitted(lm.fit2),col="red",pch=20)

# To quantify how much better is the quadratic expression to the linear one, the anova function
# does the job. anova stands for Analysis of Variance
lm.fit = lm(medv~lstat)
anova(lm.fit,lm.fit2)

# The anova function performs a hypothesis test comparing the two models. The null hypothesis is that
# the two models fit the data equally well, and the alternative hypothesis is that the full model is
# superior. If the p-value is near 0, it means that the null hypothesis is rejected, meaning that there's no evidence
# that tells that the two models are similar.

par(mfrow=c(2,2))
plot(lm.fit2)

# To produce polynomial regressions, instead of adding I() terms, the poly() function can be used

lm.fit5 = lm(medv~poly(lstat,5))
summary(lm.fit5)

plot(medv~lstat)
points(lstat,fitted(lm.fit5),col="red",pch=20)
# From the p-values, you can see that up to fifth order the parameters coefficients are still relevant.


#########################################
### Qualitative Predictors
#########################################

# The Carseats data of the ISLR library will be analyzed
# We will try to predict Sales from other variables

library(ISLR)
fix(Carseats)
attach(Carseats)

lm.fit = lm(Sales~.+Income:Advertising+Price:Age,data=Carseats)
summary(lm.fit)

# R automatically generates dummy variables for the qualitative variables.
# To check how the dummy variables are included we can use the function contrasts
contrasts(ShelveLoc)

#########################################
### Function writing
#########################################

# To create a function, it must be defined as it follows
LoadLibraries = function(){
  library(ISLR)
  library(MASS)
  print("The libraries have been loaded.")
}

# To execute the function, it must be called with the ()