#######################################
### Non-Linear Modeling
#######################################

library(ISLR)
attach(Wage)

#######################################
### Polynomial Regression and Step
### Functions
#######################################

# For polynomials, we use the command poly(variable,degree)
fit = lm(wage~poly(age,4),data=Wage)
coef(summary(fit))

# The poly() function returns a matrix whose colums are a basis of orthogonal polynomials.
# This means that each column is a linear combination of the variables age, age^2, age^3,age^4
# We can obtain each parameter directly using the attribute raw=TRUE in the function.
# This will change the coefficient estimates, but will not affect the model.
fit2 = lm(wage~poly(age,4,raw=TRUE),data=Wage)
coef(summary(fit2))

# Other ways of fitting this model:
fit2a = lm(wage~age+I(age^2)+I(age^3)+I(age^4),data=Wage)
coef(fit2a)
fit2b = lm(wage~cbind(age,age^2,age^3,age^4),data=Wage)
coef(fit2b)

# Now we'll create a grid of values for age for predictions
agelims = range(age)
age.grid = seq(from=agelims[1],to=agelims[2])
preds = predict(fit,newdata=list(age=age.grid),se=TRUE)
se.bands = cbind(preds$fit+2*preds$se.fit,preds$fit-2*preds$se.fit)

# Plot the data and add the 4th degree polynomial
par(mfrow=c(1,2),mar=c(4.5,4.5,1,1),oma=c(0,0,4,0))
# mar sets the margin size in the following order: bottom, left, top, and right
# oma sets the outer margin area in the same order

plot(age,wage,xlim=agelims,cex=.5,col="darkgrey")
# The cex attribute tells how big to magnify the text, points, etc.
title("Degree-4 Polynomial",outer=T)
lines(age.grid,preds$fit,lwd=2,col="blue")
matlines(age.grid,se.bands,lwd=1,col="blue",lty=3)

# We can now check what we stated before about the coefficients: they won't change
# the model fit
preds2 = predict(fit2,newdata=list(age=age.grid),se=TRUE)
max(abs(preds$fit-preds2$fit))

# Now, to decide the degree of the polynomial that will be used, we can perform an
# analysis of variance (ANOVA, using F-test) in order to test the null hypothesis that a model M1
# is sufficient to explain the data against the alternative hypothesis that a more 
# complex model M2 is needed
fit.1 = lm(wage~age,data=Wage)
fit.2 = lm(wage~poly(age,2),data=Wage)
fit.3 = lm(wage~poly(age,3),data=Wage)
fit.4 = lm(wage~poly(age,4),data=Wage)
fit.5 = lm(wage~poly(age,5),data=Wage)
anova(fit.1,fit.2,fit.3,fit.4,fit.5)

# For the lineal function, the p value is approximately 0 (that's why it doesn't show
# any value). We see that the p-value of the quadratic function is also low, therefore,
# The alternative hypothesis is certainly true (a more complex model is needed). 
# The cubic and degree-4 polynomials have reasonable p-values, whereas the fifth polynomial
# seems unnecessary due to it's high p-value. 

# Instead of using the anova() function, we can use the coefficients of the poly() functions
coef(summary(fit.5))
# The p-values are the same as shown before!
# But, it's better to use the anova() function, because it can be combined with other terms

fit.1 = lm(wage~education+age,data=Wage)
fit.2 = lm(wage~education+poly(age,2),data=Wage)
fit.3 = lm(wage~education+poly(age,3),data=Wage)
anova(fit.1,fit.2,fit.3)

# We can also choose the polynomial degree using cross-validation.
# Now we'll predict if an individual earns more than $250,000 per year
# For this we need to perform a logistic regression

fit = glm(I(wage>250)~poly(age,4),data=Wage,family=binomial)
preds = predict(fit,newdata=list(age=age.grid),se=T)

# The prediction function returns the predictions and confidence intervals for the 
# logit form of the logistic regression. To obtain the confidence interval for 
# Pr(Y=1|X) we need to transform this
pfit = exp(preds$fit)/(1+exp(preds$fit))
se.bands.logit = cbind(preds$fit+2*preds$se.fit, preds$fit-2*preds$se.fit)
se.bands = exp(se.bands.logit)/(1+exp(se.bands.logit))
# We could have directly computed the probabilities using type="response" in the
# predict() function, but this would have returned negative probabilities in the
# confidence intervals, which can't be true!

# We proceed to plot the results
# The type = "n" used in plot() is used to return an blank plot
# The jitter function adds noise to the values, this way observations with the same
# age value don't cover each other up!
plot(age,I(wage>250),xlim=agelims,type="n",ylim=c(0,0.2))
points(jitter(age),I((wage>250)/5),cex=.5,pch="|",col="darkgrey")
lines(age.grid,pfit,lwd=2,col="blue")
matlines(age.grid,se.bands,lwd=1,col="blue",lty=3)

### Step Functions
# To fit a step function, we use the cut() function
table(cut(age,4))
fit = lm(wage~cut(age,4),data=Wage)
coef(summary(fit))
# As we can see, the age<33.5 was cut out. Therefore, the intercept coefficient
# may be interpreted as the average salary for those under 33.5 years, and the others
# can be interpreted as the average additional salary for those in the other age groups

#######################################
### Splines
#######################################

# To fit splines, the splines library is used. As well, an appropriate matrix of basis
# functions is required, for this the bs() function comes in handy. By default, cubic
# splines are produced
par(mfrow=c(1,1))
library(splines)
fit = lm(wage~bs(age,knots=c(25,40,60)),data=Wage)
pred = predict(fit,newdata=list(age=age.grid),se=T)
plot(age,wage,col="gray")
lines(age.grid,pred$fit,lwd=2)
lines(age.grid,pred$fit+2*pred$se,lty="dashed")
lines(age.grid,pred$fit-2*pred$se,lty="dashed")
abline(v=c(25,40,60),lty=2,col="darkgreen")

# Instead of specifying the knots, we can use the function df to produce a spline with
# knots at uniform quantiles of the data
dim(bs(age,knots=c(25,40,60)))
dim(bs(age,df=6)) # Note, df stands for degrees of freedom
attr(bs(age,df=6),"knots")

# If we ever want to create a spline with another degree instead of cubic degree, the
# bs() function comes with a degree attribute that can be modified for this purpose

## Natural Splines
# For this we use the function ns()
fit2 = lm(wage~ns(age,df=4),data=Wage)
pred2 = predict(fit2,newdata=list(age=age.grid),se=T)
lines(age.grid,pred2$fit,col="red",lwd=2)

## Smoothing Splines
# For smoothing splines, the smooth.spline() function is used
# We can either specify the degrees of freedom, or use LOO cross-validation to determine the
# degrees of freedom that will be used
plot(age,wage,xlim=agelims,cex=.5,col="darkgrey")
title("Smoothing Spline")
fit = smooth.spline(age,wage,df=16)
fit2 = smooth.spline(age,wage,cv=TRUE)
fit2$df
lines(fit,col="red",lwd=2)
lines(fit2,col="blue",lwd=2)
legend("topright",legend=c("16 DF", "6.8 DF"),col=c("red","blue"),lty=1,lwd=2,cex=.8)

#######################################
### Local Regression
#######################################

# To perform local regression, the loess() function is used
plot(age,wage,xlim=agelims,cex=.5,col="darkgrey")
title("Local Regression")
fit = loess(wage~age,span=.2,data=Wage)
fit2 = loess(wage~age,span=.5,data=Wage)
lines(age.grid,predict(fit,data.frame(age=age.grid)),col="red",lwd=2)
lines(age.grid,predict(fit2,data.frame(age=age.grid)),col="blue",lwd=2)
legend("topright",legend=c("Span=0.2","Span=0.5"),col=c("red","blue"),lty=1,lwd=2,cex=.8)
# The 0.2 and 0.5 spans means that each neighborhood of the points consists of the 20%
# and 50% of the observations. The larger the span, the smoother the fit.

#######################################
### General Additive Models (GAM)
#######################################

# We'll try to predict wage using natural spline functions of year and age, and using
# education as a qualitative variable.
gaml = lm(wage~ns(year,4)+ns(age,5)+education,data=Wage)

# Now we'll fit the model using smoothing splines instead of natural splines. In order
# to fit more general sorts of GAMs that have components that can't be expressed as
# basis functions, the gam library is required

# install.packages("gam")
library(gam)
# To specify smoothing splines, the s() function is used. Year will be specified with
# 4 degrees of freedom, and age with 5. Education is leaved as it is because it's a dummy
# variable
gam.m3 = gam(wage~s(year,4)+s(age,5)+education,data=Wage)
par(mfrow=c(1,3))
plot(gam.m3,se=TRUE,col="blue")

# We see that the year functions looks linear. We'll perform a series of ANOVA tests
# to check which of the following models is the best: a GAM function that excludes year,
# a GAM that uses a linear function of year, and a GAM that uses a spline function of year
gam.m1 = gam(wage~s(age,5)+education,data=Wage)
gam.m2 = gam(wage~year+s(age,5)+education,data=Wage)
anova(gam.m1,gam.m2,gam.m3,test="F")
# We see that a GAM with a linear function of year is better than a GAM that doesn't include
# year (p-value = 0.00014). However, there's no evidence that a non-linear function of year
# is needed (p-value = 0.349)
# The summary() function produces a summary of the gam fit
summary(gam.m3)
# The Anova for Nonparametric Effects p-values correspond to a null hypthesis of a linear
# relationship versus the alternative of a non'linear relationship.
# The large p-value of year reinforces the conclusion of the ANOVA test that a linear
# function is adequate for this term, but there's clear evidence that age must be non linear

# We proceed with the predictions
preds = predict(gam.m2,newdata=Wage)

# We can also use local regression as building blocks in a GAM using the lo() function
gam.lo = gam(wage~s(year,df=4)+lo(age,span=0.7)+education,data=Wage)
plot.Gam(gam.lo,se=TRUE,col="green")

# The lo() function can also be used to create interactions before calling the gam()
# function
gam.lo.i = gam(wage~lo(year,age,span=0.5)+education,data=Wage)
# The plot of this corresponds to a two-dimensional surface, that can be plotted using
# the akima library
# install.packages('akima')
library(akima)
par(mfrow=c(1,2))
plot(gam.lo.i)

## Logistic regression using GAM
gam.lr = gam(I(wage>250)~year+s(age,df=5)+education,family=binomial,data=Wage)
par(mfrow=c(1,3))
plot(gam.lr,se=T,col="green")

# We can see that there are no high earners for the <HS category
table(education,I(wage>250))
# Therefore, we perform a fit without using this category
gam.lr.s = gam(I(wage>250)~year+s(age,df=5)+education,family=binomial,data=Wage,subset=(education!="1. < HS Grad"))
plot(gam.lr.s,se=T,col="green")
