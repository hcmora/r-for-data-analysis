require(ISLR) # require is equivalent to library
names(Smarket)
?Smarket
fix(Smarket)
dim(Smarket)
summary(Smarket)
pairs(Smarket,col=Smarket$Direction) # To differentiate the plot accordingly to the Direction, we use col
cor(Smarket[,-9]) # We exclude the direction column, because it's not qualitative
attach(Smarket)
plot(Volume)

## Logistic Regression

glm.fit = glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data=Smarket,family=binomial)
summary(glm.fit)

# The variable with smallest p value is Lag1
# We can access the coeficients by using coef(glm.fit)
# We can have the p values of the coefficients by using summary(glm.fit)$coef[,4]

# To predict probabilities, we use the predict function with type='respose'
# To check which value of Direction corresponds to probability 1 or 0, we use the contrasts function
glm.probs = predict(glm.fit,type='response')
glm.probs[1:10]
contrasts(Direction)

# To transform these probabilities to Up or Down, we simply create a vector full of Down, and use the probabilities
# to change it accordingly

glm.pred = rep("Down",1250)
glm.pred[glm.probs>0.5] = "Up"

# Another way of doing this would have been
# glm.pred = ifelse(glm.probs>0.5,"Up","Down")

# With the table function we can create a confusion matrix to check how many instances were correctly classified
table(glm.pred,Direction) # The diagonal elements of this matrix are the correct answers
# We check the percentage of correct answers
(145+507)/1250
mean(glm.pred==Direction)

# The problem with the model that we just made is that this 52.2% corresponds to the training error, because
# we used the whole data to train the model.

# Now we'll separate the data for training and for test to check the real test error of this model
train=(Year<2005) # boolean vector with with True on all the data before 2005
Smarket_2005 = Smarket[!train,] # test data
dim(Smarket_2005)
Direction_2005=Direction[!train]

# Now we use this to create our new training data, using the subset attribute to include the boolean vector
glm.fit = glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume, data=Smarket, family=binomial,subset=train)
glm.probs = predict(glm.fit,Smarket_2005,type='response')
glm.pred = rep("Down",252)
glm.pred[glm.probs>0.5] = "Up"
table(glm.pred,Direction_2005)
mean(glm.pred==Direction_2005)
mean(glm.pred!=Direction_2005) # This is the Test Error Rate, how much it fails!

## Now we proceed to exclude the variables that had realy large p-values to check if the model improves
glm.fit=glm(Direction~Lag1+Lag2,data=Smarket,family=binomial,subset=train)
glm.probs = predict(glm.fit,Smarket_2005,type="response")
glm.pred = rep("Down",252)
glm.pred[glm.probs>0.5]="Up"
table(glm.pred,Direction_2005)
mean(glm.pred==Direction_2005)
# We can see that when the model predicts that the market will fall, it's right 50%
# But when it predicts that it will go up, it's right 106/(76+106)=58.2% of the times

################################################
### Linear Discriminant Analysis
################################################

# To use LDA, we have to import the MASS library
library(MASS)
lda.fit = lda(Direction~Lag1+Lag2,data=Smarket,subset=train)
lda.fit
# The prior probabilities correspond to pi1 and pi2 respectively, that said, it means that 
# 49.2% of the training observation correspond to Down situations and 50.8% to up

# The group mean shows that before each Up day, the two days before will probably have negative returns
# and in Down days, the two days before will probably have positive returns
# The coefficients are the values that multiplicate the elements of X=x, if -0.642*Lag1-0.514*Lag2 is large
# the LDA classifier will predict a market increase, if not, it will predict a decrease
plot(lda.fit)

# Using the function predict on lda.fit returns 3 elements.
# Class is the prediction, posterior returns a matrix with the probability of the instance belonging to another class
# and x contains the linear discriminants
lda.pred = predict(lda.fit,Smarket_2005)
names(lda.pred)
lda.class = lda.pred$class
table(lda.class,Direction_2005)
mean(lda.class==Direction_2005)
# We see the same results that with the logistic regression.
# To obtain the values applying the 50% to the posterior probabilities:
sum(lda.pred$posterior[,1]>=0.5)
sum(lda.pred$posterior[,1]<0.5)
# Notice that in this case, the posterior probability calculates 1 as when the market decreases
lda.pred$posterior[1:20,1]
lda.class[1:20]
# If we want to change the threshold, we use the probability that we want in the posterior probability
# For example, if we wanted the posterior probability of decrease to be 90%
sum(lda.pred$posterior[,1]>=0.9)

################################################
### Quadratic Discriminant Analysis
################################################

# Same as LDA, it's part of the MASS library
qda.fit = qda(Direction~Lag1+Lag2,data=Smarket,subset=train)
qda.fit
# The predict() function works the same as in LDA
qda.class = predict(qda.fit,Smarket_2005)$class
table(qda.class,Direction_2005)
mean(qda.class==Direction_2005)
# In this case, QDA is actually more accurate

################################################
### K-Nearest Neighbors
################################################

# knn() function forms predictions in one command
# it requires 4 inputs:
# 1. A matrix containing the predictors associated to the training data
# 2. A matrix containing the predictors associated with the data for which we wish to make the predictions
# 3. A vector containing the class labels for the training observations
# 4. A value for K, the number of nearest neighbors to be used by the classifier

# install.packages('class')
library(class)
# the cbind function allows to create a matrix from vectors
train.X = cbind(Lag1,Lag2)[train,]
test.X = cbind(Lag1,Lag2)[!train,]
train.Direction = Direction[train]

set.seed(1) # used for cases where there are tied observations, so the random pick is the same as the book
knn.pred = knn(train.X,test.X,train.Direction,k=1)
table(knn.pred,Direction_2005)
mean(knn.pred==Direction_2005)

# Using k = 3
knn.pred = knn(train.X,test.X,train.Direction,k=3)
table(knn.pred,Direction_2005)
mean(knn.pred==Direction_2005)

# Increasing k doesn't improve the results, it seems that the best model for this data is QDA

################################################
### Application to Insurance Data
################################################

# We'll use KNN for the Caravan dataset of the ISLR library. It includes 85 predictors for 5822 individuals
dim(Caravan)
fix(Caravan)
attach(Caravan)
summary(Purchase)

# KNN is heavily influentied by the distance of parameters. For example, if we had the predictors salary and
# age, for KNN USD 1000 is a huge difference and 50 years of difference is not, even though for us this isn't the
# same case. To solve this, we have to standarize the data, so it gives the same weight to each parameter and
# to avoid inconsistencies in others (for example, in salary if it was in CLP instead of USD, the distance would
# be even bigger in number, yielding different results)

standardized.X = scale(Caravan[,-86]) # The 86 column is excluded because that is our result Purchase
var(Caravan[,1])
var(Caravan[,2])
var(standardized.X[,1])
var(standardized.X[,2])

# Now each parameter has mean 0 and std = 1
# Now we split the observations. The first 1000 will be the test values

test = 1:1000
train.X = standardized.X[-test,]
test.X = standardized.X[test,]
train.Y = Purchase[-test]
test.Y = Purchase[test]
set.seed(1)
knn.pred = knn(train.X,test.X,train.Y,k=1)
mean(knn.pred!=test.Y)
mean(test.Y!="No")

# Even though it seems that the KNN was a good predictor, being different to test.Y in 11.8%
# It happens that the number of Yes in the data is only 5.9%, so basically, just saying no to everything
# would give a better result

# Instead of analyzing the error rate as a whole, let's try to see if the KNN method can predict
# and identify the consumers that have higher chances of buying an insurance
table(knn.pred,test.Y)
# The success rate is 9/(68+9) = 11.7%. Fairly good, it means that 1 out of 10 people chosen by this method
# will actually buy the insurance. This works better than trying to sell it to anyone, where the success rate
# is really low.

# Let's analyze with K = 3
knn.pred = knn(train.X,test.X,train.Y,k=3)
table(knn.pred,test.Y)
5/26
# Success rate of 19% !

# K = 5
# Let's analyze with K = 3
knn.pred = knn(train.X,test.X,train.Y,k=5)
table(knn.pred,test.Y)
4/15
# Success rate of 26.7%!

# Now let's check if a logistic regression can yield better results!
glm.fit = glm(Purchase~.,data=Caravan,family=binomial,subset=-test)
glm.probs = predict(glm.fit,Caravan[test,],type="response")
glm.pred = rep("No",1000)
glm.pred[glm.probs>0.5] = "Yes"
table(glm.pred,test.Y)
# Nothing is predicted! Let's change the threshold to 0.25. That is, assume that anyone that has a chance to buy
# greater than 25% will buy an insurance
glm.pred = rep("No",1000)
glm.pred[glm.probs>0.25] = "Yes"
table(glm.pred,test.Y)
11/33
# 33.3% of success rate! 