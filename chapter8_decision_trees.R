#######################################
### Decision Trees
#######################################

# We use the tree library for the construction of decision trees
# install.packages('tree')
library(tree)

#######################################
### Classification Trees
#######################################

# We'll analyze the Carseats data set. We'll define the Sales variable as a binary one called
# High, where it will be Yes if Sales exceeds 8, or not otherwise
library(ISLR)
attach(Carseats)
High = ifelse(Sales<=8,"No","Yes")
Carseats = data.frame(Carseats,High)

# We now use the tree() function to fit a classification tree
tree.carseats = tree(High~.-Sales,Carseats)

# The summary function lists the variables that are used as internal nodes in the tree,
# the number of terminal nodes and the training error rate
summary(tree.carseats)

# We can now display the tree. The plot() functions plots the tree, and the text() function
# displays the node labels. The argument pretty=0 instructs R to include the category
# names for any qualitative predictors, rather than simply displaying a letter for each
# category
plot(tree.carseats)
text(tree.carseats,pretty=0)

# We see that the most important variable is the shelve location, since it's the first branch
# Calling the tree object by itself prints the branches. The * indicates terminal nodes
tree.carseats

# We'll now estimate the performance of the tree classification method
set.seed(2)
train = sample(1:nrow(Carseats),200)
Carseats.test = Carseats[-train,]
High.test = High[-train]
tree.carseats = tree(High~.-Sales,Carseats,subset=train)
tree.pred = predict(tree.carseats,Carseats.test,type="class")
table(High.test,tree.pred)
(86+59)/200

# Now we'll consider pruning the tree. The cv.tree() function performs cross-validation for
# this purpose. The argument FUN=prune.misclass is used to indicate that the classification
# error rate is the parameter for the pruning process, instead the deviance that comes by 
# default.
# The cv.tree() returns the number of terminal nodes (size), the cv error rate (dev) and the 
# cost complexity parameter used (as k, this is the same as the alpha seen in the book)

set.seed(3)
cv.carseats = cv.tree(tree.carseats,FUN=prune.misclass)
names(cv.carseats)
cv.carseats

# We see that the tree with 9 terminal nodes has the lowest cross-validation error rate.
# We can plot the error rate as a function of both size and k
par(mfrow=c(1,2))
plot(cv.carseats$size,cv.carseats$dev,type="b")
plot(cv.carseats$k,cv.carseats$dev,type="b")

# We apply the prune.misclass() function tu prune the tree
prune.carseats = prune.misclass(tree.carseats,best=9)
plot(prune.carseats)
text(prune.carseats,pretty=0)

# Let's see how well does this model perform
tree.pred = predict(prune.carseats,Carseats.test,type="class")
table(High.test,tree.pred)
(93+59)/200
# It improved the results, as well as making a more interpretable tree

#######################################
### Regression Trees
#######################################

# We'll fit a regression tree for the Boston dataset
library(MASS)
set.seed(1)
train = sample(1:nrow(Boston),nrow(Boston)/2)
tree.boston = tree(medv~.,Boston,subset=train)
summary(tree.boston)

# We see that only 3 variables are used.
# Notice that the Residual mean deviance is the same as the SSE.
# We proceed to plot the tree

par(mfrow=c(1,1))
plot(tree.boston)
text(tree.boston,pretty=0)

# We see that for lower values of lstat, the medv increases. 
# We check if we can improve this by prunning

cv.boston = cv.tree(tree.boston)
plot(cv.boston$size,cv.boston$dev,type="b")

# We see that the best prunning option is with 4 terminal nodes
prune.boston = prune.tree(tree.boston,best=4)
plot(prune.boston)
text(prune.boston,pretty = 0)

# We proceed to make the predictions with this prunned tree
yhat = predict(prune.boston,newdata=Boston[-train,])
boston.test = Boston[-train,"medv"]
plot(yhat,boston.test)
abline(0,1)
mean((yhat-boston.test)^2)

# We check the same with the unprunned tree
yhat = predict(tree.boston,newdata=Boston[-train,])
plot(yhat,boston.test)
abline(0,1)
mean((yhat-boston.test)^2)

# We see that the prunned tree yields better results!

#######################################
### Bagging and Random Forests
#######################################

# We use the randomForest() library that comes with R.
# Notice that bagging is the same as random forests, with m = n.
# Bagging: construct B regression trees using B bootstrapped training sets, and average
# the resulting predictions
# Random Forests: method that decorrelates the trees by excluding parameters that can be
# selected when a split tree is considered. The exclusion of parameters is random.

# install.packages('randomForest')
library(randomForest)
set.seed(1)
bag.boston = randomForest(medv~.,data=Boston,subset=train,mtry=13,importance=TRUE)
bag.boston

# The mtry attribute defines how many parameters can be used during the random forest process.
# Since the data is composed of 13 predictors, all the variables are used, therefore it
# is a bagging method

yhat.bag = predict(bag.boston,newdata=Boston[-train,])
plot(yhat.bag,boston.test)
abline(0,1)
mean((yhat.bag-boston.test)^2)

# We see that the MSE is considerably better than using the prunning method!
# We can also change the number of trees grown by the randomForest() method specifying
# ntree
bag.boston = randomForest(medv~.,data=Boston,subset=train,mtry=13,ntree=25)
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
mean((yhat.bag-boston.test)^2)

# To create a random forest, we simply specify another number for mrty. By default, the 
# function uses p/3 variables for regression trees, and sqrt(p) for classification trees
# We'll use p=6
set.seed(1)
rf.boston = randomForest(medv~.,data=Boston,subset=train,mtry=6,importance=TRUE)
yhat.rf = predict(rf.boston,newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2)

# The MSE is lower, meaning that the random forest method returned an improvement over
# bagging.
# With the importance() function we can see the importance of each variable
importance(rf.boston)

# The %IncMSE represents the mean decrease of accuracy in predictions if the variable 
# is excluded. The IncNodePurity is the total decrease in node impurity that results
# from splits over that variable, averaged over all trees. This is measured in RSS for
# regression trees and deviance for classification trees.
# This can be plot with the varImpPlot function

varImpPlot(rf.boston)

#######################################
### Boosting
#######################################

# The gbm package and gbm() function is used for boosting. 
# Since it's a regression problem, the distribution="gaussian" attribute is used in the
# function. For classification problems, distribution="bernoulli".
# The n.trees specifies the number of trees, and interaction.depth specifies the limit
# of the depth of each tree.

# install.packages('gbm')
library(gbm)
set.seed(1)
boost.boston = gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4)
summary(boost.boston)

# We see that lstat and rm are the most important variables.
# We can produce partial dependence plot that illustrate the marginal effect of the selected
# variables on the response after integrating out the other variables.

par(mfrow=c(1,2))
a = plot(boost.boston,i="rm")
b = plot(boost.boston,i="lstat")
print(a, position = c(0, 0, 0.5, 1), more = TRUE)
print(b, position = c(0.5, 0, 1, 1))

# Predictions
yhat.boost = predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost-boston.test)^2)

# In this case, the MSE is even lower!
# We can modify the lambda parameter for the boosting model to check if it improves the MSE.
# By default, the shrinkage parameter is 0.001. We'll test it wit 0.2

boost.boston = gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4,shrinkage=0.2,verbose=F) 
yhat.boost = predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost-boston.test)^2)

# In this case, increasing the shrinkage parameter increased the MSE.