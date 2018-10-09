# Chapter 8 Lab: Decision Trees - Classification Trees, Regression Trees, Bagging 
#   and Random Forest, and Boosting
# modified by: Billy Raseman (https://github.com/wraseman)

## clear environment
rm(list=ls())

# Fitting Classification Trees

## load packages
library(tree)  # classification and regression trees
library(ISLR)  # data for an "introduction to statistical learning with applications in R"
library(tidyverse)  # for ggplot2, tidyr, etc.

## load dataset
attach(Carseats)

## explore dataset
Carseats <- as.tibble(Carseats)  # make it easier to read
Carseats
ggplot(Carseats, aes(x=Sales)) + 
  geom_histogram()

## Add column for high/low sales
Carseats <- mutate(Carseats, High=ifelse(Sales<=8,"No","Yes") %>% factor)  # need to convert from type "chr" to "factor"
### OR using base R...
### High=ifelse(Sales<=8,"No","Yes")
### Carseats=data.frame(Carseats,High)

## Fit tree to data and visualize
tree.carseats=tree(High~.-Sales, Carseats)  # exclude Sales because it was used to create High variable
summary(tree.carseats)
plot(tree.carseats)  # plot decision tree
text(tree.carseats,pretty=0)  # add text to plot
tree.carseats$frame  # print out tree as dataframe

## Create a training and test set and evaluate performance
set.seed(2)  # for reproducibility
n.train <- 250  # number of datapoints used for training
train=sample(1:nrow(Carseats), n.train)  # sample data without replacement
Carseats.test=Carseats[-train,]  # create test dataset by removing training data
High.test=Carseats$High[-train]  # create test dataset by removing training data
tree.carseats=tree(High~.-Sales,Carseats,subset=train)  # fit tree on training data (again removing Sales from dataset)
tree.pred=predict(tree.carseats,Carseats.test,type="class") # predict class labels 
### see ?predict.tree for documentation
table.tree <- table(tree.pred,High.test)  # diagonals are number of correct
error.rate = (68+49)/(nrow(Carseats) - n.train)
error.rate  # show error rate

## The tree doesn't predict well--it is too bushy. Let's prune it.
set.seed(3)
cv.carseats=cv.tree(tree.carseats,FUN=prune.misclass)  # 10-fold cross validation
cv.carseats  
### size: size of tree (number of terminal nodes, I believe)
### dev: deviance associated with each tree
### k: value of cost complexity parameter
plot(cv.carseats)
### pick a value near the minimum (say, 14) for pruned tree

## Fit pruned tree
prune.carseats=prune.misclass(tree.carseats,best=14)
plot(prune.carseats)  # much easier to read the pruned tree
text(prune.carseats,pretty=0)
tree.pred=predict(prune.carseats,Carseats.test,type="class")
table.prune <- table(tree.pred,High.test)  # diagonals are number of correct
error.rate = (68+49)/(nrow(Carseats) - n.train)
error.rate  # the same as before


# Fitting Regression Trees
library(MASS)
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston=tree(medv~.,Boston,subset=train)
summary(tree.boston)
plot(tree.boston)
text(tree.boston,pretty=0)
cv.boston=cv.tree(tree.boston)
plot(cv.boston$size,cv.boston$dev,type='b')
prune.boston=prune.tree(tree.boston,best=5)
plot(prune.boston)
text(prune.boston,pretty=0)
yhat=predict(tree.boston,newdata=Boston[-train,])
boston.test=Boston[-train,"medv"]
plot(yhat,boston.test)
abline(0,1)
mean((yhat-boston.test)^2)

# Bagging and Random Forests
library(randomForest)
set.seed(1)
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,importance=TRUE)
bag.boston
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
plot(yhat.bag, boston.test)
abline(0,1)
mean((yhat.bag-boston.test)^2)
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,ntree=25)
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
mean((yhat.bag-boston.test)^2)
set.seed(1)
rf.boston=randomForest(medv~.,data=Boston,subset=train,mtry=6,importance=TRUE)
yhat.rf = predict(rf.boston,newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2)
importance(rf.boston)
varImpPlot(rf.boston)

# Boosting
library(gbm)
set.seed(1)
boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4)
summary(boost.boston)
par(mfrow=c(1,2))
plot(boost.boston,i="rm")
plot(boost.boston,i="lstat")
yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost-boston.test)^2)
boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4,shrinkage=0.2,verbose=F)
yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost-boston.test)^2)

