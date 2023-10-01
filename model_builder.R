#########################################################################################################################
###
### R code to build random forest model to predict NBA success from LIWC-22 features
### Author: Sean Farrell
### Last update: October 1st 2023
###
#########################################################################################################################

# Load libraries 
library(randomForest)
library(pbapply)
library(data.table)
library(caret)
library(pbmcapply)
library(dplyr)
library(fANCOVA)
library(ggplot2)
library(precrec)
library(multiROC)
library(yardstick)

# Load the training set
train <- read.csv("training_set.csv")

# Convert label column to factor
train$LABEL <- as.factor(train$LABEL)

# Perform 10-fold cross-validation repeated 10 times to tune the mtry hyperparameter
r <- lapply(1:10,function(i){
  
  # Randomly split training set into 10 chunks 
  idx <- 1:nrow(train)
  folds <- split(idx,sample(1:10,length(idx),replace=T))
  
  # Perform 10-fold cross-validation
  d <- lapply(1:10,function(k){
    print(paste(i,k))
    
    # Select the current fold for testing
    idx <- folds[[k]]
    test <- train[idx,]
    
    # Select the other 9 folds for training    
    sub <- train[-idx,-c(1:2)]
    lab <- train$LABEL[-idx]
    
    # Grid-search mtry values
    s <- pbmclapply(2:ncol(sub),function(mtry){
      
      # Build random forest model with 5000 trees
      rf <- randomForest(sub,lab,ntree=5000,mtry=mtry)
      
      # Predict on hold-out sample
      pred <- predict(rf,subset(test,select=names(sub)))
      output <- data.frame(mtry=mtry,Label=test$LABEL,Pred=pred)
      
      return(output)  
    },mc.cores=detectCores()-2)
    output <- rbindlist(s)
    return(output)
      
  })
  test <- rbindlist(d)
  return(test)
})
test <-rbindlist(r)

# Calculate accuracy for each mtry run
test$Cor <- 0
test$Cor[test$Label == test$Pred] <- 1
test <- test %>% group_by(mtry) %>% summarise(Cor=sum(Cor),Count=n()) %>% mutate(Accuracy=Cor/Count)

# Plot mtry vs accuracy
plot(test$mtry,test$Accuracy)

# Fit a local regression curve to the mtry vs accuracy plot
lo <- loess.as(test$mtry,test$Accuracy,deg=2,plot=T)
pred <- predict(lo)

# Get the mtry with maximum accuracy
mtry <- test$mtry[which(pred == max(pred))]
mtry
# 19

# Perform 10-fold cross-validation repeated 10 times to estimate final accuracy
r <- lapply(1:10,function(i){
  # Randomly split training set into 10 chunks 
  idx <- 1:nrow(train)
  folds <- split(idx,sample(1:10,length(idx),replace=T))
  
  # Perform 10-fold cross-validation
  d <- lapply(1:10,function(k){
    print(paste(i,k))
    
    # Select the current fold for testing
    idx <- folds[[k]]
    test <- train[idx,]
    
    # Select the other 9 folds for training    
    sub <- train[-idx,-c(1:2)]
    lab <- train$LABEL[-idx]
    
    # Build small forests in parallel to build up 10,000 tree forest
    s <- pbmclapply(1:1000,function(j){
      # Build random forest
      rf <- randomForest(sub,lab,ntree=10,mtry=mtry)
      return(rf)
    },mc.cores=detectCores()-2)
    rf <- do.call(randomForest::combine,s)
    
    # Predict on hold-out set
    pred <- predict(rf,subset(test,select=names(sub)))
    output <- data.frame(cbind(test,Pred=pred))
    
    return(output)
  })
  cv.df <- rbindlist(d)
  
  return(cv.df)
})
cv.df <- rbindlist(r)

# Display conusion matrix
confusionMatrix(cv.df$Pred,cv.df$LABEL)

# Plot confusion matrix 
cm <- cv.df %>% conf_mat(LABEL, Pred)
autoplot(cm,type="heatmap") +  scale_fill_gradient(low = "blue", high = "red")

