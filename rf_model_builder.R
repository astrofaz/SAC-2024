#########################################################################################################################
###
### R code to build random forest model to predict NBA success from LIWC-22 features
### Author: Sean Farrell
### Last update: December 1st 2023
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
library(smotefamily)
library(pROC)
library(fANCOVA)

# Load the training set
train <- read.csv("training_set.csv")

# Convert label column to factor
train$LABEL <- as.factor(train$LABEL)

# Drop low base rate features, word count, words per sentence and any related to punctuation
d <- lapply(7:ncol(train),function(i){
  m1 <- median(train[,i])
  m2 <- mean(train[,i])
  
  output <- data.frame(Feature=names(train)[i],Mean=m2,Median=m1)
  return(output)
})
temp <- rbindlist(d)

idx <- which(temp$Median < 0.5)
temp <- temp[idx,]

drop_vars <- c("WC","WPS","AllPunc","Period","Comma","QMark","Exclam","Apostro","OtherP","Emoji",temp$Feature)
train <- train[,!names(train) %in% drop_vars]

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
    
    # Select training set from remaining 9 folds, discarding word count and all punctuation features
    sub <- train[-idx,-c(1:5)]
    lab <- train$LABEL[-idx]
    
    # Grid-search mtry values
    s <- pbmclapply(2:ncol(sub),function(mtry){
      
      # Build random forest model with 1000 trees
      rf <- randomForest(sub,lab,ntree=1000,mtry=mtry)
      
      # Predict on hold-out sample
      pred <- predict(rf,subset(test,select=names(sub)))
      prob <- as.data.frame(predict(rf,subset(test,select=names(sub)),"prob"))
      
      output <- data.frame(mtry=mtry,Label=test$LABEL,prob,Pred=pred)
      
      return(output)  
    },mc.cores=detectCores())
    output <- rbindlist(s)
    return(output)
      
  })
  test <- rbindlist(d)
  return(test)
})
test <-rbindlist(r)

# Calculate metrics for each mtry value
mtry <- unique(test$mtry)
d <- pbmclapply(mtry,function(m){
  output <- test[test$mtry == m,]
  
  # Calculate confusion matrix metrics
  cm <- confusionMatrix(output$Pred,output$Label)
  cm <- data.frame(t(cm$byClass))
  
  # Get AUC from ROC
  r <- roc(data=output, Label, Y,auc=T,ci=T)
  auc <- as.numeric(r$ci)
  
  output <- data.frame(cbind(mtry=m,cm,AUC=auc[2],UL=auc[3],LL=auc[1]))
  return(output)
},mc.cores=detectCores())
test <- rbindlist(d)

# Fit a local regression model to mtry vs AUC plot to get optimal mtry value
lo <- loess.as(test$mtry,test$AUC,plot=T,deg=2)
pred <- predict(lo)
mtry <- test$mtry[which(pred == max(pred))]
mtry
# 12

# Perform 10-fold cross-validation repeated 10 times to estimate final accuracy
mtry <- 12
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
    
    # Select training set from remaining 9 folds, discarding word count and all punctuation features
    sub <- train[-idx,-c(1:5)]
    lab <- train$LABEL[-idx]
    
    # Build small forests in parallel to build up 10,000 tree forest
    s <- pbmclapply(1:1000,function(j){
      # Build random forest
      rf <- randomForest(sub,lab,ntree=10,mtry=mtry)
      
      return(rf)
    },mc.cores=detectCores())
    rf <- do.call(randomForest::combine,s)
    
    # Predict on hold-out set
    pred <- predict(rf,subset(test,select=names(sub)))
    prob <- as.data.frame(predict(rf,subset(test,select=names(sub)),"prob"))
    
    output <- data.frame(cbind(test,prob,Pred=pred))
    
    return(output)
  })
  cv.df <- rbindlist(d)
  
  return(cv.df)
})
cv.df <- rbindlist(r)
save(cv.df,file="cross_validated_train.rda")

# Display confusion matrix
confusionMatrix(cv.df$Pred,cv.df$LABEL,positive = "Y")

# Confusion Matrix and Statistics
# 
# Reference
# Prediction    N    Y
# N 5138 2732
# Y 1732 5728
# 
# Accuracy : 0.7088         
# 95% CI : (0.7015, 0.716)
# No Information Rate : 0.5519         
# P-Value [Acc > NIR] : < 2.2e-16      
# 
# Kappa : 0.4192         
# 
# Mcnemar's Test P-Value : < 2.2e-16      
#                                          
#             Sensitivity : 0.6771         
#             Specificity : 0.7479         
#          Pos Pred Value : 0.7678         
#          Neg Pred Value : 0.6529         
#              Prevalence : 0.5519         
#          Detection Rate : 0.3736         
#    Detection Prevalence : 0.4866         
#       Balanced Accuracy : 0.7125         
#                                          
#        'Positive' Class : Y   
       
### Explore prob thresholds to balance class predictions
d <- pbmclapply(seq(from=0.3,to=0.7,by=0.01),function(i){
  output <- cv.df
  output$Pred <- "N"
  idx <- which(output$Y >= i)  
  output$Pred[idx] <- "Y"

  cm <- confusionMatrix(output$Pred,output$LABEL)
  cm <- data.frame(t(cm$byClass))
  

  output <- data.frame(cbind(threshold=i,cm))    
  return(output)
},mc.cores=detectCores())
temp <- rbindlist(d)

summary(temp)

# Drop NA's
temp <- temp[is.na(temp$F1)==F,]

# Plot precision-recall curve
ggplot(temp) + geom_line(aes(x=Precision,y=Recall),color="red") + geom_point(aes(x=Precision,y=Recall,size=threshold),color="red")

# Calculate difference between positive and negative predicted accuracies
temp$Dif <- temp$Pos.Pred.Value - temp$Neg.Pred.Value
temp[which(abs(temp$Dif) == min(abs(temp$Dif))),]
# Optimal threshold = 0.44 to balance class accuracies


# Re-calculate confusion matrix using new threshold
temp <- cv.df
temp$Pred  <- "N"
temp$Pred[temp$Y >= 0.44] <- "Y"

confusionMatrix(temp$Pred,temp$LABEL,positive = "Y")

# Confusion Matrix and Statistics
# 
# Reference
# Prediction    N    Y
# N 3713 1652
# Y 3157 6808
# 
# Accuracy : 0.6863          
# 95% CI : (0.6789, 0.6936)
# No Information Rate : 0.5519          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.3524          
# 
# Mcnemar's Test P-Value : < 2.2e-16       
#                                           
#             Sensitivity : 0.8047          
#             Specificity : 0.5405          
#          Pos Pred Value : 0.6832          
#          Neg Pred Value : 0.6921          
#              Prevalence : 0.5519          
#          Detection Rate : 0.4441          
#    Detection Prevalence : 0.6500          
#       Balanced Accuracy : 0.6726          
#                                           
#        'Positive' Class : Y 

# Calculate how many sigma this result is
sigma <- (0.6863 - 0.6789)/1.96
(0.6863 - 0.5519)/sigma
# 35.59784

# Plot confusion matrix 
cm <- temp %>% conf_mat(LABEL, Pred)
temp <- data.frame(Prediction = c("N","Y","N","Y"), Truth = c("N","N","Y","Y"),Frequency=as.numeric(cm$table))

ggplot(temp) + geom_tile(aes(x=Truth, y=Prediction, fill=Frequency)) +
  geom_text(aes(Truth, Prediction, label = Frequency), color = "white", size = 5) +
  scale_fill_gradient(low = "blue", high = "red") +  theme_minimal() + 
  theme(axis.text = element_text(size=12),axis.title =  element_text(size=15))



### Generate Receiver Operator Curve (ROC)
r <- roc(data=cv.df, LABEL, Y,auc=T,direction="auto",ci=T)

temp <- data.frame(Specificity = r$specificities,Sensitivity=r$sensitivities)
temp2 <- data.frame(Specificity = c(0,1),Sensitivity=c(0,1))
ggplot(temp) + geom_line(aes(x=1-Specificity,y=Sensitivity),color="blue") + geom_line(data=temp2,aes(x=Specificity,y=Sensitivity),linetype="dashed") +
  theme_bw()

# Get area under curve
r

# Call:
#   roc.data.frame(data = cv.df, response = LABEL, predictor = Y,     auc = T, direction = "auto", ci = T)
# 
# Data: Y in 6870 controls (LABEL N) < 8460 cases (LABEL Y).
# Area under the curve: 0.772
# 95% CI: 0.7647-0.7793 (DeLong)


### Build final model using all data to get feature importance
sub <- train[,-c(1:6)]
lab <- train$LABEL


# Build small forests in parallel to build up 10,000 tree forest
s <- pbmclapply(1:1000,function(j){
  # Build random forest
  rf <- randomForest(sub,lab,ntree=10,mtry=mtry,localImp = TRUE)
  return(rf)
},mc.cores=detectCores())
rf <- do.call(randomForest::combine,s)
save(rf,file="final_rf_model.rda")

# Extract feature imporance
imp <- as.data.frame(rf$importance)
imp$Feature <- row.names(imp)
imp <- imp[order(imp$MeanDecreaseGini),]
imp$Feature <- factor(imp$Feature,levels=imp$Feature)
imp <- imp[order(-imp$MeanDecreaseGini),]

# Plot importance
ggplot(imp) + geom_bar(aes(x=MeanDecreaseGini,y=Feature,fill=Feature),stat = "identity") + 
  theme_bw() + theme(legend.position = "none")
  

### Generate partial dependence plots

# First, run predict on original training set (an original sin, I know) 
pred.orig <- as.data.frame(predict(rf,sub,"prob"))

# Next, cycle through each feature
d <- lapply(1:ncol(sub),function(i){
  print(names(sub)[i])
  min <- min(sub[,i])
  max <- max(sub[,i])
  
  # Iterate between min and max values with 100 steps
  x <- seq(from=min,to=max,by=(max-min)/100)
  s <- pbmclapply(x,function(val){
    
    # Modify the feature value
    temp <- sub
    temp[,i] <- val
    
    # Re-score using the RF model
    pred <- as.data.frame(predict(rf,temp,"prob"))
    output <- data.frame(Feature=names(temp)[i],Val=val,Pred=pred$Y,Orig=pred.orig$Y)
    output$Diff <- output$Orig-output$Pred
    return(output)    
  },mc.cores=detectCores())
  output <- rbindlist(s)
  
  return(output)
})
data <- rbindlist(d)
save(data,file="partial_dependence.rda")

# Aggregate and calculate mean probabilities for each feature and value
data <- data %>% group_by(Feature,Val) %>% summarise_all(list(mean))

# Order features by importance
imp <- as.data.frame(rf$importance)
imp$Feature <- row.names(imp)
imp <- imp[order(-imp$MeanDecreaseGini),]
data$Feature <- factor(data$Feature,levels=imp$Feature)

# Calculate percentage of overall importance
imp$Percentage <- round(100*imp$MeanDecreaseGini/sum(imp$MeanDecreaseGini),2)

# Generate plot titles by appending percentage to feature name
imp$Title <- paste0(imp$Feature," (",imp$Percentage,"%)")
imp$Title <- factor(imp$Title,levels=imp$Title)

# Merge with partial dependence table
data <- merge(data,imp,by="Feature")

# Plot top 20 features
idx <- which(data$Feature %in% imp$Feature[1:20])
ggplot(data[idx,]) + geom_line(aes(x=Val,y=Pred,color=Title)) + facet_wrap(~ Title,scales="free_x") + theme_bw() + theme(legend.position = "none") +
  xlab("LIWC Score") + ylab("Probability of Getting Signed by NBA Team") 

# Plot next 20 features
idx <- which(data$Feature %in% imp$Feature[21:40])
ggplot(data[idx,]) + geom_line(aes(x=Val,y=Pred,color=Title)) + facet_wrap(~ Title,scales="free_x") + theme_bw() + theme(legend.position = "none") +
  xlab("LIWC Score") + ylab("Probability of Getting Signed by NBA Team") 

# Plot next 20 features
idx <- which(data$Feature %in% imp$Feature[41:65])
ggplot(data[idx,]) + geom_line(aes(x=Val,y=Pred,color=Title)) + facet_wrap(~ Title,scales="free_x") + theme_bw() + theme(legend.position = "none") +
  xlab("LIWC Score") + ylab("Probability of Getting Signed by NBA Team") 


### Calculate simple correlations
lab <- as.numeric(train$LABEL) -1
sub <- train[,-c(1:6)]

d <- pblapply(1:ncol(sub),function(i){
  x <- cor(sub[,i],lab)
  output <- data.frame(Feature=names(sub)[i],Cor=x)
  return(output)
})
cor.df <- rbindlist(d)
cor.df <- cor.df[order(cor.df$Cor),]
write.csv(cor.df,"correlations.csv",row.names=F)

