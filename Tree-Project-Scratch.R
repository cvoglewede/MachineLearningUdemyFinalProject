

## Tree Project scratch work


library(ISLR)
library(tidyverse)
library(forcats)
library(caTools)
library(rpart)
library(rpart.plot)
df <- College
head(df)

df %>% 
  ggplot(aes(x=Grad.Rate,y=Room.Board,color=Private))+geom_point()


df %>% 
  ggplot(aes(x=F.Undergrad,fill=Private))+geom_histogram(bins = 50,color="black",position=position_stack(reverse=TRUE))

df %>% 
  ggplot(aes(x=Grad.Rate,fill=Private))+geom_histogram(bins = 50,color="black",position=position_stack(reverse=TRUE))


row.names(df[df$Grad.Rate>100,])

df["Cazenovia College","Grad.Rate"]
df["Cazenovia College","Grad.Rate"] <- 100
df["Cazenovia College","Grad.Rate"]


set.seed(103)
sample <- sample.split(df$Private, SplitRatio = 0.70)
train = subset(df, sample == TRUE)
test = subset(df, sample == FALSE)


tree_model <- rpart(Private ~ . , method='class', data= train)
printcp(tree_model)


label <- "Private"
test$pred_tree <- predict(tree_model, test[,-which(colnames(test)==label)])[,"Yes"]

head(test$pred_tree)

classification_point <- 0.5
test$predclass_tree <- ifelse(test$pred_tree>classification_point,'Yes','No')


table(test$Private,test$predclass_tree)

prp(tree_model,main="Tree Model")

library(randomForest)

rf_caret_model <- randomForest(Private ~ ., data=train, importance= TRUE)
model$confusion

model$importance
varImp(model)
varImpPlot(model,type=1)

test$pred_class_RF <- predict(model,newdata = test[,-which(colnames(test)==label)])

table(test$Private,test$pred_class_RF)


## Caret Section

library(caret)

names(getModelInfo())
modelLookup(model='RRF')


fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3)

mtry <- sqrt(ncol(train[,-which(colnames(test)==label)]))
grid <- expand.grid(.mtry=c(seq(2:15)))



# grid <- expand.grid(
#   n.trees=c(10,20,50,100,500,1000)
#   ,shrinkage=c(0.01,0.05,0.1,0.5)
#   ,n.minobsinnode = c(3,5,10)
#   ,interaction.depth=c(1,5,10))

?trainControl

rf_caret_model <- train(
  Private~.
  ,method = "rf"
  ,data=train          
  ,trControl=fitControl
  ,tuneGrid = grid
  , metric="Accuracy")

print(rf_caret_model)
plot(rf_caret_model,main="RF Model Tuning", ylab="Accuracy",xlab="parameter m, number of randomly selected predictors")
plot(varImp(rf_caret_model),main="Caret-tuned variable importance plot")
rf_caret_model$bestTune

test$pred_class_RF_tuned <- predict(rf_caret_model,test)
table(test$Private,test$pred_class_RF_tuned)


mean(test$Private==test$predclass_tree)
mean(test$Private==test$pred_class_RF)
mean(test$Private==test$pred_class_RF_tuned)



Models <- c("Tree","RandomForest","TunedRandomForest")
Model_Accuracies <- c(mean(test$Private==test$predclass_tree),
                      mean(test$Private==test$pred_class_RF),
                      mean(test$Private==test$pred_class_RF_tuned))
Model_comparison <- tibble(Models,Model_Accuracies)
Model_comparison<- Model_comparison %>% 
  arrange(desc(Model_Accuracies))
print(Model_comparison)







