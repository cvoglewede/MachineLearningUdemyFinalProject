
.libPaths( c( "~/userLibrary" ) )
install.packages('rpart')
install.packages('rpart.plot')
install.packages('randomForest')
install.packages('ISLR',repos = 'http://cran.us.r-project.org')
install.packages("class")
install.packages("datasets")
install.packages("cluster")
install.packages("corrplot")
install.packages("corrgram")
install.packages("Amelia")
install.packages("e1071")
install.packages('neuralnet',repos = 'http://cran.us.r-project.org')
install.packages("caret", dependencies = c("Depends", "Suggests"))
install.packages('skimr')

library(rpart)
library(rpart.plot)
library(randomForest)
library(ISLR)
library(class)
library(tidyverse)
library(datasets)
library(cluster)
library(corrplot)
library(caTools)
library(Amelia)
library(dplyr)
library(e1071)
library(MASS)
library(neuralnet)



# EDA

num.cols <- sapply(df, is.numeric)
cor.data <- cor(df[,num.cols])
cor.data
corrplot(cor.data,method='color')

missmap(df.train, main="Titanic Training Data - Missings Map", 
        col=c("yellow", "black"), legend=FALSE)



# Scale Continuous variables

standardized.Caravan <- scale(Caravan[,-86])


# Test/Train Splitting

sample <- sample.split(df$age, SplitRatio = 0.70) # SplitRatio = percent of sample==TRUE
train = subset(df, sample == TRUE)
test = subset(df, sample == FALSE)



# K means Clustering

set.seed(101)
irisCluster <- kmeans(iris[, 1:4], 3, nstart = 20)
irisCluster
table(irisCluster$cluster, iris$Species)
irisCluster
clusplot(iris, irisCluster$cluster, color=TRUE, shade=TRUE, labels=0,lines=0, )


# Linear Regression

df <- read.csv('student-mat.csv',sep=';')
model <- lm(G3 ~ .,train)
res <- residuals(model)
res <- as.data.frame(res)
plot(model)
G3.predictions <- predict(model,test)
results <- cbind(G3.predictions,test$G3) 
colnames(results) <- c('pred','real')
results <- as.data.frame(results)
to_zero <- function(x){
    if  (x < 0){
        return(0)
    }else{
        return(x)
    }
}
results$pred <- sapply(results$pred,to_zero)
mse <- mean((results$real-results$pred)^2)
print(mse)
mse^0.5
SSE = sum((results$pred - results$real)^2)
SST = sum( (mean(df$G3) - results$real)^2)

R2 = 1 - SSE/SST
R2


# Logistic Regression

df.train <- read.csv('titanic_train.csv')
impute_age <- function(age,class){
    out <- age
    for (i in 1:length(age)){
        
        if (is.na(age[i])){

            if (class[i] == 1){
                out[i] <- 37

            }else if (class[i] == 2){
                out[i] <- 29

            }else{
                out[i] <- 24
            }
        }else{
            out[i]<-age[i]
        }
    }
    return(out)
}
fixed.ages <- impute_age(df.train$Age,df.train$Pclass)
df.train$Age <- fixed.ages
missmap(df.train, main="Titanic Training Data - Missings Map", 
        col=c("yellow", "black"), legend=FALSE)
df.train <- select(df.train,-PassengerId,-Name,-Ticket,-Cabin)
log.model <- glm(formula=Survived ~ . , family = binomial(link='logit'),data = df.train)
summary(log.model)
set.seed(101)
split = sample.split(df.train$Survived, SplitRatio = 0.70)
final.train = subset(df.train, split == TRUE)
final.test = subset(df.train, split == FALSE)
final.log.model <- glm(formula=Survived ~ . , family = binomial(link='logit'),data = final.train)
summary(final.log.model)
fitted.probabilities <- predict(final.log.model,newdata=final.test,type='response')
fitted.results <- ifelse(fitted.probabilities > 0.5,1,0)
misClasificError <- mean(fitted.results != final.test$Survived)
print(paste('Accuracy',1-misClasificError))
table(final.test$Survived, fitted.probabilities > 0.5)



# KNN 

purchase <- Caravan[,86]
test.index <- 1:1000
test.data <- standardized.Caravan[test.index,]
test.purchase <- purchase[test.index]
train.data <- standardized.Caravan[-test.index,]
train.purchase <- purchase[-test.index]
set.seed(101)
predicted.purchase <- knn(train.data,test.data,train.purchase,k=1)
head(predicted.purchase)
mean(test.purchase != predicted.purchase)
predicted.purchase <- knn(train.data,test.data,train.purchase,k=3)
mean(test.purchase != predicted.purchase)
predicted.purchase <- knn(train.data,test.data,train.purchase,k=5)
mean(test.purchase != predicted.purchase)
predicted.purchase = NULL
error.rate = NULL

for(i in 1:20){
    set.seed(101)
    predicted.purchase = knn(train.data,test.data,train.purchase,k=i)
    error.rate[i] = mean(test.purchase != predicted.purchase)
}
k.values <- 1:20
error.df <- data.frame(error.rate,k.values)

ggplot(error.df,aes(x=k.values,y=error.rate)) + geom_point()+ geom_line(lty="dotted",color='red')


# Tree-based Model

tree <- rpart(Kyphosis ~ . , method='class', data= kyphosis)
printcp(tree)
plot(tree, uniform=TRUE, main="Main Title")
text(tree, use.n=TRUE, all=TRUE)
prp(tree)


# Random Forest

model <- randomForest(Kyphosis ~ .,   data=kyphosis)
print(model) # view results
importance(model) # importance of each predictor


# SVM

model <- svm(Species ~ ., data=iris)
summary(model)
predicted.values <- predict(model,iris[1:4])
table(predicted.values,iris[,5])
# Tune for combos of gamma 0.5,1,2
# and costs 1/10 , 10 , 100
tune.results <- tune(svm,train.x=iris[1:4],train.y=iris[,5],kernel='radial',
                  ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
summary(tune.results)
tuned.svm <- svm(Species ~ ., data=iris, kernel="radial", cost=1, gamma=0.5)
summary(tuned.svm)
tuned.predicted.values <- predict(tuned.svm,iris[1:4])
table(tuned.predicted.values,iris[,5])


# Neural Net

set.seed(101)
data <- Boston
maxs <- apply(data, 2, max) 
mins <- apply(data, 2, min)
scaled <- as.data.frame(scale(data, center = mins, scale = maxs - mins))
split = sample.split(scaled$medv, SplitRatio = 0.70)

train = subset(scaled, split == TRUE)
test = subset(scaled, split == FALSE)
n <- names(train)
f <- as.formula(paste("medv ~", paste(n[!n %in% "medv"], collapse = " + ")))
nn <- neuralnet(f,data=train,hidden=c(5,3),linear.output=TRUE)
plot(nn)
predicted.nn.values <- compute(nn,test[1:13])
true.predictions <- predicted.nn.values$net.result*(max(data$medv)-min(data$medv))+min(data$medv)
test.r <- (test$medv)*(max(data$medv)-min(data$medv))+min(data$medv)
MSE.nn <- sum((test.r - true.predictions)^2)/nrow(test)
MSE.nn
error.df <- data.frame(test.r,true.predictions)
ggplot(error.df,aes(x=test.r,y=true.predictions)) + geom_point() + stat_smooth()

