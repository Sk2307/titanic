# Importing the dataset
train <- read.csv('train.csv' , header = T , na.strings = c(""," ","NA"))
test <- read.csv('test.csv' , header = T , na.strings = c(""," ","NA"))

test$Survived <- NA
train$Set <- 'train'
test$Set <- 'test'

full <- rbind(train , test)

# Data PreProcessing
library(tidyverse)
head(train)
dim(train)
summary(train)
str(train)

full$Pclass <- as.factor(full$Pclass)
full <- full[ , 2:13]
full$Survived <- as.factor(full$Survived)
str(full)

# Missing Values
summary_na <- function(df) {
  count_na <- sapply(df , function(col) {
    sum(is.na(col))
  })
  return(data.frame(count_na , percent_na = (count_na / nrow(df)) * 100))
}

summary_na(full)

full$Embarked[which(is.na(full$Embarked))] <- full$Embarked[which.max(full$Embarked)]
summary_na(full)

full$Fare[which(is.na(full$Fare))] <- mean(full$Fare , na.rm = T)
summary_na(full)

full$Title <- (gsub('(.*, )|(\\..*)', '', full$Name))

table(full$Title)
table(full$Sex , full$Title)

title_mr <- c('Capt','Col','Don','Dr','Jonkheer','Major','Rev','Sir')
title_miss <- c('Mlle','Ms')
title_mrs <- c('Dona','Lady','Mme','the Countess')

full$Title[which(full$Title %in% title_mr)] = 'Mr'
full$Title[which(full$Title %in% title_miss)] = 'Miss'
full$Title[which(full$Title %in% title_mrs)] = 'Mrs'

str(full)
full$Title <- as.factor(full$Title)


full$FamilySize <- full$SibSp + full$Parch + 1

full$Family_Size[full$FamilySize == 1]   <- 'Single'
full$Family_Size[full$FamilySize < 5 & full$FamilySize >= 2]   <- 'Small'
full$Family_Size[full$FamilySize >= 5]   <- 'Large'

full$Family_Size=as.factor(full$Family_Size)



library(rpart)
age_tree <- rpart(Age ~ Pclass+SibSp+Parch+Fare+Embarked+Title , data = full)
rpart.plot::rpart.plot(age_tree)

full$pred_age <- predict(age_tree,full)
ggplot(data = full , aes(x = Age , y = pred_age)) +
  geom_point() +
  geom_smooth()

full$Age[which(is.na(full$Age))] = full$pred_age

full <- full[,-c(3,10,14)]

#Data Wrangling
#1. Rich Class vs Poor Class
ggplot(data = full[1:891, ] , aes(x = Pclass ,fill= Survived)) +
  geom_bar()+
  labs(x = 'Pclass' , y = 'Count' , fill = 'Survival')+
  ggtitle('Survival Rate : Rich vs Poor')

#2. Male vs Female in each Class
ggplot(data = full[1:891,] , aes(x = Sex , fill = Survived)) +
  geom_bar()+
  facet_wrap(~Pclass) +
  labs(x = 'Sex' , y = 'Count' , fill = 'Survived') +
  ggtitle('Survival Rate : Male vs Female')

#3. Title vs Survival Rate
ggplot(data = full[1:891,] , aes(x = Title , fill = Survived)) +
  geom_bar() +
  labs(x = 'Title' , y = 'Count' , fill = 'Survived')+
  ggtitle('Survival Rate : Titles')

#4. Family Size and Survival Rate
ggplot(data = full[1:891,] , aes(x = Family_Size , fill = Survived)) +
  geom_bar() +
  labs(x = 'Family Size' , y = 'Count' , fill = 'Survived') +
  ggtitle('Survival Rate : Family Size')

#5. Emabarked vs Survival Rate
ggplot(data = full[1:891,] , aes(x = Embarked , fill = Survived)) +
  geom_bar() +
  labs(x = 'Embarked' , y = 'Count' , fill = 'Survived') +
  ggtitle('Survival Rate : Embarked')

# Making Predictions
train <- full[1:891,]
test <- full[892:1309,]
train <- train[,-c(7,10,12)]
test <- test[,-c(1,7,10,12)]
str(train)
train$Survived <- as.numeric(train$Survived)

library(caTools)
set.seed(123)
split = sample.split(train$Survived, SplitRatio = 2/3)
training_set = subset(train, split == TRUE)
test_set = subset(train, split == FALSE)

# SVM Regressor(Linear)
set.seed(123)
regressor_svm_l= svm(formula = Survived~. , data = training_set1 ,
                     type = 'eps-regression' , kernel = 'linear')

regressor_svm_l
y_pred_svml <- predict(regressor_svm_l , newdata = test_set )
cm <- table(test_set[,1] , y_pred_svml)
cm <- confusionMatrix(y_pred_svml , test_set$Survived)
# SVM Regressor(Radial)
library(e1071)
set.seed(123)
regressor_svm = svm(formula = Survived ~ . , data = training_set , scale = T,
                    type = 'eps-regression' , kernel = 'radial') 
y_pred_svm <- predict(regressor_svm , newdata = test_set)

summary(regressor_svm)
library(caret)
cm <- confusionMatrix(y_pred_svm , test_set$Survived)

# Decision Tree
library(rpart)
library(rpart.plot)
regressor_dt <- rpart(formula = Survived~., data = training_set ,
                      control = rpart.control(minsplit = 1) , method = 'class')
summary(regressor_dt)
rpart.plot(regressor_dt , extra = 3 , fallen.leaves = T)
y_pred_dt <- predict(regressor_dt , newdata = test_set , type = "class")
cm_dt <- table(test_set[,1] , y_pred_dt)

# Random Forest
library(randomForest)
set.seed(1234)

regressor_rf <- randomForest(x = training_set[,-1] ,
                             y = training_set[,1] , ntree = 500 ,
                             importance = T)
regressor_rf
y_pred_rf <- predict(regressor_rf , newdata = test_set)
postResample(y_pred_rf , test_set[,1])

varImpPlot(regressor_rf)
training_set1 <- training_set
training_set1 <- training_set1[,-c(5,6,8)]

regressor_rf1 <- randomForest(x = training_set1[,-1] ,
                             y = training_set1[,1] , ntree = 500)
regressor_rf1
y_pred_rf1 <- predict(regressor_rf1 , newdata = test_set)
postResample(y_pred_rf1 , test_set[,1])