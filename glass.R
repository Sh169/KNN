install.packages('caTools')  #for train and test data split
install.packages('dplyr')    #for Data Manipulation
install.packages('ggplot2')  #for Data Visualization
install.packages('class')    #KNN 
install.packages('caret')    #Confusion Matrix
install.packages('corrplot') #Correlation Plot
install.packages("heuristica")
library(heuristica)
library(caTools)
library(dplyr)
library(ggplot2)
library(caret)
library(class)
library(corrplot)

glass<-read.csv("F:/Assignments/KNN/glass.csv",
                  col.names=c("RI","Na","Mg","Al","Si","K","Ca","Ba","Fe","Type"))
#Standardise the data
standard.features <- scale(glass[,0:9])

#Join the standardized data with the target column
data<- cbind(standard.features,glass[10])

#Check if there are any missing values to impute. 
anyNA(data)

head(data)

#Visualize the data
corrplot(cor(data))

set.seed(101)

#Train and Test Data
sample <- sample.split(data$Type,SplitRatio = 0.70)
train <- subset(data,sample==TRUE)
test <- subset(data,sample==FALSE)

#KNN Model
predicted.type <- knn(train[0:9],test[0:9],train$Type,k=1)
#Error in prediction
error <- mean(predicted.type!=test$Type)
#Confusion Matrix
confusionMatrix(table(predicted.type,test$Type))
#Accuracy achieved is 63%


predicted.type <- NULL
error.rate <- NULL

for (i in 1:10) {
  predicted.type <- knn(train[0:9],test[0:9],train$Type,k=i)
  error.rate[i] <- mean(predicted.type!=test$Type)
  
}

#Choosing K Value by Visualization
knn.error <- as.data.frame(cbind(k=1:10,error.type =error.rate))


ggplot(knn.error,aes(k,error.type))+ 
  geom_point()+ 
  geom_line() + 
  scale_x_continuous(breaks=1:10)+ 
  theme_bw() +
  xlab("Value of K") +
  ylab('Error')

#When k=3 accuracy achieved is 65%
predictedtype<-knn(train[0:9],test[0:9],train$Type,k=5)
#Error in prediction
error<-mean(predictedtype!=test$Type)
#ConfusionMatrix
confusionMatrix(table(predictedtype,test$Type))



