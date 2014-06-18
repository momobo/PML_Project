# raw code project
getwd()
# install.packages("caret")

setwd("~\\..\\Google Drive\\Data Science\\08_PracticalMachineLearning\\PML_Project\\code\\raw")

library(caret)
library(ggplot2)
dat <-   read.csv("..\\..\\data\\pml-training.csv", na.strings= c("NA", "#DIV/0!"))
proof <- read.csv("..\\..\\data\\pml-testing.csv",  na.strings= c("NA", "#DIV/0!"))


# divide test and training
# Create a building data set and validation set
set.seed(7)
inBuild <- createDataPartition(y=dat$classe,  p=0.7, list=FALSE)
validation <- dat[-inBuild,]; 
buildData <- dat[inBuild,]

# Intern data test divide in train and test. For performance reason we keep
# the train set extremely small
set.seed(11)
small <- createDataPartition(y=buildData$classe,  p=0.1, list=FALSE)
train.small <- buildData[small,] 
testing     <- buildData[-small,]
dim(train.small)
dim(testing)

# exploration
str(buildData)
head(buildData)
# classe depend heavily on X
# qplot(X, user_name, col=classe, data=buildData)+geom_point(position = "jitter")
qplot(X, new_window, col=classe, data=buildData)+geom_point(position = "jitter")

# eliminate columns relative to the experiment
colElim <- c("X" , "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
             "new_window", "num_window", "cvtd_timestamp")
colConserv <- setdiff(names(train.small), colElim)
train.small <- subset(train.small, T, colConserv)

# ------------ real feature selection --------------------------------------
# pre process: eliminate the near zero variance variable
nzv <- nearZeroVar(train.small)
train.small <- train.small[, -nzv]
dim(train.small)

# save classe
classe <- train.small$classe
#train.small <- train.small[, -c("X")]

# eliminate non numeric 
sa <- sapply(train.small, is.numeric )
train.small <- train.small[,-which(sa==F)]

#  eliminate the non correlable
descrCorr <- cor(train.small)
train.small <- train.small[, -which(is.na(descrCorr))]
descrCorr <- cor(train.small)

#  eliminate the high correlate
highCorr <- findCorrelation(descrCorr, 0.9) 
train.small <- train.small[, -highCorr]

# set the predicted 
train.small$classe <- classe
dim(train.small)

# save features
names <- names(train.small)

# --------------------------------------------------------------
#head(train.small)
#
# random forest http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr
# It is computationally intensive, so we load it only if necessary.
RERUN=FALSE
if(RERUN){
    set.seed(13)
    modRf <- train(classe ~ . , data=train.small, trControl = ctrl, method="rf",  prox=T, number=5)
    save(modRf, file = "..\\data\\my_model.rda")
} else{
    load("..\\data\\my_model.rda")
}
# varImp(modRf) # importance

# predict
predRf <- predict(modRf,testing); 
matrixRf <- confusionMatrix(predRf, testing$classe)
# accuracy
matrixRf$overall[1]

#-----------------------------------------------------------------------------
# test with other models : Penalized Multinomial Regression
modLDA <- train(classe ~ . , data=train.small, method="PenalizedLDA")
predLDA <- predict(modLDA,testing);
matrixLDA <- confusionMatrix(predLDA, testing$classe)
# accuracy
matrixLDA$overall[1]
# 
# # nnet not work, kern either
# # pda: penalized discriminant analysis
modPda <- train(classe ~ . , data=train.small, method="pda")
predPda <- predict(modPda,testing);
matrixPda <- confusionMatrix(predPda, testing$classe)
# accuracy
matrixPda$overall[1]
#-----------------------------------------------------------------------------

# we choose RF
# retrain with more data 
set.seed(13)
getTrain   <- createDataPartition(y=buildData$classe,  p=0.3, list=FALSE)
train       <- buildData[getTrain,] 
testing     <- buildData[-getTrain,]

# filter the feature
train <- subset(train, T, names)

RERUN=FALSE
if(RERUN){
    set.seed(13)
    ctrl <- trainControl(method = "cv", repeats = 4)
    modRf_Fin <- train(classe ~ . , data=train, trControl = ctrl, method="rf",  prox=T, number=5)
    save(modRf_Fin, file = "..\\data\\my_model_final.rda")
} else{
    load("..\\data\\my_model_final.rda")
}

predRf <- predict(modRf_Fin,testing); 
matrixRf_Fin <- confusionMatrix(predRf, testing$classe)
matrixRf_Fin$overall[1]


# check our resulting model over the validation set
predRfVal <- predict(modRf_Fin,validation);
matrixRfVal <- confusionMatrix(predRfVal, validation$classe)
matrixRfVal$overall[1]

# confusion table:
matrixRfVal$table

###########################################################################
# check the problem 
intersect(names(proof), names(testing))
predRfProof <- predict(modRf, proof)
length(predRfProof)


# ------------------------ END ----------------------------
answers <- predRfProof
setwd("answers")
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

#  create a folder where you want the files to be written. 
# Set that to be your working directory and run:
    
pml_write_files(answers)



--------
?varImp
?confusionMatrix
















