# Machine-Learning

library(data.table)
library(Matrix)
library(xgboost)
library(caret)
library(dplyr)
library(tidyverse)
library(mlr)      
library(knitr)
library("lattice")


## Titanic

train  <- fread('C:/Kaggle/train.csv', sep=",", na.strings = "NA")

test   <- fread('C:/Kaggle/test.csv' , sep=",", na.strings = "NA")


train <- train_orig %>%  mutate(dataset = "train")

test <- test_orig  %>%  mutate(dataset = "test")

combined <- bind_rows(train, test)


summarizeColumns(combined) %>%
  kable(digits = 2)
  
  
  combined <- combined %>%
  mutate_at(
    .vars = vars("Survived", "Pclass", "Sex", "Embarked"),
    .funs = funs(as.factor(.))
  )
  
  imp <- impute(
  combined,
  classes = list(
    factor = imputeMode(),
    integer = imputeMean(),
    numeric = imputeMean()
  )
)


summarizeColumns(combined) %>%
  kable(digits = 2)
  
  
combined <- normalizeFeatures(combined, target = "Survived")



summarizeColumns(combined) %>%
  kable(digits = 2)
  
  
  combined <- createDummyFeatures(
  combined, target = "Survived",
  cols = c(
    "Pclass",
    "Sex",
    "Embarked"
  )
  
  
  
summarizeColumns(combined) %>%
  kable(digits = 2)
  
  
  
train <- combined %>%
  filter(dataset == "train") %>%
  select(-dataset)
  
  
  
test <- combined %>%
  filter(dataset == "test") %>%
  select(-dataset)
  
  
  
  
train$Name   <- NULL
train$Ticket <- NULL
train$Cabin <- NULL


test$Ticket <- NULL
test$Name   <- NULL
test$Cabin  <- NULL
test$Survived  <- NULL


trainTask <- makeClassifTask(data = train, target = "Survived", positive = 1)
testTask <- makeClassifTask(data = test, target = "Survived")



getParamSet("classif.xgboost")



xgb_learner <- makeLearner(
  "classif.xgboost",
  predict.type = "response",
  par.vals = list(
    objective = "binary:logistic"))
    
    
    
    
  xgb_params <- makeParamSet(
    # The number of trees in the model (each one built sequentially)
    makeIntegerParam("nrounds", lower = 50, upper = 200),
    # number of splits in each tree
    makeIntegerParam("max_depth", lower = 1, upper = 10),
    # "shrinkage" - prevents overfitting
    makeNumericParam("eta", lower = .1, upper = .5),
    # L2 regularization - prevents overfitting
    makeNumericParam("lambda", lower = -1, upper = 0, trafo = function(x) 10^x)
  )



control <- makeTuneControlRandom(maxit = 100)


resample_desc <- makeResampleDesc("CV", iters = 5)



tuned_params <- tuneParams(
  learner = xgb_learner,
  task = trainTask,
  resampling = resample_desc,
  par.set = xgb_params,
  control = control,
  measures = acc
)

## Best Parameters
head(tuned_params)


xgb_tuned_learner <- setHyperPars(
  learner = xgb_learner,
  par.vals = tuned_params$x
)

xgb_model <- train(xgb_tuned_learner, trainTask)

getFeatureImportance(xgb_model)

plot(getFeatureImportance(xgb_model))


result <- predict(xgb_model, newdata = test)


prediction <- result$data %>%
  select(PassengerID = id, Survived = response) %>%
  # Put back the original passenger IDs. No sorting has happened, so
  # everything still matches up.
  mutate(PassengerID = test_orig$PassengerId)
  
  write_csv(prediction, "final_prediction.csv")


  
  
)
