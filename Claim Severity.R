#####################################################################################
### This script is broken into 5 parts and provides example code to:
#   1) Read the libraries and dataset
#   2) Create some simple features which can be used in a predictive model
#   3) Recreate train and test now that features have been created on both
#   4) Build a simple GBM on a random 70% of train, validate on the other 30% and calculate the quadratic weighted kappa
#   5) Score the test data and create a submission file
##############################################
###################################################################################



###############################################################################
#Step 1: Read in the data and define variables as either a factor or numeric
#########################################################################################


#### ALLSTATE Claim Severity Challenge #######
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
#if(!require(tidyr)) install.packages("tidyr", repos = "http://cran.us.r-project.org")
#if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
#if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(forecast)) install.packages("forecast", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(PCAmixdata)) install.packages("PCAmixdata", repos = "http://cran.us.r-project.org")
#if(!require(FactoMineR)) install.packages("FactoMineR", repos = "http://cran.us.r-project.org")
#if(!require(factoextra)) install.packages("factoextra", repos = "http://cran.us.r-project.org")
#if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(Matrix)) install.packages("Matrix", repos = "http://cran.us.r-project.org")
#if(!require(methods)) install.packages("methods", repos = "http://cran.us.r-project.org")
if(!require(mlbench)) install.packages("mlbench", repos = "http://cran.us.r-project.org")
#if(!require(scales)) install.packages("scales", repos = "http://cran.us.r-project.org")
#if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")

#library(data.table)
#library(mlr3)



# Read the CSV file and Load the data:
raw_train <- read.csv("https://raw.githubusercontent.com/jayeshjohn/Claim-Severity/master/Claim%20Severity.csv")

#Check the dataset dimensions
dim(raw_train)

# remove ID column from training set as it is not needed for training
raw_train$id <- NULL

sd(raw_train$loss)
mean(raw_train$loss)
median(raw_train$loss)
range(raw_train$loss)

# Plot Loss
plot(x = 1:nrow(raw_train), y = raw_train$loss, type = "h", main = "Loss Plot", xlab = "Row Number", ylab = "Loss")

## Split the dataset having Categorical and Continuous fields
cont_train <- raw_train[, -grep("^cat", colnames(raw_train))]
cat_train <- raw_train[,-grep("^cont",colnames(raw_train))]
str(cont_train)
str(cat_train)


cont_train %>% 
  gather(-loss, key = "key", value = "value") %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free", ncol = 3) +
  geom_density()


cont_train %>% 
  gather(-loss, key = "var", value = "value") %>% 
  ggplot(aes(x = value, y = log(loss))) +
  geom_point() +
  geom_smooth(method = lm) +
  facet_wrap(~ var, scales = "free", ncol = 3)



# Combining Train and Test datasets
#all_data = rbind(Train[,1:131], Test)
#Train <- raw_train
#all_data <- raw_train[,1:130]
#head(all_data)
### Data Pre-Processing
## Treating Continous variables

# Cont fields, plot the correlation
corr <- cor(cont_train, method = c('pearson'))
corrplot(corr, method = "circle", order = "hclust", type = "lower", diag = FALSE)


#Running correlation on continuous variables 
#corr = cor(cont_train, method = c('pearson'))
corr.df = as.data.frame(as.table(corr))
subset(corr.df[order(-abs(corr.df$Freq)),], abs(Freq) > 0.8)

#Removing Varibles with correlations greater than 80% - cont 9, cont 10, cont12, cont13
cont_vars_all = cont_train[,-c(9,10,12,13)]

# Convert Categorical Character fields into Factors
cat_train <- cat_train %>% mutate_if(is.character,as.factor)

# Get the length of cat data levels
cat_data_levels <- sapply(seq(1, ncol(cat_train)), function(d){
  length(levels(cat_train[,d]))
})

# Plot the cat data levels
plot(cat_data_levels, type = "l", col = "blue")

## We can split the Categorical columns into three:
#  1. Binary Columns with 2 Levels - Columns 1:72
#  2. Columns with 3-9 Levels - Columns 73:98
#  3. Columns with over 9 Levels - Columns 99:116


## Treating Binary variables (PCA) 

#extracting the binary variables from all_data
binary_all <- cat_train[,1:72]

# Converting the data to numeric 
binaryNum_all <- binary_all %>% mutate_if(is.factor,as.numeric)

#performing PCA on the numeric matrix of binary variables 
pca_all <- princomp(binaryNum_all, cor = TRUE, scores = TRUE)
#pca_all and choose 25 variables
summary(pca_all)
Binary_vars_all = as.data.frame(pca_all$scores[,1:25])

## Treating the 3-9 level categorical variables (Categorical PCA)

#extracting the 3-9 level categorical variables from the whole dataset
#categorical_all = cat_train[73:116]

#creating two categories 
#cat1_all = categorical_all[ ,1:16]
#cat2_all = categorical_all[ ,c(17:26, 30)]
cat1_all = cat_train[ ,73:88]
cat2_all = cat_train[ ,c(89:98, 30)]

#running pca on both categories separately 
pca_cat1_all = PCAmix(X.quali = cat1_all, ndim = 30, rename.level = TRUE)
pca_cat2_all = PCAmix(X.quali = cat2_all, ndim = 50, rename.level = TRUE)
summary(pca_cat1_all)


#extracting the scores to create a dataframe of the new variables from the step above
cat1_vars_all = data.frame(pca_cat1_all$scores)
cat2_vars_all = data.frame(pca_cat2_all$scores)
#head(cat2_vars_all)

#changing the column names so as to not mix with cat1 variables 
colnames(cat2_vars_all) <- paste("cat", colnames(cat2_vars_all), sep = "_")
colnames(cat2_vars_all)

## Treating the above 9 level categorical variables (converting into numeric) 

#extracting all more than 9 level variables and converting them into numeric values 
#high_level_vars_all = categorical_all[,27:44]
high_level_vars_all = cat_train[,99:116]

#high_level_vars_numeric_all <-  as.data.frame(sapply(high_level_vars_all, as.numeric))
str(high_level_vars_all)
high_level_vars_all <- high_level_vars_all %>% mutate_if(is.character,as.factor)
high_level_vars_numeric_all <- high_level_vars_all %>% mutate_if(is.factor,as.numeric)
str(high_level_vars_numeric_all)

#target <- cat_train[,117]
#final dataset
final.dataset.all <- cbind(Binary_vars_all, cat1_vars_all, cat2_vars_all, high_level_vars_numeric_all, cont_vars_all)
head(final.dataset.all)

#Train_data = final.dataset.all
#setDT(Train_data)
#class(Train_data)

#adding the target variable to the training dataset
#target = Train$loss
#target = as.data.frame(target)
#Train_data = as.data.table(Train_data)
#Train_data = cbind(Train_data, target)
#head(Train_data)

#splitting training into 2- train and valid
set.seed(505)
train_index <- createDataPartition(final.dataset.all$loss, p = .7, list = FALSE, times =1)
tr <- final.dataset.all[train_index,]
head(tr)
ts <- final.dataset.all[-train_index,]
ncol(ts)

#subsetting the columns we need
#tr = tr[,1:134]
#ts = ts[,1:134]
#class(tr)
#head(tr)
tr <- data.frame(tr)
############ 

mean(tr$loss)


######################################################
### MODEL 1: Single Value Mean
######################################################

simple_mean <- mean(raw_train$loss)
pred_var <- ts$loss
pred_var[] <- simple_mean

#final_RMSE <- RMSE(simple_mean, obs = ts$loss)
#final_RMSE
Avgmodel_Accuracy <- accuracy(pred_var, ts$loss)
df1 <- as.data.frame(Avgmodel_Accuracy)
avgmodel_mae <- df1$MAE

# add mae results in a table 
mae_results <- data.frame(Method = "Single Value Mean", MAE = avgmodel_mae) 
mae_results %>% knitr::kable() 


######################################################
### MODEL 2: Single Value Mean - Best Cutoff
######################################################

# function to compute best possible Mean Value #
n <- seq(1000,4000,1)

exact_prob <- function(n){
  temp_var[] <- n   # vector of fractions for mult. rule
  var_accuracy <- accuracy(temp_var, ts$loss)
  df1 <- as.data.frame(var_accuracy)
  df1$MAE
}

# applying function element-wise to vector of n values
eprob <- sapply(n, exact_prob)
best_mae_cutoff <- min(eprob)

best_mean_cutoff <- n[which.min(eprob)]
best_mean_cutoff

#temp_var <- ts$loss
#temp_var[] <- best_cutoff
#Best_avgmodel_Accuracy <- accuracy(temp_var, ts$loss)
#Best_avgmodel_Accuracy

mae_results <- bind_rows(mae_results, 
                          data_frame(Method="Best Mean Cutoff",   
                                     MAE = best_mae_cutoff)) 
mae_results %>% knitr::kable() 


######################################################
### MODEL 3: Linear Regression with PCA data
######################################################


fit <- lm(loss ~ ., data = tr)
valid_pred <- predict(fit, ts)


## Measures
final_RMSE <- RMSE(pred = valid_pred, obs = ts$loss)
final_R2 <-R2(pred = valid_pred, obs = ts$loss)
final_RMSE

LinearReg_PCA_Accuracy = accuracy(valid_pred, ts$loss)
df2 <- as.data.frame(LinearReg_PCA_Accuracy)
linear_regression_mae <- df2$MAE

mae_results <- bind_rows(mae_results, 
                         data_frame(Method="Linear Regression Model",   
                                    MAE = linear_regression_mae)) 
mae_results %>% knitr::kable() 


######################################################
### MODEL 4: xgBoost Model
######################################################

#### Model preparation ####

#extracting the loss variable from the training set 
tr_label = tr$loss
ts_label = ts$loss

#plotting the histogram to check the distribution
hist(tr_label)
hist(ts_label)

#Part of feature engineering- transforming the loss variable to obtain normal-like distribution
tr_label_log = log(tr$loss + 200)
ts_label_log = log(ts$loss + 200)

#plotting the histogram to check the new distribution
hist(tr_label_log)
hist(ts_label_log)

#converting the train and test dataframes to a matrix
tr_matrix = as.matrix(tr, rownames.force = NA)
ts_matrix = as.matrix(ts, rownames.force = NA)

#converting the train and test dataframes to a sparse matrix
tr_sparse = as(tr_matrix, "sparseMatrix")
ts_sparse = as(ts_matrix, "sparseMatrix")

#For xgboost, using xgb.DMatrix to convert data table into a matrix is most recommended
dtrain = xgb.DMatrix(data = tr_sparse[,1:133], label = tr_label_log )
dtest = xgb.DMatrix(data = ts_sparse[,1:133], label = ts_label_log )
class(dtrain)

## preparation for xgboost model 

#defining default parameters
#params <- list(booster = "gbtree", objective = "reg:linear", eta=0.1, gamma=0, 
#               max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1, lambda=0, alpha=1)
params <- list(booster = "gbtree", objective = "reg:linear", eta=0.1, gamma=0, nthread = 8,
               max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1, 
               early_stopping_rounds = 25, eval_metric = "mae", lambda=0, 
               prediction = TRUE, alpha=1)

#Using the inbuilt xgb.cv function to calculate the best nround for this model. 
#In addition, this function also returns CV error, which is an estimate of test error.

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, 
                 showsd = T, stratified = T, print_every_n = 10, maximize = F)


# finding the best nrounds value
xgbcv
xgbcv$best_iteration


#training the model on default parameters with nrounds = 100 
xgb1 <- xgb.train (params = params, 
                   data = dtrain, 
                   nrounds = 100, 
                   watchlist = list(val=dtest,train=dtrain), 
                   print_every_n = 10, 
                   maximize = F)
#                   eval_metric = "mae")


#model prediction
xgbpred = predict(xgb1,dtest)

#taking the antilog of the predictions and subtracting 200 to get the error value in terms of the original loss variable
preds = exp(xgbpred)-200

#checking the accuracy of the the model using MAE
xBoost_Accuracy = accuracy(preds, ts_label)
df3 <- as.data.frame(xBoost_Accuracy)
xBoost_mae <- df3$MAE

mae_results <- bind_rows(mae_results, 
                         data_frame(Method="xBoost Model",   
                                    MAE = xBoost_mae)) 
mae_results %>% knitr::kable() 

feat_importance <- varImp(xgb1)

imp_DF <- data.frame(features = row.names(feat_importance[[1]]),
                     importance_val =  round(feat_importance[[1]]$Overall, 2)
                     
) 

imp_DF <- arrange(imp_DF, desc(importance_val))

#Plot the top 10 features with their importances
ggplot(head(imp_DF, 20), aes(x = reorder(features, importance_val), y = importance_val)) +
  geom_bar(stat = "identity", fill = 'tan4') + coord_flip()

