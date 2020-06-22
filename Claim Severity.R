############################# ALLSTATE Claim Severity ###############################
#####################################################################################
### This script is divided into 5 parts:
#   1) Load the libraries and read the CSV dataset
#   2) Visualization on dataset features - Categorical and Continuous Columns
#   3) Apply PCA transformation based on column levels
#   4) Create Train and Test partitions on PCA transformed and Raw dataset
#   5) Prediction Modeling and compare MAE for each of the Models
#####################################################################################

#####################################################################################
# Step 1: Load the libraries and read the CSV dataset
#####################################################################################

# Note: The below scripts have been coded and executed with R 4.0 version

# The Library loading/installation process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(forecast)) install.packages("forecast", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(PCAmixdata)) install.packages("PCAmixdata", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(viridis)) install.packages("viridis", repos = "http://cran.us.r-project.org")


# Read the CSV file and Load the data:
raw_data <- read.csv("https://raw.githubusercontent.com/jayeshjohn/Claim-Severity/master/Claim%20Severity.csv")

# Check the dataset dimensions and Loss SD, Mean, Range
dim(raw_data)
sd(raw_data$loss)
mean(raw_data$loss)
range(raw_data$loss)

# remove ID column from training set as it is not needed for training
raw_data$id <- NULL

# see 'Claim Severity' dataset in tidy format
raw_data %>% tibble()

#####################################################################################
# Step 2: Visualization on dataset features - Categorical and Continuous fields
#####################################################################################

# Plot Loss
plot(x = 1:nrow(raw_data), y = raw_data$loss, type = "h", main = "Loss Plot", 
     xlab = "Row Number", ylab = "Loss", col = viridis(nrow(raw_data)))

## Split the dataset having Categorical and Continuous fields
cont_fields <- raw_data[, -grep("^cat", colnames(raw_data))]
cat_fields <- raw_data[,-grep("^cont",colnames(raw_data))]

# Continuous variable Density distribution 
cont_fields %>% 
  gather(-loss, key = "key", value = "value") %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free", ncol = 3) +
  geom_density(aes(fill = "loss")) +
  scale_color_viridis(discrete = TRUE, option = "D")+
  scale_fill_viridis(discrete = TRUE) +
  theme_minimal()

# Continuous Variable Distribution with Loss
# To reduce the plot size we use the 10000 rows
cont_fields_plot <- cont_fields[1:10000,]
cont_fields_plot %>% 
  gather(-loss, key = "var", value = "value") %>% 
  ggplot(aes(x = value, y = log(loss))) +
  geom_point(size=1, alpha=0.3) +
  geom_smooth(method = lm) +
  theme(legend.position = "bottom") +
  facet_wrap(~ var, scales = "free", ncol = 3)

### Data Pre-Processing

## Treating Continous variables
# Cont fields, plot the correlation
corr <- cor(cont_fields, method = c('pearson'))
corrplot(corr, method = "circle", order = "hclust", type = "lower", diag = FALSE)

# Running correlation on continuous variables 
corr.df <- as.data.frame(as.table(corr))
subset(corr.df[order(-abs(corr.df$Freq)),], (abs(Freq) > 0.8 & abs(Freq) < 1))

# Removing Variables with correlations greater than 80% - cont 9, cont 10, cont12, cont13
cont_vars_all <- cont_fields[,-c(9,10,12,13)]

#####################################################################################
# Step 3: Apply Principal Component Analysis (PCA) transformation based on levels
#####################################################################################

# Convert Categorical Character fields into Factors
cat_fields <- cat_fields %>% mutate_if(is.character,as.factor)

# Get the length of cat data levels
cat_data_levels <- sapply(seq(1, ncol(cat_fields)), function(d){
  length(levels(cat_fields[,d]))
})

# Plot the cat data levels
barplot(cat_data_levels, type = "l", ylab="Cat Column Levels", 
        xlab = "Cat Column Index", col = viridis(117))

##############################################################################
## We split the Categorical columns into three:
#  1. Binary Columns with 2 Levels - Columns 1:72
#  2. Columns with 3-9 Levels - Columns 73:98
#  3. Columns with over 9 Levels - Columns 99:116
#############################################################################

## 1. PCA on Categorical Variables Binary: 2 Levels

# Extracting the binary variables from all_data
binary_all <- cat_fields[,1:72]

# Converting the variables to Numeric 
binaryNum_all <- binary_all %>% mutate_if(is.factor,as.numeric)

# Performing PCA on the numeric matrix of binary variables 
pca_all <- princomp(binaryNum_all, cor = TRUE, scores = TRUE)

# Choose 25 PCA Scores
summary(pca_all)
Binary_vars_all <- as.data.frame(pca_all$scores[,1:25])

## 2. PCA on Categorical Variables: 3-9 Levels

# Extracting the 3-9 level categorical variables from the whole dataset

# Creating two categories 
cat1_all <- cat_fields[ ,73:88]
#cat2_all <- cat_fields[ ,c(89:98, 30)]
cat2_all <- cat_fields[ ,89:98]


# Running pca on both categories separately 
#pca_cat1_all <- PCAmix(X.quali = cat1_all, ndim = 30, rename.level = TRUE)
#pca_cat2_all <- PCAmix(X.quali = cat2_all, ndim = 50, rename.level = TRUE)
pca_cat1_all <- PCAmix(X.quali = cat1_all, ndim = 15, rename.level = TRUE, graph = FALSE)
pca_cat2_all <- PCAmix(X.quali = cat2_all, ndim = 10, rename.level = TRUE)

# Extracting the scores to create a dataframe of the new variables from the step above
cat1_vars_all <- data.frame(pca_cat1_all$scores)
cat2_vars_all <- data.frame(pca_cat2_all$scores)

# Changing the column names so as to not mix with cat1 variables 
colnames(cat2_vars_all) <- paste("cat", colnames(cat2_vars_all), sep = "_")
colnames(cat2_vars_all)

## 3. PCA on Categorical Variables: >9 Levels

#extracting all more than 9 level variables and converting them into numeric values 
high_level_vars_all <- cat_fields[,99:116]

# Converting variables to Numeric
high_level_vars_all <- high_level_vars_all %>% mutate_if(is.character,as.factor)
high_level_vars_numeric_all <- high_level_vars_all %>% mutate_if(is.factor,as.numeric)

# final dataset
final.dataset.all <- cbind(Binary_vars_all, cat1_vars_all, cat2_vars_all, high_level_vars_numeric_all, cont_vars_all)
ncol(final.dataset.all)

# We see that the data features are transformed into 79 Rows including Loss

#####################################################################################
# Step 4: Create Train and Test partitions on PCA transformed and Raw dataset
#####################################################################################

set.seed(500)

# Raw Data - splitting datasets into 2 - train and test
raw_data_fac <- raw_data %>% mutate_if(is.character,as.factor)
raw_data_num <- raw_data_fac %>% mutate_if(is.factor,as.numeric)

raw_train_index <- createDataPartition(raw_data_num$loss, p = .7, list = FALSE, times =1)
raw_tr <- raw_data_num[raw_train_index,]
raw_ts <- raw_data_num[-raw_train_index,]

# PCA Data - splitting training into 2 - train and test
train_index <- createDataPartition(final.dataset.all$loss, p = .7, list = FALSE, times =1)
tr <- final.dataset.all[train_index,]
ts <- final.dataset.all[-train_index,]

# PCA Minimal Cols
#pca_train_index <- createDataPartition(data_pca_all$loss, p = .7, list = FALSE, times =1)
#pca_tr <- data_pca_all[pca_train_index,]
#pca_ts <- data_pca_all[-pca_train_index,]

#####################################################################################
# Step 5: Prediction Modeling and compare MAE for each of the Models
#####################################################################################

######################################################
### MODEL 1: The Average Model
######################################################

# Naive Model that takes Mean of Train dataset as the loss outcome
tr <- data.frame(tr)
simple_mean <- mean(raw_data$loss)
pred_var <- ts$loss
pred_var[] <- simple_mean

# Checking the accuracy of the the model using MAE
avgmodel_accuracy <- accuracy(pred_var, ts$loss)
df1 <- as.data.frame(avgmodel_accuracy)
avgmodel_mae <- df1$MAE

# Add MAE results in the table 
mae_results <- data.frame(Method = "The Average Model", MAE = avgmodel_mae) 
mae_results %>% knitr::kable() 

######################################################
### MODEL 2: Linear Regression with PCA data
######################################################

fit <- lm(loss ~ ., data = tr)
valid_pred <- predict(fit, ts)

# Checking the accuracy of the the model using MAE
linear_reg_pca_accuracy = accuracy(valid_pred, ts$loss)
df2 <- as.data.frame(linear_reg_pca_accuracy)
linear_reg_pca_mae <- df2$MAE

# Add MAE results in the table 
mae_results <- bind_rows(mae_results, 
                         data_frame(Method="Linear Regression Model - PCA",   
                                    MAE = linear_reg_pca_mae)) 
mae_results %>% knitr::kable() 

######################################################
### MODEL 3: Linear Regression with Raw data
######################################################

fit <- lm(loss ~ ., data = raw_tr)
valid_pred <- predict(fit, raw_ts)

# Checking the accuracy of the the model using MAE
linear_reg_raw_accuracy = accuracy(valid_pred, raw_ts$loss)
df3 <- as.data.frame(linear_reg_raw_accuracy)
linear_reg_raw_mae <- df3$MAE

# Add MAE results in the table 
mae_results <- bind_rows(mae_results, 
                         data_frame(Method="Linear Regression Model - Raw",   
                                    MAE = linear_reg_raw_mae)) 
mae_results %>% knitr::kable()


######################################################
### MODEL 4: xgBoost Model - Using PCA Data
######################################################

#### Model preparation ####

# Extracting the loss variable from the training set 
tr_label <- tr$loss
ts_label <- ts$loss

# Plotting the histogram to check the distribution
hist(tr_label, main="Histogram of Loss", xlab="Loss",col = viridis(10))
hist(ts_label, main="Histogram of Loss", xlab="Loss",col = viridis(10))

# Feature engineering- transforming the loss variable to obtain normal-like distribution
tr_label_log <- log(tr$loss + 200)
ts_label_log <- log(ts$loss + 200)

# Plotting the histogram to check the new distribution
hist(tr_label_log, main="Histogram of Log(Loss)", xlab="Log(Loss)",col = viridis(20))
hist(ts_label_log, main="Histogram of Log(Loss)", xlab="Log(Loss)",col = viridis(20))

# Converting the train and test dataframes to a matrix
tr_matrix <- as.matrix(tr, rownames.force = NA)
ts_matrix <- as.matrix(ts, rownames.force = NA)

# Converting the train and test dataframes to a sparse matrix
tr_sparse <- as(tr_matrix, "sparseMatrix")
ts_sparse <- as(ts_matrix, "sparseMatrix")

# For xgboost, using xgb.DMatrix to convert data table into a matrix
dtrain <- xgb.DMatrix(data = tr_sparse[,1:78], label = tr_label_log )
dtest <- xgb.DMatrix(data = ts_sparse[,1:78], label = ts_label_log )


## preparation for xgboost model 

# defining default parameters
params <- list(booster = "gbtree", objective = "reg:linear", eta=0.1, gamma=0, nthread = 8,
               max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1, 
               early_stopping_rounds = 25, eval_metric = "mae", lambda=0, 
               prediction = TRUE, alpha=1)

# Using the xgb.cv function to calculate the best nround for this model. 

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 1500, nfold = 5, 
                 showsd = T, stratified = T, print_every_n = 100, maximize = F,
                 verbose = TRUE)


# Finding the best nrounds value
xgbcv$best_iteration


# Training the model on the best tuning parameters with nrounds = 1101. 
# This value is derived based on prior tuning and using it directly on train to save some computation time.
xgb1 <- xgb.train (params = params, 
                   data = dtrain, 
                   nrounds = 1101, 
                   watchlist = list(val=dtest,train=dtrain), 
                   print_every_n = 100, 
                   verbose = TRUE,
                   maximize = F)


# Model prediction
xgbpred <- predict(xgb1,dtest)

# Take the antilog of the predictions and subtracting 200 to get the error value in terms of the original loss variable
preds <- exp(xgbpred)-200

# Checking the accuracy of the the model using MAE
xboost_accuracy <- accuracy(preds, ts_label)
df4 <- as.data.frame(xboost_accuracy)
xboost_mae <- df4$MAE

# Add MAE results in the table 
mae_results <- bind_rows(mae_results, 
                         data_frame(Method="xBoost Model - PCA",   
                                    MAE = xboost_mae)) 
mae_results %>% knitr::kable() 


######################################################
### MODEL 5: xgBoost Model - Using Numeric Raw Data
######################################################

#### Model preparation ####

# Extracting the loss variable from the training set
tr_label <- raw_tr$loss
ts_label <- raw_ts$loss

# Feature engineering- transforming the loss variable to obtain normal-like distribution
tr_label_log <- log(raw_tr$loss + 200)
ts_label_log <- log(raw_ts$loss + 200)

# Converting the train and test dataframes to a matrix
tr_matrix <- as.matrix(raw_tr, rownames.force = NA)
ts_matrix <- as.matrix(raw_ts, rownames.force = NA)

# Converting the train and test dataframes to a sparse matrix
tr_sparse <- as(tr_matrix, "sparseMatrix")
ts_sparse <- as(ts_matrix, "sparseMatrix")

# For xgboost, using xgb.DMatrix to convert data table into a matrix
dtrain <- xgb.DMatrix(data = tr_sparse[,1:130], label = tr_label_log )
dtest <- xgb.DMatrix(data = ts_sparse[,1:130], label = ts_label_log )

## preparation for xgboost model

#defining default parameters
params <- list(booster = "gbtree", objective = "reg:linear", eta=0.1, gamma=0, nthread = 8,
               max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1, 
               early_stopping_rounds = 25, eval_metric = "mae", lambda=0, alpha=1)

# Using the xgb.cv function to calculate the best nround for this model. 

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 1500, nfold = 5, 
                 showsd = T, stratified = T, print_every_n = 100, maximize = F)

# finding the best nrounds value
xgbcv$best_iteration


# Training the model on the best tuning parameters with nrounds = 1181. 
# This value is derived based on prior tuning and using it directly on train to save some computation time.
xgb1 <- xgb.train (params = params, 
                   data = dtrain, 
                   nrounds = 1181, 
                   watchlist = list(val=dtest,train=dtrain), 
                   print_every_n = 100,
                   verbose = TRUE,
                   maximize = F)

# Model prediction
xgbpred <- predict(xgb1,dtest)

# Taking the antilog of the predictions and subtracting 200 to get the error value in terms of the original loss variable
preds <- exp(xgbpred)-200

# Checking the accuracy of the the model using MAE
xboost_raw_accuracy <- accuracy(preds, ts_label)
df5 <- as.data.frame(xboost_raw_accuracy)
xboost_raw_mae <- df5$MAE

# Add MAE results in the table 
mae_results <- bind_rows(mae_results, 
                         data_frame(Method="xBoost Model - Raw",   
                                    MAE = xboost_raw_mae)) 
mae_results %>% knitr::kable()


################################# End of Script ################################