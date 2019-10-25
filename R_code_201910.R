################################################################
################################################################

## Name: Machine Learning methods for Economists: Application on an experiment 
## of Social Voting (Paper: Gerber, A. S., Green, D. P., & Larimer, C. W. (2008). 
## Social pressure and voter turnout: Evidence from a large-scale field experiment. 
## American political Science review, 102(1), 33-48.)

## Author: Dominik Prugger (replicating the Machine Learning course material of Prof. Susan Athey, Stanford)
## Date: October 2019



# Begin by deleting all previous working space and loading the required packages
rm(list = ls())

# State the packages required for this analysis
packages <- c("devtools"
              ,"randomForest"
              ,"rpart" # decision tree
              ,"rpart.plot" # enhanced tree plots
              ,"ROCR"
              ,"Hmisc"
              ,"corrplot"
              ,"texreg"
              ,"glmnet"
              ,"reshape2"
              ,"knitr"
              ,"xtable"
              ,"lars"
              ,"ggplot2"
              ,"matrixStats"
              ,"plyr"
              ,"stargazer")

# Check if these packages are already installed on the computer, if not install them 
list.of.packages <- packages
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "http://cran.us.r-project.org")

# Use sapply to require all packages in one
sapply(packages, require, character.only = TRUE)


# Specify working directory and load the data
setwd("~/Utrecht/GithubCode/ML and causal inference")

# load the csv.file
filename <- 'socialneighbor.csv'
social <- read.csv(file = filename, na = character())

# some simple print statements
print(paste("Loaded csv:", filename, " ..."))
colnames(social)

# Idea: Demonstrating the usefulness of AI/Machine Learning for Causal inference is better if the data set is "Wide". 
# This is currently, with only 69 observations not the case. Thus I create some extra "noise". That is unrelevant data
# that the ML algorithms then have to disregard. 

# I generate 13 noise variables and label them accordingly

set.seed(123)
noise.covars <- matrix(data = runif(nrow(social) * 13),
                       nrow = nrow(social), ncol = 13)
noise.covars <- data.frame(noise.covars)
names(noise.covars) <- c("noise1", "noise2", "noise3", "noise4", "noise5", "noise6",
                         "noise7", "noise8", "noise9", "noise10", "noise11", "noise12","noise13")

# Add these noise covariates to the social data
working <- cbind(social, noise.covars)

# For quickness, I just use a subset of the data, this is optional
set.seed(333)
working <- working[sample(nrow(social), 20000), ]

# Here I select some covariates, which will be used for the analysis. Generally, the more variables the better, 
# as it is easier to fix the overfitting problem than to fix the underfitting problem. 

covariate.names <- c("yob", "hh_size", "sex", "city", "g2000","g2002", "p2000", "p2002", "p2004"
                     ,"totalpopulation_estimate","percent_male","median_age", "percent_62yearsandover"
                     ,"percent_white", "percent_black", "median_income",
                     "employ_20to64", "highschool", "bach_orhigher","percent_hispanicorlatino",
                     "noise1", "noise2", "noise3", "noise4", "noise5", "noise6",
                     "noise7", "noise8", "noise9", "noise10", "noise11", "noise12","noise13")

# The main outcome variable is whether a person voted or not. I thus rename it to Y and extract it 
names(working)[names(working)=="outcome_voted"] <- "Y"

Y <- working[["Y"]]

# The main treatment variable is whether a person the "your neighbours have voted" letter. I thus relabel it w and extract it
names(working)[names(working)=="treat_neighbors"] <- "W"
W <- working[["W"]]

# Also extract the selected covariates
covariates <- working[covariate.names]

# For some of the following applications, I require scaled covariates, that are mean of 0 and standard deviation of 1
# with default settings, will calculate the mean and standard deviation of the entire vector,
# then "scale" each element by those values by subtracting the mean and dividing by the sd. Last create the new data frames
# This can be or more usually be done by scaling only the 

covariates.scaled <- scale(covariates)
processed.unscaled <- data.frame(Y, W, covariates)
processed.scaled <- data.frame(Y, W, covariates.scaled)


# I split the data into two parts: Validation and test data set, by setting a set seed (for replication) and then divide
# the data using a 90/10 split. This can be changed to e.g. a 80/20
# some of the models in the tutorial will require training, validation, and test sets. This gives the random rownumbers

set.seed(44)
smplmain <- sample(nrow(processed.scaled), round(9*nrow(processed.scaled)/10), replace=FALSE)

# now subset the data
processed.scaled.train <- processed.scaled[smplmain,]
processed.scaled.test <- processed.scaled[-smplmain,]

# For the outcome variable
y.train <- as.matrix(processed.scaled.train$Y, ncol=1)
y.test <- as.matrix(processed.scaled.test$Y, ncol=1)

# create 45-45-10 sample for cross-validation
smplcausal <- sample(nrow(processed.scaled.train),
                     round(5*nrow(processed.scaled.train)/10), replace=FALSE)
processed.scaled.train.1 <- processed.scaled.train[smplcausal,]
processed.scaled.train.2 <- processed.scaled.train[-smplcausal,]


#************************************************************************************************************

# Analysis of the data

#************************************************************************************************************


# Creating Formulas
# For many of the models, we will need a "formula"
# This will be in the format Y ~ X1 + X2 + X3 + ...
# For more info, see: http://faculty.chicagobooth.edu/richard.hahn/teaching/formulanotation.pdf

print(covariate.names)
#Adding a plus
sumx <- paste(covariate.names, collapse = " + ")  # "X1 + X2 + X3 + ..." for substitution later
interx <- paste(" (",sumx, ")^2", sep="")  # "(X1 + X2 + X3 + ...)^2" for substitution later

# Y ~ X1 + X2 + X3 + ...
linearnotreat <- paste("Y",sumx, sep=" ~ ")
linearnotreat <- as.formula(linearnotreat)
linearnotreat

# Y ~ W + X1 + X2 + X3 + ...
linear <- paste("Y",paste("W",sumx, sep=" + "), sep=" ~ ")
linear <- as.formula(linear)
linear

# Y ~ W * (X1 + X2 + X3 + ...)
# ---> X*Z means include these variables plus the interactions between them
linearint <- paste("Y", paste("W * (", sumx, ") ", sep=""), sep=" ~ ")
linearint <- as.formula(linearint)
linearint


#####################
# Linear Regression #
#####################
lm.linear <- lm(linear, data=processed.scaled)
summary(lm.linear)

lm.linearhet <- lm(linearhet, data=processed.scaled)
summary(lm.linearhet)


#######################
# Logistic Regression #
#######################
# See:http://www.ats.ucla.edu/stat/r/dae/logit.htm

# The code below estimates a logistic regression model using
# the glm (generalized linear model) function.
mylogit <- glm(linear, data = processed.scaled, family = "binomial")
summary(mylogit)

## Interpretation: Controlled for all the control variables, just the letter alone increases the 
## change to vote by nearly 47 percent. 


##################################
# LASSO variable selection + OLS #
##################################

# see https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html
# and also help(glmnet)

# LASSO takes in a model.matrix
# First parameter is the model (here we use linear, which we created before)
# Second parameter is the dataframe we want to creaate the matrix from
linear.train <- model.matrix(linear, processed.scaled.train)[,-1]
linear.test <- model.matrix(linear, processed.scaled.test)[,-1]
linear.train.1 <- model.matrix(linear, processed.scaled.train.1)[,-1]
linear.train.2 <- model.matrix(linear, processed.scaled.train.2)[,-1]

# Use cross validation to select the optimal shrinkage parameter lambda
# and the non-zero coefficients
lasso.linear <- cv.glmnet(linear.train.1, y.train[smplcausal,],  alpha=1, parallel=TRUE)

# prints the model, somewhat information overload,
# but you can see the mse, and the nonzero variables and the cross validation steps
lasso.linear

# plot & select the optimal shrinkage parameter lambda
# Note: if you see an error message "figure margins too large",
# try expanding the plot area in RStudio.
plot(lasso.linear)
lasso.linear$lambda.min
lasso.linear$lambda.1se

# lambda.min gives min average cross-validated error
# lambda.1se gives the most regularized model such that error is
# within one standard error of the min; this value of lambda is used here.

# List non-zero coefficients found. There are two ways to do this.
coef(lasso.linear, s = lasso.linear$lambda.1se) # Method 1
coef <- predict(lasso.linear, type = "nonzero") # Method 2

# index the column names of the matrix in order to index the selected variables
colnames <- colnames(linear.train.1)
selected.vars <- colnames[unlist(coef)]

# do OLS using these coefficients
linearwithlass <- paste("Y", paste(append(selected.vars, "W"),collapse=" + "), sep = " ~ ")
linearwithlass <- as.formula(linearwithlass)
lm.linear.lasso <- lm(linearwithlass, data=processed.scaled.train.2)
yhat.linear.lasso <- predict(lm.linear.lasso, newdata=processed.scaled.test)
summary(lm.linear.lasso)




