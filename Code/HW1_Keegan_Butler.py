## 1.Create a dataset with 10,000 rows and 4 random variables: 2 of them normally distributed, 2 uniformly distributed. 

## Import necessary libraries
import random
from sympy import *
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

## Set Seed
random.seed(12345678)

## Set limits for uniform "x" variables
x1_lower = 1
x1_upper = 2

x2_lower = -5
x2_upper = 5

## Set mean and standard deviation for gaussian "x" variables
x3_mean = 0
x3_stdev = 1

x4_mean = 10
x4_stdev = 2

## Generate uniform random variables
x1 = [random.uniform(x1_lower,x1_upper) for x in range(0,10000)] ## Uniformly random (1:2).
x2 = [random.uniform(x2_lower,x2_upper)  for x in range(0,10000)] ## Uniformly random (-5:5).

## Generate gaussian random variables
x3 = [random.gauss(x3_mean,x3_stdev) for x in range(0,10000)] ## Normally distributed, mean: 0, stdev: 1. 
x4 = [random.gauss(x4_mean,x4_stdev) for x in range(0,10000)] ## Normally distributed, mean: 10, stdev: 2. 

## Check that the random variables executed correctly
print("x1 length:", len(x1), "x1 min:", min(x1), "x1 max:", max(x1))
print("x2 length:", len(x2), "x2 min:", min(x2), "x2 max:", max(x2))
print("x3 length:", len(x3), "x3 mean:", np.mean(x3), "x3 standard deviation:", np.std(x3))
print("x4 length:", len(x4), "x4 mean:", np.mean(x4), "x4 standard deviation:", np.std(x4))

## 2. Add another variable ("y") as a linear combination, with some coefficients of your choice, of the 4 variables above;
## The Square of one variables; and some random "noise" (randomly distributed, centered at 0, small variance). 

## Set noise limits
noise_lower = -100
noise_upper = 100

## Generate noise
noise = random.uniform(noise_lower, noise_upper)

## Generate beta coefficients
beta = [3,1,2,3.5]

## Initialize y
y=[]

## Generate y as a linear combination of the x variables with the beta's as their constants, and noise
for i in range(len(x1)):
    ytemp= beta[0]*x1[i]**2 + beta[1]*x2[i]+ beta[2]*x3[i] + beta[3]*x4[i] + noise
    y.append(ytemp)

## Initialize x1 squared
x1_sq = []

## Square x1 values
for i in range(len(x1)):
    x1_sq_temp = x1[i]**2
    x1_sq.append(x1_sq_temp)

## Create data data frame.
data = Matrix([y, x1_sq, x2, x3, x4]).T

# Check data ran correctly printing y, number of variables in the data, and the mean of y
print("y length:", len(y))
print("data MxN:", shape(data)[1])
print("mean y:", np.mean(data.col(0)))

## 3. Split the dataset in #2 into 70% for training and 30% for testing.

## Set number of random variables created
it=10000

## Generate 3000 random numbers from 1 to 10000
sample_indx = random.sample(range(0, it), 3000)


## Initialize test and training data rows
data_test_row = []
data_train_row = []

## For loop to assign each row of the data to test or training data based off the random numbers from sample_indx
for i in range(data.shape[0]):
    if i in sample_indx:
        data_test_row.append(data.row(i))
    else:
        data_train_row.append(data.row(i))

## Group the rows of training and test data together into a single matrix
data_test = Matrix.vstack(*data_test_row)
data_train = Matrix.vstack(*data_train_row)

## Print the training and test matricies' shapes to ensure that they are correct
print("Data test shape:", shape(data_test))
print("Data train shape",shape(data_train))

## 4. Estimate the linear regression coefficients using OLS for the training data;
## Compute the Mean Standard Error on both the training dataset, and the testing dataset. 

## reset the training data incase del_col command deletes one of the training data columns. 
## This should not happen but its good to have it just in case. 
data_train = Matrix.vstack(*data_train_row)

## Create the OLS regression function in order to perform the OLS regression
def ols_regression(data_matrix):
    ## Separate out the y column
    y_reg = data_matrix.col(0)

    ## Create intercept column
    const_ = [1 for x in range(0,shape(data_matrix)[0])]
    const_ = Matrix([const_]).T

    ## create the design matrix by deleting the y column from data and adding the constant column
    X_reg = data_matrix
    if shape(X_reg)[1] == 5:
        X_reg.col_del(0)
    X_reg = X_reg.col_insert(0,const_)

    ## Calculate regression coefficients
    beta_hat_reg = ((X_reg.T*X_reg).inv())*(X_reg.T)*y_reg

    ## Calculate error
    error_reg = y_reg-X_reg*beta_hat_reg

    ## Calculate MSE
    MSE_reg = (error_reg.T*error_reg)/X_reg.shape[0]
    
    return beta_hat_reg, MSE_reg

## Run the training data through the model
beta_hat_train, MSE_train = ols_regression(data_train)

## Print the results of the training data
print("Training Betas:", beta_hat_train)
print("Training MSE:", MSE_train)

## Test the model created in the above steps using the training data
## Note to professor: I am assuming we are training the model on the training dataset then using that trained model on the test dataset. 

## reset the training data BECAUSE the del_col command deletes the y column of the training data when putting it into the design matrix. 
data_test = Matrix.vstack(*data_test_row)

## Create the columns for the intercept
test_const = [1 for x in range(0,shape(data_test)[0])]
test_const = Matrix([test_const]).T

## Separate out the y column, and add the intercept column to the design matrix
y_test = data_test.col(0)
x_test = data_test.col_del(0)
x_test = data_test.col_insert(0,test_const)

## Calculate the predicted values for the testing data
test_pred = x_test*beta_hat_train

## Calculate error and MSE for the testing data
error_test = y_test - test_pred
mse_test = (error_test.T*error_test)/x_test.shape[0]

## Print the results of the test
print("Testing MSE:", mse_test)

## 5. Use bootstrapping to create 10 other samples from the data you created in #2 above.

## Set number of simulations
sims = 10

## Create a dataframe in order to sample from
samp_data = pd.DataFrame({'y': y,'x1': x1_sq,'x2': x2,'x3': x3,'x4': x4})

# Initialize bootstrap sample list
bs_samp = []

## run the sampling through the sample data dataframe
for s in range (sims):
    bs_sample = samp_data.sample(10000, replace = True)
    bs_samp.append(bs_sample)

## Print one of the bootstrap runs to ensure it ran properly
bs_samp[9]

## 6. Estimate the linear regression coefficients using OLS for each of the 10 bootstrap samples in #5. 

## Initialize beta and MSE vectors (I think they're vectors)
beta_bs = []
mse_bs = []

## Run OLS regression through the bootstrap samples
for s in range (sims):
    data_reg = Matrix(bs_samp[s])
    
    beta_bs_temp, mse_bs_temp  = ols_regression(data_reg)
    
    beta_bs.append(beta_bs_temp)
    mse_bs.append(mse_bs_temp)

## 7. For each linear regression parameter, use the estimates computed in #6 and get the mean and standard deviation. 

## Initialize the beta values that came from your bootstrap regressions
beta_0_bs = []
beta_1_bs = []
beta_2_bs = []
beta_3_bs = []
beta_4_bs = []

## Assign each bootstrap regression coefficient to the vector of its coefficient (ie B0 values go to the beta_o_bs vector)
for s in range (sims):
    for b in range (beta_count):
        if b == 0:
            beta_0_temp = beta_bs[s][b]
            beta_0_bs.append(beta_0_temp)
        elif b == 1:
            beta_1_temp = beta_bs[s][b]
            beta_1_bs.append(beta_1_temp)
        elif b == 2:
            beta_2_temp = beta_bs[s][b]
            beta_2_bs.append(beta_2_temp)
        elif b == 3:
            beta_3_temp = beta_bs[s][b]
            beta_3_bs.append(beta_3_temp)
        elif b == 4:
            beta_4_temp = beta_bs[s][b]
            beta_4_bs.append(beta_4_temp)

## Print the beta values from each bootstrap sample to ensure you assigned them properly
print("Beta 0 Values:", beta_0_bs)
print("Beta 1 Values:", beta_1_bs)
print("Beta 2 Values:", beta_2_bs)
print("Beta 3 Values:", beta_3_bs)
print("Beta 4 Values:", beta_4_bs)

## Calculate the mean of the bootstrap sampled regression coefficients
mean_beta_0_bs = np.mean(beta_0_bs)
mean_beta_1_bs = np.mean(beta_1_bs)
mean_beta_2_bs = np.mean(beta_2_bs)
mean_beta_3_bs = np.mean(beta_3_bs)
mean_beta_4_bs = np.mean(beta_4_bs)

## Print the mean of the bootstrap sampled regression coefficients
print("Mean Beta 0:", mean_beta_0_bs)
print("Mean Beta 1:", mean_beta_1_bs)
print("Mean Beta 2:", mean_beta_2_bs)
print("Mean Beta 3:", mean_beta_3_bs)
print("Mean Beta 4:", mean_beta_4_bs)

## Calculate the standard of the bootstrap sampled regression coefficients. 
## The code wouldn't run without making sure each of the beta values were as a float so make sure theyre converted.
std_beta_0_bs = np.std([float(b) for b in beta_0_bs])
std_beta_1_bs = np.std([float(b) for b in beta_1_bs])
std_beta_2_bs = np.std([float(b) for b in beta_2_bs])
std_beta_3_bs = np.std([float(b) for b in beta_3_bs])
std_beta_4_bs = np.std([float(b) for b in beta_4_bs])

## Print standard deviations
print("Standard Deviation Beta 0:", std_beta_0_bs)
print("Standard Deviation Beta 1:", std_beta_1_bs)
print("Standard Deviation Beta 2:", std_beta_2_bs)
print("Standard Deviation Beta 3:", std_beta_3_bs)
print("Standard Deviation Beta 4:", std_beta_4_bs)

## 8. What can you say about the coefficients in #4 looking at the results in #7. 

## I can say that based off the bootstrap sampling of the beta coefficients the variance of all of the coefficients is remarkably low. 
## This likely has something to do with how small the noise was as a factor in generating the y values, 
## at the very least compared to the linear combination of the x random variables. 
## It also surely has something to do with the number of observations being so large on top of that, that we were assuredly going to get
## an accurate representation of the "population" although in this case the population is theoretical because we did not generate a
## population and then sample from it, we just generated random samples from a theoretical population that took the shape of
## the linear combination of the x values with a little bit of noise added in. 