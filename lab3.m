%% information
% facial age estimation
% regression method: linear regression

%% settings
clear;
clc;

% path 
database_path = './data_age.mat';
result_path = './results/';

% initial states
absTestErr = 0;
cs_number = 0;


% cumulative error level
err_level = 5;

%% Training 
load(database_path);

nTrain = length(trData.label); % number of training samples
nTest  = length(teData.label); % number of testing samples
xtrain = trData.feat; % feature
ytrain = trData.label; % labels

w_lr = regress(ytrain,xtrain);
   
%% Testing
xtest = teData.feat; % feature
ytest = teData.label; % labels

yhat_test = xtest * w_lr;

%% Compute the MAE and CS value (with cumulative error level of 5) for linear regression 

% Compute the absolute error of each test image
abs_error = abs(yhat_test - ytest);
% Mean of the absolute error
mean_abs_error = mean(abs_error);

% Check if the absolute error is greater than the threshold value
% Calculate the percentage of error values greater than the threshold value
cs_value = sum(abs_error <= err_level) / nTest *100;


%% generate a cumulative score (CS) vs. error level plot by varying the error level from 1 to 15. The plot should look at the one in the Week6 lecture slides
% Create an array to store the cumulative score for different error levels
cumulative_score = zeros(15,1);
% Calculate the score for 1:15 error levels
for i = 1:15
    cumulative_score(i) = sum(abs_error <= i) / nTest *100;
end

figure
plot(cumulative_score, '-o');
xlabel('Error Levels');
ylabel('Cumulative Score');
title('CS value against the cumulative error level');
%% Compute the MAE and CS value (with cumulative error level of 5) for both partial least square regression and the regression tree model by using the Matlab built in functions.

% Partial Least Square Regression

%Training the PLS Regression model
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats] = plsregress(xtrain, ytrain, 10);
% Estimate the model on the test data
yhat_plsr = [ones(nTest,1) xtest] * BETA;

% Compute the absolute error of each test image for PLS model
abs_error_plsr = abs(yhat_plsr - ytest);
% Mean absolute error of PLS Regression model
mean_abs_error_plsr = mean(abs_error_plsr);

% Check if the absolute error is greater than the threshold value
% Calculate the percentage of error values greater than the threshold value
cs_value_plsr = sum(abs_error_plsr <= err_level) / nTest * 100;

% Regression Tree Model

%Training the Regression Tree model
tree = fitrtree(xtrain,ytrain);
% Estimate the model on the test data
yhat_regTree = predict(tree, xtest);

% Compute the absolute error of each test image for Regression Tree model
abs_error_regTree = abs(yhat_regTree - ytest);
% Mean absolute error of Regression Tree model
mean_abs_error_regTree = mean(abs_error_regTree);

% Check if the absolute error is greater than the threshold value
% Calculate the percentage of error values greater than the threshold value
cs_value_regTree = sum(abs_error_regTree <= err_level) / nTest * 100;
%% Compute the MAE and CS value (with cumulative error level of 5) for Support Vector Regression by using LIBSVM toolbox

% Training the Support Vector Regression model
Tb1 = fitrsvm(xtrain, ytrain);
% Estimate the model on the test data
yhat_svr = predict(Tb1, xtest);

% Compute the absolute error of each test image for Support Vector Regression model
abs_error_svr = abs(yhat_svr - ytest);
% Mean absolute error of Support Vector Regression model
mean_abs_error_svr = mean(abs_error_svr);

% Check if the absolute error is greater than the threshold value
% Calculate the percentage of error values greater than the threshold value
cs_value_svr = sum(abs_error_svr <= err_level) / nTest * 100;