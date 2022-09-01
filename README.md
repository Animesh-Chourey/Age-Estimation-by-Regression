# Age Estimation by Regression

The implementation done here is on age estimation from facial images. The program takes the AAM parameters as representation of human faces and learn a regression function to predict age for an unseen face.

Following taks have been performed here:
* "Multiple Linear Regression model" is used through regress() function to learn regression model.
* The learned linear regression model is applied on the test data to estimate the age for each test data point.
* Mean Absolute Error (MAE) and Cumulative Score (CS) is computed for both partial least squares regression model and the regression tree model.
* Finally, MAE and CS is computed for Support Vector Regression.

For more details look into the report.