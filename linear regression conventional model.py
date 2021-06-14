import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# reading csv with FOA data
data = pd.read_csv("glen_data.csv")
# want to predict the tensile strength based on fiber content
# first column is tensile strength so it is Y - the predicted
Y = data.iloc[:, 0].values.reshape(-1, 1)
# second column is fiber content so it is X - the variable
X = data.iloc[:, 1].values.reshape(-1, 1)

# set up the linear regression model and fit the model to the data
linear_regress = LinearRegression()
linear_regress.fit(X, Y)
Y_predictor = linear_regress.predict(X)

# plot the data and predictions
plt.scatter(X, Y)
plt.xlabel('Fiber Content')
plt.ylabel('Tensile Strength (MPa)')
plt.plot(X, Y_predictor, color='red')
plt.show()

# stats work
# calculating squared error
error = Y - Y_predictor
se = np.sum(error**2)
print("Squared Error = ", se)
# calculating mean square error
n = np.size(X)  # n is the number of data points
mse = se/n
print("Mean Squared Error = ", mse)
# calculating root mean square error
rmse = np.sqrt(mse)
print("Root Mean Square Error = ", rmse)
# calculating R^2
Y_mean = np.mean(Y)
SSt = np.sum((Y - Y_mean)**2)
R2 = 1 - (se/SSt)
print("R^2 = ", R2)

# getting the equation of the plotted line
X_mean = np.mean(X)
Y_mean = np.mean(Y)
Sxy = np.sum(X * Y) - n * X_mean * Y_mean
Sxx = np.sum(X * X) - n * X_mean * X_mean
b1 = Sxy / Sxx
b0 = Y_mean - b1 * X_mean
print("Slope b1 is", b1)
print("Intercept b0 is", b0)
print("Equation of the line is: Y =", b1, "*X +", b0)
