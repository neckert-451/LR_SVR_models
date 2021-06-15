from matplotlib import pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# reading in data into python
data = pd.read_csv("glen_data.csv")
# want to predict the tensile strength based on fiber content
# first column is tensile strength so it is Y - the predicted
X = data.iloc[:, 0].values.reshape(-1, 1)
# second column is fiber content so it is X - the variable
Y = data.iloc[:, 1].values.reshape(-1, 1)

#
# LR model
linear_regress = LinearRegression()
linear_regress.fit(X, Y)
LR_predictor = linear_regress.predict(X)

#
# SVR Model
X1 = X.ravel()
Y1 = Y.ravel()
X1 = data.iloc[:, 0].values
print(X1.shape)
# second column is tensile strength so it is Y - the predicted
Y1 = data.iloc[:, 1].values
print(Y1.shape)
# reshape y to a column vector
Y1 = np.array(Y).reshape(-1, 1)
print(Y1.shape)

# check to see the data is being read
print(data.head(5))

# feature scaling to normalize data
sc_X1 = StandardScaler()
sc_Y1 = StandardScaler()
# convert 1D arrays to 2D arrays
X1 = sc_X1.fit_transform(X1.reshape(-1, 1))
print(X1.shape)
Y1 = sc_Y1.fit_transform(Y1.reshape(-1, 1))
print(Y1.shape)

# training the SVR model (will be trained on 2% of the data)
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

# fitting SVR to the data
# the kernel is important so determine which one you need to use
# rbf = radial basis function kernel which is the one DataRobot suggested
regressor = SVR(kernel="rbf")
regressor.fit(X_train.reshape(-1, 1), Y_train.reshape(-1, 1))

# predicting test results
Y_pred = regressor.predict(X_test)
print(Y_pred.shape)
Y_pred = sc_Y1.inverse_transform(Y_pred)

# comparing the test set with the predicted values
sc_Y1 = sc_Y1.inverse_transform(Y_test.reshape(-1))
df = pd.DataFrame({"Real Values": sc_Y1, "Predicted Values": Y_pred})
print(sc_Y1)
print(df)

#
# plot the data and predictions
# format the plot
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

# plot the original data
plt.scatter(X, Y, label="Actual Data")

# plot the LR model data
plt.scatter(X, LR_predictor, color="red", label="LR Prediction")
# plt.plot(X, LR_predictor, "r--")

# plot the SVR model data
# plt.scatter(sc_X1.inverse_transform(X_test), sc_Y1, color="black")
plt.scatter(sc_X1.inverse_transform(X_test), Y_pred, color="black", label="SVR Prediction")

# nice labels
plt.xlabel("Fiber Content")
plt.ylabel("Tensile Strength (MPa)")
plt.title("Conventional LR Model vs. SVR Model")
plt.legend()
plt.show()

#
# stats work for LR model
# calculating squared error
LR_error = Y - LR_predictor
LR_se = np.sum(LR_error**2)
print("LR SE = ", LR_se)
# calculating mean square error
n = np.size(X)  # n is the number of data points
LR_mse = LR_se/n
print("LR MSE = ", LR_mse)
# calculating root mean square error
LR_rmse = np.sqrt(LR_mse)
print("LR RMSE = ", LR_rmse)
# calculating R^2
LR_Y_mean = np.mean(Y)
LR_SSt = np.sum((Y - LR_Y_mean)**2)
LR_R2 = 1 - (LR_se/LR_SSt)
print("LR R^2 = ", LR_R2)

#
# stats work for SVR AI model
# calculating squared error
SVR_error = sc_Y1 - Y_pred
SVR_se = np.sum(SVR_error**2)
print("SVR SE = ", SVR_se)
# calculating mean square error
n = np.size(X)  # n is the number of data points
SVR_mse = SVR_se/n
print("SVR MSE = ", SVR_mse)
# calculating root mean square error
SVR_rmse = np.sqrt(SVR_mse)
print("SVR RMSE = ", SVR_rmse)
# calculating R^2
SVR_Y_mean = np.mean(Y)
SVR_SSt = np.sum((Y - SVR_Y_mean)**2)
SVR_R2 = 1 - (SVR_se/SVR_SSt)
print("SVR R^2 = ", SVR_R2)
