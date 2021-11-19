import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

# reading data into python
data = pd.read_csv("material_data.csv")
# want to predict the tensile strength based on fiber content
# first column is fiber content so it is X - the variable
X = data.iloc[:, 0].values
print(X.shape)
# second column is tensile strength so it is Y - the predicted
y = data.iloc[:, 1].values
# z is the y that we'll use for stats at the end (otherwise, y is transformed for the model
z = data.iloc[:, 1].values
print(y.shape)
# reshape y to a column vector
y = np.array(y).reshape(-1, 1)
print(y.shape)

# check to see the data is being read
print(data.head(5))

# feature scaling to normalize data
sc_X = StandardScaler()
sc_y = StandardScaler()
# convert 1D arrays to 2D arrays
X = sc_X.fit_transform(X.reshape(-1, 1))
print(X.shape)
y = sc_y.fit_transform(y.reshape(-1, 1))
print(y.shape)

# training the SVR model (will be trained on 2% of the data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# fitting SVR to the data
# the kernel is important so determine which one you need to use
# rbf = radial basis function kernel which is the one DataRobot suggested
regressor = SVR(kernel="rbf")
regressor.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

# predicting test results
y_pred = regressor.predict(X_test)
print(y_pred.shape)
y_pred = sc_y.inverse_transform(y_pred)
print(y_pred.shape)

# comparing the test set with the predicted values
sc_y = sc_y.inverse_transform(y_test.reshape(-1))
df = pd.DataFrame({"Real Values": sc_y, "Predicted Values": y_pred})
print(sc_y)
print(df)

# plotting the actual data and predicted data
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X_test), sc_y, color="blue", label="Actual Data")
# in order to plot predicted data (with a line) we need to transform and sort the values
# this function transforms the sc_X vector array into an array so we can plot it
sc_XX = sc_X.inverse_transform(X_test.reshape(-1))
print(sc_XX)
# sorted sc_XX values - USE THIS WHEN PLOTTING
sorted_sc_XX = sorted(sc_XX)
print(sorted_sc_XX)
# sorted y_pred values - USE THIS WHEN PLOTTING
sorted_y_pred = sorted(y_pred)
print(sorted_y_pred)
# finally, we can plot the predicted values via scatter and line plots
plt.scatter(sorted_sc_XX, sorted_y_pred, color="black", label="Predicted Data")
plt.plot(sorted_sc_XX, sorted_y_pred, color="black")
plt.xlabel("Fiber Content")
plt.ylabel("Tensile Strength (MPa)")
plt.title("AI Model: Support Vector Regression (SVR)")
plt.legend()
plt.show()

# stats work
# calculating squared error
error = sc_y - y_pred
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
z_mean = np.mean(z)
print(z)
SSt = np.sum((z - z_mean)**2)
R2 = 1 - (se/SSt)
print("R^2 = ", R2)
