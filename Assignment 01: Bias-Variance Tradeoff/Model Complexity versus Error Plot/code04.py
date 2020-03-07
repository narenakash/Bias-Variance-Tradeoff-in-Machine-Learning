from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from prettytable import PrettyTable
from matplotlib import pyplot as plt
import numpy as np
import pickle

bias_list = []
bias_square_list = []
variance_list = []
total_error_list = []
yP = np.zeros((80, 1))

#######################################################
# Unpicking the data and storing them in an array
#######################################################

dataXTrain = open('./Q2_data/X_train.pkl', 'rb')
datasetXTrain = pickle.load(dataXTrain)
dataXTrain.close()

dataYTrain = open('./Q2_data/Y_train.pkl', 'rb')
datasetYTrain = pickle.load(dataYTrain)
dataYTrain.close()

dataXTest = open('./Q2_data/X_test.pkl', 'rb')
datasetXTest = pickle.load(dataXTest)
dataXTest.close()

dataFxTest = open('./Q2_data/Fx_test.pkl', 'rb')
datasetFxTest = pickle.load(dataFxTest)
dataFxTest.close()

#######################################################
# Run Linear Regression Algorithm for the 20 train sets
#######################################################

for i in range(0, 20):
    """Train the linear regression model and run it with 
    the test set for prediction."""

    regression = LinearRegression()
    regression.fit(datasetXTrain[i][: , np.newaxis], datasetYTrain[i][: , np.newaxis])
    yPrediction = regression.predict(datasetXTest[: , np.newaxis])

    """Calculate the bias and variance for the trained
    model on the testing set"""
    yP = np.append(yP, yPrediction, axis=1)

#######################################################
# Run Polynomail Regression for the 20 train sets
#######################################################

for j in range(2, 10):
    for i in range(0, 20):
        """Train the polynomial regression model and run it with 
        the test set for prediction."""
        polynomial_features = PolynomialFeatures(degree = j)
        x_poly = polynomial_features.fit_transform(datasetXTrain[i][ : , np.newaxis])
        x_poly_test = polynomial_features.fit_transform(datasetXTest.reshape(-1, 1))

        regression_model = LinearRegression()
        regression_model.fit(x_poly, datasetYTrain[i].reshape(-1, 1))
        y_polyPrediction = regression_model.predict(x_poly_test)

        """Calculate the bias and variance for the trained
        model on the testing set"""
        yP = np.append(yP, y_polyPrediction, axis=1)

#######################################################
# Average Bias and Variance for each class of functions
# Tabluate them using prettytable library
#######################################################

table = PrettyTable(["Degree", "Average Bias", "Average Bias Square", "Average Variance"])

for i in range(0, 9):
    yMean = np.mean(yP[:,i*20+1:i*20+21], axis = 1)[: , np.newaxis]

    bias_array = np.absolute(yMean - datasetFxTest[:, np.newaxis])
    bias_list.append(round(np.mean(bias_array), 3))
    bias_square_list.append(round(np.mean((bias_array)**2), 3))

    variance_array = np.mean((yMean - yP[:,i*20+1:i*20+21])**2)
    variance_list.append(round(variance_array, 4))
    total_error_list.append(variance_list[i] + bias_square_list[i])

    table.add_row([i + 1, bias_list[i], bias_square_list[i], variance_list[i]])

print(table)

#######################################################
# Normalizing the Average Bias^2 and Variance Values
# Unity Based Normalization or, Min-Max Normalization
#######################################################

arr1_max = max(bias_square_list)
arr1_min = min(bias_square_list)
bias_min_max_normalized_aggregate = [(value - arr1_max)/(arr1_max - arr1_min) for value in bias_square_list]

arr2_max = max(variance_list)
arr2_min = min(variance_list)
variance_normalized_aggregate = [(value - arr2_max)/(arr2_max - arr2_min) for value in variance_list] 

#######################################################
# Plotting Model Complexity vs Bias^2 and Variance
#######################################################

MC = [1, 2, 3, 4, 5, 6, 7, 8, 9]
plt.xlabel('Model Complexity')
plt.ylabel('Error') 
plt.plot(MC, bias_min_max_normalized_aggregate, color='g', label="Bias Square Normalized")
plt.plot(MC, variance_normalized_aggregate, color='r', label="Variance Normalized")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
plt.show()

MC = [1, 2, 3, 4, 5, 6, 7, 8, 9] 
plt.xlabel('Model Complexity')
plt.ylabel('Error')
plt.plot(MC, bias_square_list, color='g', label="Bias Square")
plt.plot(MC, variance_list, color='r', label="Variance")
plt.plot(MC, total_error_list, color='b', label="Approximate Total Error")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
plt.show()