from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from prettytable import PrettyTable
from matplotlib import pyplot as plt
import numpy as np
import pickle

bias_array = []
variance_array = []
bias_aggregate = []
variance_aggregate = []

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

    print("Train Set: " + str(i + 1) + " Degree: 1")

    regression = LinearRegression()
    regression.fit(datasetXTrain[i][: , np.newaxis], datasetYTrain[i][: , np.newaxis])
    yPrediction = regression.predict(datasetXTest[: , np.newaxis])

    """Calculate the bias and variance for the trained
    model on the testing set"""
    bias_value = round(np.average(yPrediction) - np.average(datasetFxTest[: , np.newaxis]), 3)
    variance_value = round(np.var(yPrediction), 3)

    bias_array.append(bias_value)
    variance_array.append(variance_value)
    print("Bias: " + str(bias_value) + "   Variance: " + str(variance_value) + "\n")
    # input()

#######################################################
# Run Polynomail Regression for the 20 train sets
#######################################################

for j in range(2, 10):
    for i in range(0, 20):
        """Train the polynomial regression model and run it with 
        the test set for prediction."""
        print("Train Set: " + str(i + 1) + " Degree: " + str(j))

        polynomial_features = PolynomialFeatures(degree = j)
        x_poly = polynomial_features.fit_transform(datasetXTrain[i][ : , np.newaxis])
        x_poly_test = polynomial_features.fit_transform(datasetXTest.reshape(-1, 1))

        regression_model = LinearRegression()
        regression_model.fit(x_poly, datasetYTrain[i].reshape(-1, 1))
        y_polyPrediction = regression_model.predict(x_poly_test)

        """Calculate the bias and variance for the trained
        model on the testing set"""
        bias_value = round(np.average(y_polyPrediction) - np.average(datasetFxTest[ : , np.newaxis]), 3)
        variance_value = round(np.var(y_polyPrediction[ : , np.newaxis]), 3)

        bias_array.append(bias_value)
        variance_array.append(variance_value)
        print("Bias: " + str(bias_value) + "   Variance: " + str(variance_value) + "\n")
        # input()

#######################################################
# Average Bias and Variance for each class of functions
# Tabluate them using prettytable library
#######################################################

table = PrettyTable(["Degree", "Average Bias", "Average Variance"])

for i in range(0, 9):
    bias_aggregate.append(round(np.mean(bias_array[i : i + 20]), 3))
    variance_aggregate.append(round(np.average(variance_array[i : i + 20]), 3))
    table.add_row([i + 1, bias_aggregate[i], variance_aggregate[i]])

bias_square_aggregate = np.square(bias_aggregate)
print(table)

#######################################################
# Normalizing the Average Bias^2 and Variance Values
# Unity Based Normalization or, Min-Max Normalization
#######################################################

arr1_max = np.amax(bias_square_aggregate[: , np.newaxis]).item()
arr1_min = np.amin(bias_square_aggregate[: , np.newaxis]).item()
bias_min_max_normalized_aggregate = [(value - arr1_max)/(arr1_max - arr1_min) for value in bias_square_aggregate]

arr2_max = np.amax(variance_aggregate).item()
arr2_min = np.amin(variance_aggregate).item()
variance_normalized_aggregate = [(value - arr2_max)/(arr2_max - arr2_min) for value in variance_aggregate] 

#######################################################
# Plotting Model Complexity vs Bias^2 and Variance
#######################################################

MC = [1, 2, 3, 4, 5, 6, 7, 8, 9] 
plt.plot(MC, bias_min_max_normalized_aggregate, color='g', label="Bias Square Normalized")
plt.plot(MC, variance_normalized_aggregate, color='r', label="Variance Normalized")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
plt.show()