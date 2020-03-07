from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from prettytable import PrettyTable
from matplotlib import pyplot as plt
import numpy as np
import pickle

bias_array = []
bias_aggregate = []
variance_array = []
variance_aggregate = []

#######################################################
# Unpicking the data and storing them in an array
#######################################################

data = open('./Q1_data/data.pkl', 'rb')
dataset = pickle.load(data)
data.close()

#######################################################
# Spliting into training and testing data sets
#######################################################

x = np.array(dataset[ : , 0])[ :, np.newaxis]
y = np.array(dataset[ : , 1]).reshape(-1, 1)

xTrains, xTest, yTrains, yTest = train_test_split(x, y, test_size = 0.1, random_state = 0)
xTrain = np.array_split(xTrains, 10)
yTrain = np.array_split(yTrains, 10)
# np.set_printoptions(threshold = np.inf)

#######################################################
# Setting up matplotlib subplots for evaluating models
#######################################################

xx = np.linspace(-1, 6, 200)
XX = xx[ : , np.newaxis]
fig, axs = plt.subplots(9, 10)

#######################################################
# Run Linear Regression Algorithm for the 10 train sets
#######################################################

for i in range(0, 10):
    print("Train Set: " + str(i + 1) + " Degree: 1")

    regression = LinearRegression()
    regression.fit(xTrain[i], yTrain[i])
    yPrediction = regression.predict(xTest)
    
    bias_value = round(np.average(yPrediction)- np.average(yTest), 3)
    variance_value = round(np.var(yPrediction), 3)

    bias_array.append(bias_value)
    variance_array.append(variance_value)
    print("Bias: " + str(bias_value) + "   Variance: " + str(variance_value) + "\n")

    yy = regression.coef_.item() * xx
    axs[0, i].scatter(xTrain[i], yTrain[i], color='g', label="Test Set")
    axs[0, i].plot(xx, yy , 'r', label="Prediction")
    # axs[0, i].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
    # input()

#######################################################
# Run Polynomail Regression for the 10 train sets
#######################################################

for j in range(2, 10):
    for i in range(0, 10):
        print("Train Set: " + str(i + 1) + " Degree: " + str(j))

        polynomial_features = PolynomialFeatures(degree = j)
        x_poly = polynomial_features.fit_transform(xTrain[i])
        x_poly_test = polynomial_features.fit_transform(xTest)

        regression_model = LinearRegression()
        regression_model.fit(x_poly, yTrain[i])
        y_polyPrediction = regression_model.predict(x_poly_test)
        
        X = polynomial_features.fit_transform(XX)
        YY = regression_model.predict(X)

        bias_value = round(np.average(y_polyPrediction) - np.average(yTest), 3)
        variance_value = round(np.var(y_polyPrediction), 3)
        bias_array.append(bias_value)
        variance_array.append(variance_value)
        print("Bias: " + str(bias_value) + "   Variance: " + str(variance_value) + "\n")

        axs[j -1, i].set_ylim([-25, 25])
        axs[j - 1, i].scatter(xTrain[i], yTrain[i], color='g', label="Test Set")
        axs[j - 1, i].plot(xx, YY , 'r', label="Prediction")
        # axs[j - 1, i].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
        # input()

#######################################################
# Average Bias and Variance for each class of functions
#######################################################
table = PrettyTable(["Degree", "Average Bias", "Average Variance"])

for i in range(0, 9):
    bias_aggregate.append(round(np.mean(bias_array[i : i + 10]), 3))
    variance_aggregate.append(round(np.average(variance_array[i : i + 10]), 3))
    table.add_row([i + 1, bias_aggregate[i], variance_aggregate[i]])

print(table)

#######################################################
# Plotting the Testing Set and Prediction Function
#######################################################

plt.show()