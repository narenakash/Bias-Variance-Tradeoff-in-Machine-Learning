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
yP = np.zeros((500, 1))

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
    # print("Train Set: " + str(i + 1) + " Degree: 1")

    regression = LinearRegression()
    regression.fit(xTrain[i], yTrain[i])
    yPrediction = regression.predict(xTest)

    yP = np.append(yP, yPrediction, axis=1) 

    yy = regression.coef_.item() * xx
    axs[0, i].scatter(xTrain[i], yTrain[i], color='g', label="Test Set")
    axs[0, i].plot(xx, yy , 'r', label="Prediction")

#######################################################
# Run Polynomail Regression for the 10 train sets
#######################################################

for j in range(2, 10):
    for i in range(0, 10):
        # print("Train Set: " + str(i + 1) + " Degree: " + str(j))

        polynomial_features = PolynomialFeatures(degree = j)
        x_poly = polynomial_features.fit_transform(xTrain[i])
        x_poly_test = polynomial_features.fit_transform(xTest)

        regression_model = LinearRegression()
        regression_model.fit(x_poly, yTrain[i])
        y_polyPrediction = regression_model.predict(x_poly_test)

        yP = np.append(yP, y_polyPrediction, axis=1)
        
        X = polynomial_features.fit_transform(XX)
        YY = regression_model.predict(X)

        axs[j -1, i].set_ylim([-25, 25])
        axs[j - 1, i].scatter(xTrain[i], yTrain[i], color='g', label="Test Set")
        axs[j - 1, i].plot(xx, YY , 'r', label="Prediction")

#######################################################
# Average Bias and Variance for each class of functions
#######################################################
table = PrettyTable(["Degree", "Average Bias", "Average Bias Square", "Average Variance"])

# print(yMean.shape)
# input()

for i in range(0, 9):
    yMean = np.mean(yP[:,i*10+1:i*10+11], axis = 1)[: , np.newaxis]
    # bias_array = yP[:,i+1:i+11] - yMean
    bias_array = np.absolute(yMean - yTest)
    bias_list.append(round(np.mean(bias_array), 3))
    bias_square_list.append(round(np.mean((bias_array)**2), 3))

    variance_array = np.mean((yMean - yP[:,i*10+1:i*10+11])**2, axis=1)
    variance_list.append(round(np.mean(variance_array, axis=0), 4))
    table.add_row([i + 1, bias_list[i], bias_square_list[i], variance_list[i]])

print(table)

#######################################################
# Plotting the Testing Set and Prediction Function
#######################################################

input()
plt.show()
