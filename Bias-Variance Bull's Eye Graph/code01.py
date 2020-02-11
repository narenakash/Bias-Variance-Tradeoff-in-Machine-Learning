from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from matplotlib import pyplot as plt
import numpy as np
import pickle

data = open('./Q1_data/data.pkl', 'rb')
dataset = pickle.load(data)
data.close()

x = np.array(dataset[ : , 0])[ :, np.newaxis]
y = np.array(dataset[ : , 1]).reshape(-1, 1)

xTrains, xTest, yTrains, yTest = train_test_split(x, y, test_size = 0.1, random_state = 0)
xTrain = np.array_split(xTrains, 10)
yTrain = np.array_split(yTrains, 10)
np.set_printoptions(threshold = np.inf)

bias_array = []
bias_aggregate = []
variance_array = []
variance_aggregate = []

for i in range(0, 10):
    print("Train Set: " + str(i) + " Degree: 1")
    regression = LinearRegression()
    regression.fit(xTrain[i], yTrain[i])
    yPrediction = regression.predict(xTest)

    bias_array.append(mean_absolute_error(yTest, yPrediction))
    print(mean_absolute_error(yTest, yPrediction))
    variance_array.append(np.var(yPrediction))
    print(np.var(yPrediction))
    print(regression.coef_)
    input()

for j in range(2, 10):
    for i in range(0, 10):
        print("Train Set: " + str(i) + " Degree: " + str(j))
        polynomial_features = PolynomialFeatures(degree = j)
        x_poly = polynomial_features.fit_transform(xTrain[i])
        x_poly_test = polynomial_features.fit_transform(xTest)
        regression_model = LinearRegression()
        regression_model.fit(x_poly, yTrain[i])
        y_polyPrediction = regression_model.predict(x_poly_test)

        bias_array.append(mean_absolute_error(yTest, y_polyPrediction))
        print(mean_absolute_error(yTest, y_polyPrediction))
        variance_array.append(np.var(y_polyPrediction))
        print(np.var(y_polyPrediction))
        print(regression_model.coef_)
        input()

for i in range(0, 9):
    bias_aggregate.append(np.mean(bias_array[i:i + 10]))
    variance_aggregate.append(np.mean(variance_array[i:i+ 10]))

print(bias_aggregate)
print(variance_aggregate)


