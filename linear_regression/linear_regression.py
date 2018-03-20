import matplotlib.pyplot as pl
import numpy as np

f = open('truck_food_profit.in') # 1 feature
#f = open ('housing_prices.in') # 2 features

data = f.readlines()

# we need to put an arbitrary initial value in theta
theta = np.array([[0, .1, .2, .3, .4]]) # theta will hold the value that explains the model of our algorithm (number of features + 1 [bias that is the first!])

# automaticaly gets the number of features
features_x = (np.array([line.split(',')[:] for line in data]).shape[1]) - 1 # quantity of features in the dataset, (number of columns - 1) ex: 2 features in x
theta = np.array(theta[: , 0 : features_x + 1]) # reshape de size of theta
train = 0.7 # percentage of all dataset that will be used to train our model
# the last column of our data must be the value that is dependant of these features (our y)

x = np.array([line.split(',')[0:features_x] for line in data], dtype = float)
y = np.array([line.split(',')[features_x:(features_x+1)] for line in data], dtype = float)

# discretizing our features (needed if the range is to big)
x = x / (np.max(x, 0))

# just to see how it works
#print(np.max(x, 0))
#print(np.min(x, 0))

# add a column of 1's in the house features (so we could multiplicate this matrix with theta)
x = np.insert(x, 0, 1, axis = 1)

total = x.shape[0] # number of lines in the matrix
qt_train = int(total * train) # quantity of rows that will train our model is 70% of total of samples

x_train = x[:qt_train]
y_train = y[:qt_train]
x_test = x[qt_train:]
y_test = y[qt_train:]

# cost function (measure the distance between the points and the theta line)
def costFunction(x, y, theta):
    data_size = x.shape[0] # size of the dataset
    y_hat = x.dot(theta.T) # multiply the data with our theta Transposed
    cost = np.sum((y_hat - y)**2)
    return cost / (2 * data_size)

# predicted 'y' (used after we calculate our theta with gradient descendant)
def predicted_y(x, theta):
    return (x.dot(theta.T))

# gradient descendant algorithm
def gradient_descendant(x, y, theta, alpha, n):
    data_size = x.shape[0]
    J = []
    for i in range(1, n):
        y_hat = x.dot(theta.T)
        error = (y_hat - y)
        error = error * x
        error = np.sum(error, 0) / data_size
        theta = theta - (alpha * error)
        J.append(costFunction(x, y, theta))
    return theta, J

theta, J = gradient_descendant(x, y, theta, 0.01, 1000) # applies gradient descendant to approach ou theta

#y_hat = predicted_y(x_train, theta)
y_hat = predicted_y(x_test, theta) # calculate the predicted values that our model can get from the theta that we learned and our data test

if features_x == 1: # if we can plot it in a linear model
    #pl.plot(x_train[:, 1], y_train, '*', color = 'b')
    #pl.plot(x_train[:, 1], y_hat, color = 'r')
    pl.plot(x_test[:, 1], y_test, '*', color = 'b')
    pl.plot(x_test[:, 1], y_hat, color = 'r')
    pl.show()
else:
    print("theta:", theta)
    print("error: ", costFunction(x_test, y_test, theta))
