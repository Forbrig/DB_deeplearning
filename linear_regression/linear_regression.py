import numpy as np

f = open('linear_regression.in')

data = f.readlines()

## we need to put an arbitrary initial value in theta
theta = np.array([[0, .5, .8]]) ## theta will hold the value that explains the model of our algorithm
features_x = 2 ## quantity of features in the dataset, ex: 2 features in x
train = 0.7 ## percentage of all dataset that will be used to train our model
## the last column of our data must be the value that is dependant of these features (our y)

x = np.array([line.split(',')[0:features_x] for line in data], dtype = int)
y = np.array([line.split(',')[features_x:(features_x+1)] for line in data], dtype = int)

## add a column of 1's in the house features (so we could multiplicate this matrix with theta)
x = np.insert(x, 0, 1, axis = 1)

total = x.shape[0] ## number of lines in the matrix
qt_train = int(total * train) ## quantity of rows that will train our model is 70% of total of samples

x_train = x[:qt_train]
y_train = y[:qt_train]
x_test = x[qt_train:]
y_test = y[qt_train:]


def costFunction(x, y, theta):
    data_size = x.shape[0] ## size of the dataset
    y_hat = x.dot(theta.T) ## multiply the data with our theta Transposed
    cost = np.sum((y_hat - y)**2)
    return cost/(2*data_size)

res = costFunction(x_train, y_train, np.array(theta))

print(res)
