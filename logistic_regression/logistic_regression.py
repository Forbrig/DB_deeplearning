# is a kind of classification problem
import matplotlib.pyplot as pl
import numpy as np

f = open('Admission.txt')

data = f.readlines()

theta = np.array([[0, .1, .2, .3, .4]])

# automaticaly gets the number of features
features_x = (np.array([line.split(',')[:] for line in data]).shape[1]) - 1 # quantity of features in the dataset, (number of columns - 1) ex: 2 features in x
theta = np.array(theta[: , 0 : features_x + 1]) # reshape de size of theta
# the last column of our data must be the value that is dependant of these features (our y)

x = np.array([line.split(',')[0:features_x] for line in data], dtype = float)
y = np.array([line.split(',')[features_x:(features_x+1)] for line in data], dtype = float)

# discretizing our features (needed if the range is to big)
#x = x / (np.max(x, 0))

#sigmoid function
#def g(x, theta):
#    return 1 / (1 + np.e**(x.dot(theta.T))

#cost function
#def costFunction(x, y, theta):
#    return cost

pos = (y == 1).ravel() #ravel transform it in a list
neg = (y == 0).ravel()


pl.plot(x[pos, 0], x[pos, 1], 'x', c = 'r')
pl.plot(x[neg, 0], x[neg, 1], 'o', c = 'b')
pl.show()
