import matplotlib.pyplot as pl
import numpy as np

f = open('Admission.txt')

data = f.readlines()

features_x = (np.array([line.split(',')[:] for line in data]).shape[1]) - 1

x = np.array([line.split(',')[0:features_x] for line in data], dtype = float)
y = np.array([line.split(',')[features_x:(features_x+1)] for line in data], dtype = float)

n_clusters = 2 #number of centroids
clusters = []

data_size = np.shape(x)[0]

centroids = np.zeros([n_clusters, features_x])
#print(clusters)

# print(np.amin(x, axis = 0)[0], np.amax(x, axis = 0)[0])
# print(np.amin(x, axis = 1)[1], np.amax(x, axis = 1)[1])

#initialize the centroids inside the range of the points
for c in range(n_clusters):
    cx = np.random.randint(np.min(x[0]), np.max(x[0]))
    cy = np.random.randint(np.min(x[1]), np.max(x[1]))
    centroids[c] = np.array((cx, cy))

print(centroids)


#initial centroids
pl.plot(centroids[:,0], centroids[:,1], '*', c = 'green')
pl.plot(x[:,0], x[:,1], 'o', c = 'blue')
#pl.show()

# for i in range(1000):
#     di = np.sum(np.sqrt((x[i , :] - c) ** 2), 1)
#     print(di)
