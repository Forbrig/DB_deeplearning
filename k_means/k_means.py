import matplotlib.pyplot as pl
import numpy as np

#initialize the centroids inside the range of the points
def init_centroids(n_clusters, centroids, data_size):
    for i in range(n_clusters):
        centroids[i] = x[np.random.randint(data_size)]
    return centroids

#cost function (seems a little bit off)
def costFunction(x, centroids):
    costs = np.zeros((len(centroids),1))
    for i in range(x.shape[0]):
        cost = np.sum(np.sqrt((x[i,:] - centroids)**2), 1)
        argmin = np.argmin(cost)
        costs[argmin] = costs[argmin] + cost[argmin]
    return np.sum(costs)

#clusterizating function
def clusterizate(clusters, iter):
    association = np.zeros((data_size,1)) # number of samples
    for j in range(iter):
        old_clusters = clusters.copy()
        for i in range(data_size):
            di = np.sum(np.sqrt((x[i,:] - clusters)**2), 1)
            association[i] = np.argmin(di)
        for i in range(len(clusters)):
            points = x[association[:,0] == i]
            if points.size != 0: # if there is no element in the cluster
                mean_points = np.mean(points, axis=0)
                clusters[i] = mean_points
    return clusters, association

############################################
f = open('RelationNetwork.csv')
data = f.readlines()

features_x = (np.array([line.split(',')[:] for line in data]).shape[1]) - 1
x = np.array([line.split(',')[0:features_x] for line in data], dtype = float)
y = np.array([line.split(',')[features_x:(features_x+1)] for line in data], dtype = float)

# normalizing the features
x = (x - np.mean(x, 0)) / np.std(x, 0, ddof=1)
data_size = np.shape(x)[0]

#getting the set of different sets of clusters and plotting the elbow graph
elbows = np.zeros((15,1))
lowest_cost = 123456789
best_n_clusters = 1
for n_clusters in range(1, 16):
    centroids = np.zeros([n_clusters, features_x])
    centroids = init_centroids(n_clusters, centroids, data_size)
    centroids, association = clusterizate(centroids, 25)
    cost = costFunction(x, centroids)
    if cost < lowest_cost:
        lowest_cost = cost
        best_n_clusters = n_clusters
    elbows[n_clusters -1] = cost
    print("Finished set of", n_clusters, "cluster(s).")
pl.plot(range(1,16), elbows, 'o', c='b')
pl.show()
print("I don't know how to pick the elbow automatically, so i'm just getting the set of cluster with minor error...")
print("Best number of clusters", best_n_clusters, "with cost", lowest_cost)

# choose the best configuration
lowest_cost = 123456789
iteration = 0

centroids = np.zeros([best_n_clusters, features_x])
centroids = init_centroids(best_n_clusters, centroids, data_size)
for n in range(50, 100):
    print("Clusterizing", n, "times.")
    centroids, association = clusterizate(centroids, 1)
    cost = costFunction(x, centroids)
    if cost < lowest_cost:
        iteration = n
        lowest_cost = cost

print("Best clusters found in iteration", iteration)
