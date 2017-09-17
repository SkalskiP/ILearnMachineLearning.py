import numpy as np
import pandas as pd
import random

def kmeans(k, epsilon = 0, distance = 'euclidian'):
    # list to store past centroid
    history_centroids = []
    # set the distance calculation type
    if distance == 'euclidian':
        dist_method = euclidian
        dataset = pd.read_csv('Mall_Customers.csv')
        dataset = dataset.iloc[:, [3, 4]].values
        # get the number of rows (instances) and columns (features) from dataset
        num_instances, num_features = dataset.shape
        # define k centroids (how many clusters do we want to find?) chose randomly
        prototypes = dataset[np.random.randint(0, num_instances-1, size=k)]
        # set these to our list of past centroid (to show progress over time)
        history_centroids.append(prototypes)
        # to keep track of centroid at every iteration
        prototypes_old = np.zeros(prototypes.shape)
        # to store clusters
        belongs_to = np.zeros((num-instances, 1))
        norm = dist_method(prototypes, prototypes_old)
        iteration = 0