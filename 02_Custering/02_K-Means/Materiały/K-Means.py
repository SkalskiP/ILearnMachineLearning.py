import numpy as np
import pandas as pd
import random

def kmeans(k, epsilon = 0, distance = 'euclidian'):
    """
    k - iloć klastrów na jaką podzielimy nasze dane
    epsilon - minimalny dopuszczalny przez nas błąd, po przekroczeniu tego progu program zostanie wstrzymany
    distance - sposób obliczania odległoci pomiędzy punktami
    """
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
        belongs_to = np.zeros((num_instances, 1))
        norm = dist_method(prototypes, prototypes_old)
        iteration = 0
        
        while norm > epsilon:
            iteration += 1
            norm = dist_method(prototypes, prototypes_old)
            # for each instnce in the dataset
            for index_instance, instance in enumerate(dataset):
                # find a distance vector of size k
                dist_vec = np.zeros((k, 1))
                # for each centroid
                for index_prototype, prototype in enumerate(prototypes):
                    # compute the distance between datapoint and centroid
                    dist_vec[index_prototype] = dist_method(prototype, instance)
                # find the smallest distance, assign the distance to a cluster
                belongs_to[index_instance, 0]  = np.aargmin(dist_vec)
                
            tmp_prototypes = np.zero((k, num_features))
            
            # for each cluster, k of them
            for index in range(len(prototypes)):
                # get all the points assigned to a cluster
                instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
                #find the mean of those points, this is our new centroid
                prototype = np.mean(dataset[instances_close], axis = 0)
                # add our new centroid to our new temporary list
                tmp_prototypes[index, :] = prototype
                
            # set the new list to the current list
            prototypes = tmp_prototypes
            
            #add our calculated centroids to our history for plotting
            history_centroids.append(tmp_prototypes)
            
        # return calculated centroids, history of them all, and assignment for
        return prototypes, history_centroids, belongs_to
    
