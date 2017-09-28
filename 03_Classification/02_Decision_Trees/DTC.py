import numpy as np

class Tree():
    
    def __init__(self, label = None, dimention = None, value = None):
        # If this node is leaf this field wil hold label of class that is predicted
        self.label = label
        # Index of feature that we need to check
        self.dimention = dimention
        # Value of featue that we need to chceck
        self.value = value
        # Left leaf
        self.left = None
        # Right leaf
        self.right = None

labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
classes = np.array([0, 1])
features = np.array(
    [[  2.77124472,   1.78478393],
     [  1.72857131,   1.16976141],
     [  3.67831985,   2.81281357],
     [  3.96104336,   2.61995032],
     [  2.99920892,   2.20901421],
     [  7.49754587,   3.16295355],
     [  9.00220326,   3.33904719],
     [  7.44454233,   0.47668338],
     [ 10.12493903,   3.23455098],
     [  6.64228735,   3.31998376]])

def gini_index(groups_label, classes):
    """
    Method to calculate Gini Index for split dataset.
    """
    
    # Number of samples 
    n_samples = sum([group.shape[0] for group in groups_label])
    # Initiation of variable to hold Gini index
    gini_index = 0
    # Iteration over data groups
    for group in groups_label:
        # Number of samples in group
        g_samples = group.shape[0]
        # Avoid division by 0
        if g_samples == 0:
            continue
        
        # Temp variable to hold score for single group of samples
        score = 0
        
        # Getting list of classes that exist in dataset
        # Along with their counts
        unique, counts = np.unique(group, return_counts=True)
        # Bulding dictionary combining class label with its count
        frequency = dict(zip(unique, counts))
        
        # Iteration over class labels
        for single_class in classes:
            # Checking if class label exist in samles group
            if single_class in frequency.keys():
                # Calculation of probability
                p = frequency[single_class]/g_samples
                score += p**2
        gini_index += (1 - score) * (g_samples/n_samples)
    return gini_index

def get_split(features, labels, classes):
    """
    Select the best split point for a dataset
    """
    # Number of samles in dataset
    n_samples = labels.shape[0]
    
    # Number of features in dataset
    n_features = features.shape[1]
    
    # Best index and best value of gini index
    b_dim, b_index, b_value, b_gini, array_1, array_2 = None, None, None, None, None, None

    # Iteration over dimention
    for d in range(0, n_features):
        #Sorting features and labels by values of d-th feature
        order = features[:, d].argsort()
        features_sorted = features[order]
        labels_sorted = labels[order]

        # Iteration over possible split points
        for i in range(1, n_samples):
            # Pair of arrays after splitting
            spited = np.split(labels_sorted, [i])
            # Calculated gini value
            gini_val = gini_index(spited, classes)
            # Checking if we found better split
            if b_gini == None or gini_val < b_gini:
                b_gini = gini_val
                b_index = i
                b_dim = d
                b_value = features_sorted[i, d]
                array_1 = spited
                array_2 = np.split(features_sorted, [i])
            
    return b_gini, b_dim, b_index, b_value, array_1, array_2
        
print(get_split(features, labels, classes))

