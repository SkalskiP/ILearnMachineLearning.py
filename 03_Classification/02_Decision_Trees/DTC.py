import numpy as np

test = np.array([0, 0, 1, 1, 0, 0, 1, 1])
test1 = [np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])]
classes = np.array([0, 1])

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

def get_split(labels, classes):
    """
    Select the best split point for a dataset
    """
    # Number of samles in dataset
    n_samples = labels.shape[0]
    
    # Best index and best value of gini index
    b_index, b_gini = None, None

    # Iteration over possible split points
    for i in range(1, n_samples):
        # Pair of arrays after splitting
        spited = np.split(labels, [i])
        # Calculated gini value
        gini_val = gini_index(spited, classes)
        # Checking if we found better split
        if b_gini == None or gini_val < b_gini:
            b_gini = gini_val
            b_index = i
            
    return b_index
        
print(get_split(test, classes))

        