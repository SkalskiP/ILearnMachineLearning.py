import numpy as np

test = np.array([[0, 0, 1, 1],[0, 0, 1, 1]])
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
        
        score = 0
        
        unique, counts = np.unique(group, return_counts=True)
        frequency = dict(zip(unique, counts))
        
        for single_class in classes:
            if single_class in frequency.keys():
                p = frequency[single_class]/g_samples
                score += p**2
        gini_index += (1 - score) * (g_samples/n_samples)
    return gini_index
        
print(gini_index(test, classes))