import numpy as np

class Node():
    """
    Supporting class used to build decision tree.
    """
    def __init__(self, label = None, dimention = None, value = None):
        # If this node is leaf this field wil hold label of class that is predicted
        self.label = label
        # Index of feature that we need to check
        self.dim = dimention
        # Value of featue that we need to chceck
        self.value = value
        # Left leaf
        self.left = None
        # Right leaf
        self.right = None
        
    def __str__(self):
        if self.label == None:
            return("Is value of X" + str(self.dim) + " > then " + str(self.value))
        else:
            return("Label: " + str(self.value))
    
class MyDecisionTreeClassifier():
    """
    max_depth : The maximum depth of the tree. If None, then nodes are expanded until
                all leaves are pure or until all leaves contain less than
                min_samples_split samples.
    
    min_samples_split : The minimum number of samples required to split an internal node.
    """
    def __init__(self, 
                 max_depth = 100, 
                 min_samples_split=2):
        
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        
    def fit(self, X, y):
        """
        Method to build a decision tree classifier from the training set (X, y).
        
        X : Array of features values; shape = [n_samples, n_features]
        
        y : Array of labels; shape = [n_samples]
        """
        
        # X and y need to have the same number of samples
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y need to be equal.")
        
        # Finding and saving all possible class labels
        self.classes_ = np.unique(y)
        
        self.X = X
        self.y = y

        # Root of decision tree
        self.tree_ = Node()
        # Building a tree (fitting data)
        self.split(self.X, self.y, self.tree_, 1)
        
        
    def split(self, features, labels, node, depth):
        """
        Create child splits for a node or make terminal.
        """
        
        # If set of samples is too small
        # Or node depth is to big
        # We take as label the most frequent class
        if(labels.shape[0] <= self.min_samples_split or depth >= self.max_depth):
            unique, counts = np.unique(labels, return_counts=True)
            # Finding most frequent class in data set
            node.label = unique[np.argmax(counts)]
            return
        
        # Finding best split
        # split_dim : index of feature used in current slit
        # split_value : limit value in current slit
        # spl_labels : groups of labels createdafter splitting
        # spl_features : groups of features createdafter splitting
        split_dim, split_value, spl_labels, spl_features = get_split(features, labels, self.classes_)
        
        # Building children
        node.left = Node()
        node.right = Node()
 
        # Assigning values to node fields
        node.value = split_value
        node.dim = split_dim
        
        # Recursive fitting        
        self.split(spl_features[0], spl_labels[0], node.left, depth+1)
        self.split(spl_features[1], spl_labels[1], node.right, depth+1)
        
    def predict(self, X_test):
        """
        Method used for label prediction for multiple samples
        """
        # Allocating memory for predictions array
        predictions = np.empty(X_test.shape[0])
        # Looping over samples in test set
        for index, sample in enumerate(X_test):
            predictions[index] = single_prdict(sample, self.tree_)
    
        return predictions
        
def single_prdict(features, node):
    """
    Function goes through decision tree and finds label value of given set of features.
    """
    # Going through decision tree until we find node with defined label.
    while node.label == None:
        if features[node.dim] > node.value:
            node = node.right
        else:
            node = node.left
    return node.label

    
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
    b_dim, b_value, b_gini, array_1, array_2 = None, None, None, None, None

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
                b_dim = d
                b_value = (features_sorted[i, d] + features_sorted[i-1, d])/2
                array_1 = spited
                array_2 = np.split(features_sorted, [i])
            
    return b_dim, b_value, array_1, array_2

import pandas as pd
dataset = pd.read_csv('../../00_Datasets/Iris.csv')

feature_columns = ['SepalLengthCm', 'PetalWidthCm']
X = dataset[feature_columns].values
y = dataset['Species'].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


labels = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 0])
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
   
# print(get_split(features, labels, classes))

Test = MyDecisionTreeClassifier()
Test.fit(X_train, y_train)
# print(Test.tree_.left.value)
# print(Test.tree_.right.value)
y_pred_train = Test.predict(X_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred_train)
print(cm)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
plt.figure(figsize=(12,8))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
sns.set_style("whitegrid")

# tworzymy siatke pixeli, która posłuży do wizualizacji klastrów
X1, X2 = np.meshgrid(np.arange(start = X_train[:,0].min() - 1, stop = X_train[:, 0].max() + 1, step = 0.02),
                     np.arange(start = X_train[:,1].min() - 1, stop = X_train[:, 1].max() + 1, step = 0.02))

# wizualizacja pól klastrów
plt.contourf(X1, X2, Test.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.2, cmap = plt.cm.summer)

# granice wykresów
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap = plt.cm.summer)

plt.show()