import numpy as np
import pandas as pd
import scipy as sp

class MyKNN():
    
    def __init__(self, n_neighbors=5):
        self.n_neighbors=n_neighbors
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        
    def predict(self, X_test):
        
        # number of predictions to make and number of features inside single sample
        n_predictions, n_features = X_test.shape
        
        # allocationg space for array of predictions
        predictions = np.empty(n_predictions, dtype=int)
        
        # loop over all observations
        for i in range(n_predictions):
            predictions[i] = single_prediction(self.X, self.y, X_test[i, :], self.n_neighbors)

        return(predictions)

            
def single_prediction(X, y, x_train, k):
    
    # number of samples inside training set
    n_samples = X.shape[0]
    
    # create array for distances and targets
    distances = np.empty(n_samples, dtype=np.float64)
    targets = np.empty(k, dtype=int)
    
    for i in range(n_samples):
        distances[i] = (x_train - X[i]).dot(x_train - X[i])
        
    distances = sp.c_[distances, y]
    sorted_distances = distances[distances[:,0].argsort()]
    targets = sorted_distances[0:k,1]
    unique, counts = np.unique(targets, return_counts=True)
   
    return(unique[np.argmax(counts)])
    


# Importing the dataset
dataset = pd.read_csv('../../00_Datasets/Iris.csv')

feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = dataset[feature_columns].values
y = dataset['Species'].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier = MyKNN(n_neighbors=10)
classifier.fit(X_train, y_train)
print(classifier.predict(X_test))

from sklearn.neighbors import KNeighborsClassifier
# Instantiate learning model (k = 10)
classifier = KNeighborsClassifier(n_neighbors=10)

# Fitting the model
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred)