# SIMPLE LINEAR REGRESSION
# Metoda przyjmuje jako argumenty dwa numpy array'e

import numpy as np


class MySimpleLinearRegression:
    
    def __init__(self, fit_intercept=True):
        # zmienna informująca o tym czy należy obliczać wpółczynnik punktu /
        # przecięcia z osią y
        self.fit_intercept = fit_intercept
    
    def fit(self, X, y, sample_weight=None):
        # weryfikujemy czy dostarczone dane mają odpowiedni format
        # X musi być macierzą o wymiarach (R, 1)
        # Y musi być wektorem o wymiarach (R,)
        
        if X.ndim == 2 and y.ndim == 1 and X.size == y.size:
            # przy pomocy metody najmniejszych kwadratów obliczamy /
            # współczynnik nachylenia prostej coef_
            
            # srednia wartoć wektorów X oraz Y
            X_mean = np.sum(X)/np.size(X)
            y_mean = np.sum(y)/np.size(y)
            
            # licznik
            nominator = 0
            # mianownik
            denominator = 0
            
            for i in range(X.size):
                     
                nominator += (X[i][0] - X_mean) * (y[i] - y_mean)
                denominator += (X[i][0] - X_mean)**2
            
            # współczynnik nachylenia prostej coef_
            self.coef_ = nominator/denominator
            
            if self.fit_intercept:
                # współczynnikpunktu przecięcia z osią y
                self.intercept_ = y_mean - np.dot(X_mean, self.coef_)
                
        else:
            raise ValueError("Matrix size is incorrect")
        
import pandas as pd
dataset = pd.read_csv('Salary_Dataset.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)
my_regressor = MySimpleLinearRegression()
my_regressor.fit(X_train, y_train)