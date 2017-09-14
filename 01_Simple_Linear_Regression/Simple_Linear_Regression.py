# SIMPLE LINEAR REGRESSION

class MySimpleLinearRegression:
    
    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=1):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.n_jobs = n_jobs
    
    def fit(self, X, y, sample_weight=None):
        if(len(X) != len(y)):
            raise ValueError("X and y should have equal length")
        else:            
            # mean values of lists X and y
            X_mean = sum(X)/len(X)
            y_mean = sum(y)/len(y)
            
            nominator = 0
            denominator = 0
            
            for X_i, y_i in zip(X, y):
                nominator += (X_i - X_mean)*(y_i - y_mean)
                denominator += (X_i - X_mean)**2
            
            self.coef_ = nominator/denominator
            
            if(self.fit_intercept):
                self.intercept_ = y_mean - self.coef_ * X_mean
                
    def predict(self, X):
        return [self.intercept_ + self.coef_ * x for x in X]
        
"""               
X = [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 
     3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0, 
     6.8, 7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3, 10.5]
Y = [39343.0, 46205.0, 37731.0, 43525.0, 39891.0, 56642.0,
     60150.0, 54445.0, 64445.0, 57189.0, 63218.0, 55794.0,
     56957.0, 57081.0, 61111.0, 67938.0, 66029.0, 83088.0,
     81363.0, 93940.0, 91738.0, 98273.0, 101302.0, 113812.0,
     109431.0, 105582.0, 116969.0, 112635.0, 122391.0, 121872.0]

test = MySimpleLinearRegression()
test.fit(X, Y)
print(test.coef_)
print(test.intercept_)
print(test.predict([5, 2]))
"""
