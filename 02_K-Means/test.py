class MyKMeans():
    
    def __init__(self, n_clusters=8, init='random', n_init=10, max_iter=300, tol=0.0001):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        
        
    def k_means(self, X, n_clusters=8, init='random', n_init=10, max_iter=300, tol=0.0001):
        
        # ilosc niezaleznych wykonan algorytmu musi byc wieksza od 0
        if n_init <= 0:
            raise ValueError("Invalid number of initializations."
                         " n_init=%d must be bigger than zero." % n_init)
        
        # maksymalna ilosc obrotow petli bez osiagniecia zbieznosci musi byc wieksza od 0
        if max_iter <= 0:
            raise ValueError('Number of iterations should be a positive number,'
                             ' got %d instead' % max_iter)
        
        best_labels, best_inertia, best_centers = None, None, None
        
        for i in range(n_init):
            labels, inertia, centers, n_iter_ = kmeans_single(X, n_clusters, max_iter=max_iter, init=init, tol=tol)
        
        if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_
                
        return best_centers, best_labels, best_inertia, best_n_iter
    
    def fit(self, X):
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = self.k_means(X=X, 
                                                                                   n_clusters=self.n_clusters,
                                                                                   init = self.init,
                                                                                   n_init = self.n_init,
                                                                                   max_iter = self.max_iter,
                                                                                   tol = self.tol)
        




###############################################################################

def kmeans_single(X, n_clusters, max_iter=300, init='random', tol=1e-4):
    """
    Implementacja pojedynczego przebiegu algorytmu na KMeans
    The Lloyd Algorithm
    """

    best_labels, best_inertia, best_centers = None, None, None
    
    # Inicjacja centroid losowymi punktami 
    centers = init_centroids(X, n_clusters, init)

    # Iteracja wykona się maksymalnie tyle razy ile podano w argumentach funkcji
    for i in range(max_iter):
        # Tworzymy nowy array do którego kopiujemy stare centoidy
        centers_old = centers.copy()
        
        # labels assignment is also called the E-step of EM
        labels, inertia = \
            labels_update(X, centers)

        
        centers = centers_update(X, labels, n_clusters)


        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = squared_norm(centers_old - centers)
        if center_shift_total <= tol:

            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = labels_update(X, centers)

    return best_labels, best_inertia, best_centers, i + 1

##############################################################################

def init_centroids(X, k, init):
    """
    Funkcja zwracająca listę losowo wylosowanych centroid.
    Wybrane centroidy należą do listy wszystkich wprowadzonych punków.
    
    X : array, shape (n_samples, n_features)
    k : poszukiwana iloć klastrów
    init : tryb wyszukiwania centroid
    
    """
    
    n_samples = X.shape[0]
    
    if isinstance(init, str) and init == 'random':
        seeds = np.random.permutation(n_samples)[:k]
        centers = X[seeds]
    
    return centers

##############################################################################
    
def labels_update(X, centers):
    
    labels, mindist = min_distances(X, centers)
    # cython k-means code assumes int32 inputs
    labels = labels.astype(np.int32)
    inertia = mindist.sum()
    return labels, inertia

##############################################################################
# UPROSZCZONA WERSJA pairwise_distances_argmin_min
    
def min_distances(X, Y):
    # Allokacja pamieci dla arrayów wyjsciowych
    # Labels będzie przechowywać etykiety z numerem kalstr do ktorego zostal przyparzadkowany punkt
    labels = np.empty(X.shape[0], dtype=np.intp)
    # Distancs bedzie przechowywało odleglosc pomiedzy punktem a centroida klastra
    distances = np.empty(X.shape[0])
    
    # Ilosc probek
    n_samples = X.shape[0]
    # Ilosc kalstrow
    n_clusters = Y.shape[0]
    
    # Iteracja po punktach 
    for i in range(n_samples):
        # Obecnie przyznany klaster
        cur_claster = None
        # Odległoć do obecnie przyznanego klastra
        cur_distance = float('inf')
        
        # Iteracja po klastrach
        for j in range(n_clusters):
            # Obliczenie normy euklidesowej dla analiowanego pnktu oraz centroidy
            temp_distance = (X[i]-Y[j]).dot(X[i]-Y[j])
            if temp_distance < cur_distance:
                cur_claster = j
                cur_distance = temp_distance
                
        labels[i] = cur_claster
        distances[i] = cur_distance
        
    return labels, distances
            

###############################################################################
# UPROSZCZONA WERSJA _k_means._centers_dense
    
def centers_update(X, labels, n_clusters):
    # Ilosc probek
    n_samples = X.shape[0]
    # Ilosc parametrów
    n_features = X.shape[1]
    
    # Allokacja pamięci dla wektora który będzie przechowywał współrzędne centroid
    centers = np.zeros((n_clusters, n_features), dtype=np.float64)
    
    # Wektor opisujący liczebnoć każdego klastra
    n_samples_in_cluster = points_in_clusters(labels, n_clusters)
    
    for i in range(n_samples):
        for j in range(n_features):
            centers[labels[i], j] += X[i, j]
            
    for i in range(n_clusters):
        for j in range(n_features):
            centers[i, j] /= n_samples_in_cluster[i]
            
    return centers


###############################################################################
    
def points_in_clusters(labels, n_clusters):
    # Allokacja pamięci dla numpy arraya któRy będzie przchowywał liczeboci klastrów
    cluster_sizes = np.zeros(n_clusters, dtype=np.float64)
    
    for point in labels:
        cluster_sizes[point] += 1
        
    return cluster_sizes

###############################################################################

import pandas as pd
import numpy as np
from sklearn.utils.extmath import squared_norm

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

claster = MyKMeans(n_clusters=3)
claster.fit(X)
print(claster.cluster_centers_)