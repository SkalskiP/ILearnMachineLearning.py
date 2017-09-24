import numpy as np
from sklearn.utils.extmath import squared_norm

class MyKMeans():
    """
    Klasa obsługująca wykonanie algorytmu KMeans.
    
    Konstruktor przyjmuje następujące parametry:
        n_clusters : liczba klastrów na jaką chcemy podzielić zbiór danych
        
        init : sposób wstępnego wyboru centroid
        
        n_init : ilość przejść prze algorytm kmeans - ze względu na fakt,
        że algorytm kmeans pozwala jedynie na odnalezienie lokalnego minimum 
        funkcji inercji, a jej ostateczna wartoć jest silnie zależna od wstępnego
        wyboru centroid, algorytm kmeans uruchamiany jest wielokrotnie co
        pozwala w znaczącym stopniu zminimalizować wpływ losowości na końcowy wynik
        
        max_iter : maksymalna dopuszczalna ilość iteracji w ramach pojedyńczego
        przejcia przez algorytm kmeans; jeżeli algorytm nie osiagnie zbieżności
        jego działanie zostanie przerwane po max_iter przejściach
        
        tol : maksymalna akceptowalna różnca pomiędzy wartością inercji w następujących
        po sobie przejściach algorytmuj jeżeli różnica inercji będzie mniejsza od tol,
        uznajemy, że algorytm osiągnął zbieżność
    """
    
    def __init__(self, n_clusters=8, init='random', n_init=10, max_iter=300, tol=0.0001):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        
    def fit(self, X):
        """
        Metoda, która w oparciu o parametry podane w konstruktorze klasy oblicza:
            
            - współrzędne centroid poszczególnych klastrów
            
            - nadaje etykiety punktom zbioru X przyporzadkowując je do odpowiednich
            klastrów
            
            - oblicza lokalne minimum funkcji inercji
            
            - oblicza liczbę iteracji koniecznych do osiągnięcia zbieżności
            
            Metoda przyjmuje następujące parametry:
                X : zbiór pomiarów w postaci numpy array o wymairach [liczba próbek,
                liczba cech]
        """
        
        # weryfikacja czy liczba próbek zawartych w zbiorze X jest wieksza od poszukiwanej liczby klastrów
        if X.shape[0] < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                X.shape[0], self.n_clusters))
        
        # egzakucja algorytmu kmeans z uwzględnieniem parametrów podanych w konstruktorze
        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = k_means(X=X, 
                                                                                   n_clusters=self.n_clusters,
                                                                                   init = self.init,
                                                                                   n_init = self.n_init,
                                                                                   max_iter = self.max_iter,
                                                                                   tol = self.tol)
        
# =============================================================================
        
def k_means(X, n_clusters=8, init='random', n_init=10, max_iter=300, tol=0.0001):
    """
    Ze względu na znaczący wpływ czynnika losowego na otrzymane wyniki, algorytm
    kmeans należy wykonać kilkukrotnie. Niniejsza funckcja odpowiada za przeprowadzenie 
    n_init prób, oraz wybór najlepszego wyniku to znaczy takiego, w którym wartość
    inercji jest możliwie najniższa. 
    """
    
    # ilość przejść przez algorytm musi byc wieksza od 0
    if n_init <= 0:
        raise ValueError("Invalid number of initializations."
                     " n_init=%d must be bigger than zero." % n_init)
    
    # maksymalna ilość ideracji w ramach jednego przejścai przez algorytm kmeans musi byc wieksza od 0
    if max_iter <= 0:
        raise ValueError('Number of iterations should be a positive number,'
                         ' got %d instead' % max_iter)
    
    # inicjacja zmiennych przechowujących etykiety, wartość inercji oraz współrzędne centroid dla "najlepszej" próby 
    best_labels, best_inertia, best_centers = None, None, None
    
    # wykonanie n_init prób algorytmu kmeans
    for i in range(n_init):
        labels, inertia, centers, n_iter_ = kmeans_single(X=X, 
                                                          n_clusters = n_clusters, 
                                                          max_iter=max_iter, 
                                                          init=init, 
                                                          tol=tol)
        
        # porównanie wyników z ostatniej próby z najlepszym dotychczas otrzymanym wynikiem
        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia
            best_n_iter = n_iter_
            
    return best_centers, best_labels, best_inertia, best_n_iter

# =============================================================================

def kmeans_single(X, n_clusters, max_iter=300, init='random', tol=1e-4):
    """
    Implementacja pojedynczego przejcia przez algorytm Lloyd'a na KMeans
    """

    # inicjacja zmiennych przechowujących etykiety, wartość inercji oraz współrzędne centroid dla ostatniej próby  
    best_labels, best_inertia, best_centers = None, None, None
    
    # inicjacja centroid
    centers = init_centroids(X, n_clusters, init)

    # wykonanie maksymalnie max_iter iteracji w celu osiągniecia zbieżności
    for i in range(max_iter):
        # tworzymy nowy array do którego kopiujemy stare centoidy
        centers_old = centers.copy()
        
        # nadanie pomiarom odpowiednich etykiet oraz obliczenie wartości inercji
        labels, inertia = labels_update(X, centers)

        # obliczenie lokalizacji nowych centroid
        centers = centers_update(X, labels, n_clusters)

        # jeżeli wartości inercji otrzymana w ostatniej iteracji jest najmniejsza
        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        # obliczenie wartości zbierznosci centroid w dwóch kolenych iteracjach
        center_shift_total = squared_norm(centers_old - centers)
        # przerwanie działania algorytmu - osiągnieto zbieżność
        if center_shift_total <= tol:
            break
    
    # jeżeli nie osiągnieto zbieżności przerywamy algorytm
    if center_shift_total > 0:
        # obliczamy etykiety, wartość inercji dla ostatcznych współrzędnych centroid
        best_labels, best_inertia = labels_update(X, centers)

    return best_labels, best_inertia, best_centers, i + 1

# =============================================================================

def init_centroids(X, k, init = 'random'):
    """
    Funkcja zwracająca numpy array, k losowo wybranych centroid.
    Wybrane centroidy należą do przekazanego zbioru wszystkich punków X.
    W zależności od wartości zmiennej init zostaje wybrany inny algorytm
    wyboru centroid:
        - "random" - prowadzi do całkowicie losowego wyboru centroid
        - "kmeans++" - ulepszony sposób wyboru centroid skutkujący lepszą zbierznocią
        algorytmu oraz lepszym doborem klastrów
    
    X : array, shape (n_samples, n_features)
    k : poszukiwana iloć klastrów
    init : tryb wyszukiwania centroid
    
    """
    
    # ilość próbek w przekazanym zbiorze danych
    n_samples = X.shape[0]
    
    # losowy wbór centroid
    if isinstance(init, str) and init == 'random':
        # array zawierający losowo wybrane indeksy punktów
        seeds = np.random.permutation(n_samples)[:k]
        # array zawierający losowo wybrane punkty ze zbioru X
        centers = X[seeds]
        
    # wybór centroid za pomocą algorytmu kmeans++
    if isinstance(init, str) and init == 'kmeans++':
        # array zawierający losowo wybrane punkty ze zbioru X
        centers = k_means_pp(X, k)
        
    return centers

# =============================================================================

def k_means_pp(X, k):
    """
    Implementacja algorytmu kmeans++, pozwwalacjącego na lepszą inicjację centroid,
    a w efekcie na lepszą średnią zbierzność algorytmu kmeans.
    """
    pass

# =============================================================================
    
def labels_update(X, centers):
    """
    Funkcja zwraca numer id klastra do którego należą koeljne pomiary oraz
    wartość inercji dla obecnego układu klastrów
    """
    
    # nadanie punktom etykiet 
    labels, min_dist = min_distances(X, centers)
    # obliczenie funkcji inercji dla obecnego układu klastrów
    # inercja jest w istocie sumą bezwładnoci wszystkich klastróW
    inertia = min_dist.sum()
    return labels, inertia

# =============================================================================
    
def min_distances(X, Y):
    """
    Uproszczona wersja pairwise_distances_argmin_min z biblioteki sklearn.
    Funkcja zwraca numer id klastra do którego należą koeljne pomiary oraz
    ich odległość do środka tego kalstra.
    """
    # allokacja pamieci dla arrayów wyjsciowych
    # labels będzie przechowywać etykiety z numerem kalstr do ktorego zostal przyparzadkowany punkt
    labels = np.empty(X.shape[0], dtype=np.intp)
    # distancs bedzie przechowywało odleglosc pomiedzy punktem a centroida klastra
    distances = np.empty(X.shape[0])
    
    # ilosc probek
    n_samples = X.shape[0]
    # ilosc kalstrow
    n_clusters = Y.shape[0]
    
    # iteracja po punktach 
    for i in range(n_samples):
        # obecnie przyznany klaster
        cur_claster = None
        # odległoć do obecnie przyznanego klastra
        cur_distance = float('inf')
        
        # iteracja po klastrach
        for j in range(n_clusters):
            # obliczenie normy euklidesowej dla analiowanego pnktu oraz centroidy
            temp_distance = (X[i]-Y[j]).dot(X[i]-Y[j])
            if temp_distance < cur_distance:
                cur_claster = j
                cur_distance = temp_distance
                
        labels[i] = cur_claster
        distances[i] = cur_distance
        
    return labels, distances
            
# =============================================================================
    
def centers_update(X, labels, n_clusters):
    """
    Uproszczona wersja _k_means._centers_dense z biblioteki sklearn.
    Funkcja zwraca zaktualizowaną lokalizacje centroid.
    """
    # ilosc probek
    n_samples = X.shape[0]
    # ilosc parametrów
    n_features = X.shape[1]
    
    # allokacja pamięci dla wektora który będzie przechowywał współrzędne centroid
    centers = np.zeros((n_clusters, n_features), dtype=np.float64)
    
    # wektor opisujący liczebnoć każdego klastra
    n_samples_in_cluster = points_in_clusters(labels, n_clusters)
    
    for i in range(n_samples):
        for j in range(n_features):
            centers[labels[i], j] += X[i, j]
            
    for i in range(n_clusters):
        for j in range(n_features):
            centers[i, j] /= n_samples_in_cluster[i]
            
    return centers


# =============================================================================
    
def points_in_clusters(labels, n_clusters):
    """
    Funkcja oblicza liczebność każdego klastra
    """
    # allokacja pamięci dla numpy arraya który będzie przchowywał liczeboci klastrów
    cluster_sizes = np.zeros(n_clusters, dtype=np.float64)
    
    for point in labels:
        cluster_sizes[point] += 1
        
    return cluster_sizes
