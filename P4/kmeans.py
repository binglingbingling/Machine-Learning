import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
            Note: Number of iterations is the number of time you update the assignment
        ''' 
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE


        
        means_ind = np.random.choice(N, self.n_cluster, replace=True)
        means = x[means_ind]
        J = np.inf
        x_mu = np.zeros((N, self.n_cluster))
        membership = np.zeros((N))
        
        for i in range(1,self.max_iter+1):
            r_ik = np.zeros((N, self.n_cluster))
            number_of_updates = i
            for j in range(self.n_cluster):
                x_mu[:, j] = np.sum(np.square(means[j] - x), axis=1)
            membership = np.argmin(x_mu, axis=1)
            r_ik[np.arange(N), membership] = 1
            Jnew = 0
            for i in range(N):
                x_mu_i = means[np.argmax(r_ik[i])] - x[i]
                Jnew += np.dot(x_mu_i.T,x_mu_i)
            Jnew = Jnew/N
            if abs(J-Jnew) <= self.e:
                break
            J = Jnew
            means = np.array([np.mean(x[membership==k], axis=0) for k in range(self.n_cluster)])
        return means, membership, number_of_updates


        # DONOT CHANGE CODE BELOW THIS LINE

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        kmeans = KMeans(self.n_cluster, self.max_iter, self.e)
        centroids, membership, number_of_updates = kmeans.fit(x)     
        centroid_labels = np.zeros((self.n_cluster))
        for k in range(self.n_cluster):
            votes = np.bincount(y[np.where(membership == k)])
            centroid_labels[k] = np.argmax(votes)
        self.centroids = centroids


        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE

        dist_norm = np.linalg.norm(x - np.expand_dims(self.centroids, axis=1), axis=2)
        labels = self.centroid_labels[np.argmin(dist_norm, axis=0)]
        # DONOT CHANGE CODE BELOW THIS LINE
        return labels

