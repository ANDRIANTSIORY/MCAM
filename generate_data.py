import numpy as np
import itertools

# The tensor of size  (n1,n2,m) 
# m : number of times 
# n1 : number of individuals
# n2 : number of features
# k1 : cluster size inside mode-1
# k2 : cluster size inside mode-2
# k3 : cluster size inside mode-3
# cluster : the nomber of cluster in each mode, we assume that the three modes have the same number of clusters
# sigma1 : signal strength for the "first data"
# k : number of clusters in each mode for the "second data"


class Generate_tensorBiclustering():
    def __init__(self, m=0, n1=0, n2=0, k1=1, k2=1, k3=1, cluster = 1, gamma = 1, k = 0):
        self._m = m
        self._n1 = n1
        self._n2 = n2
        self._k1 = k1
        self._k2 = k2
        self._k3 = k3
        self._gamma = gamma
        self._k = k
        self._cluster = cluster



# data with two blocks of triclustering inside
# tensor decomposed into rank two


    def first_data(self):

        T = np.zeros((self._m, self._n1, self._n2))
        r = 0 # counter
        # create a random vector 
        # first rank
        for s in range(self._cluster):
            v = np.zeros((self._m))
            u = np.zeros((self._n1))
            w= np.zeros((self._n2))
            X = np.zeros((self._m,self._n1,self._n2))

            for i in range(r, r+(self._k3)):
                v[i] = 1/np.sqrt( self._k3)
                #print(r+(weight * self._k3))

            for i in range(r, r+ ( self._k1)):
                u[i] = 1/np.sqrt( self._k1)

            for i in range(r, r+ ( self._k2)):
                w[i] = 1/np.sqrt( self._k2)

            for i in range(self._m):
                for j in range(self._n1):
                    for k in range(self._n2):
                        X[i,j,k] = self._gamma * v[i]*u[j]*w[k]

            T += X
            r += max(self._k1,  self._k2,  self._k3)
            

        return T + np.random.normal(0, 1,  size= (self._m, self._n1, self._n2))
 

    # MULTI-WAY CLUSTERING VIA TENSOR BLOCK MODEL

    # Generate a non-sparse data as defined in the Multiway clustering via TBM
    # the function takes two additional two parameter
    # the block mean, uniform distribution [-u,u] 
    # and the variance of the gaussian noise model  N(0, sigma)
    # sigma is the standard deviation of the gaussian random variable

    def second_data(self, u, v, sigma):
        T = np.zeros((self._m, self._m, self._m))
        partition = [i for i in range(self._m)]
        partition = np.array_split(partition, self._k)
        partition = [list(i) for i in partition]
        for i in partition:
            for j in partition:
                for k in partition:
                    a = np.random.uniform(u, v, 1)        # u < v 
                    for l in list(itertools.product(i,j,k)):
                        T[l[0],l[1],l[2]] = a

        Z = np.random.normal(0, sigma, ((self._m, self._m, self._m)))    # N(0, sigma)

        return T + Z





        