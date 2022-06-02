from sklearn import preprocessing
import scipy.linalg as la
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.decomposition import PCA
import sys
import heapq
from statistics import mode
from sklearn.cluster import KMeans,AgglomerativeClustering,SpectralClustering,DBSCAN,AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import SparsePCA


# I remove the norm 
class Multiway_via_spectral():
    def __init__(self, tensor, k=[10,10,10], norm = "normalized"):
        self._number = k                             #LIM == NUMBER
        self._tensor = tensor
        self._dim = tensor.shape
        self._norm = norm
        self._result, self._c_prime, self._r_ = self.method()


    def get_result(self):     # V^tV
        return self._result

    def get_c_prime(self):
        return self._c_prime

    def get_r_(self):
        return self._r_

    def normed_and_covariance(self, M):
        #M = M / 255   # we add it only for mnist dataset
        if self._norm == "centralized":
            m = np.mean(M, axis=0).reshape(1,-1)  # mean of each column
            M = M - m
            M = (1/len(M[0,:])) * ((M.T).dot(M))
        elif self._norm == "normalized":
            M = normalize(M, axis=0)
            M = (M.T).dot(M)
        else:
            M = (M.T).dot(M)
        M = (M.T).dot(M)
        return M

    def number_eigenspace(self, vector):
        intermediate = [vector[g] - vector[g+1] for g in range(0, len(vector) - 1)]
        return np.argmax(intermediate) + 1   # the index starts with zero (we add 1 to make it from 1)

    def clustering(self, data, d):
        result_spectral = SpectralClustering(n_clusters=d, assign_labels='discretize', random_state=0).fit(data)
        result_affinityProp = AffinityPropagation(random_state=20).fit(data)
        return result_spectral.labels_, result_affinityProp.labels_

    def clusteringMatrix(self, Listd, method = "AgglomerativeClustering", epsilon = 1, min_point = 2):
        data = self._c_prime[1]
        result = []
        for i in range(3):
            if method == "AgglomerativeClustering" :
                result1 = AgglomerativeClustering(n_clusters=Listd[i], affinity='euclidean', linkage='ward').fit(data[i])
                result.append(result1.labels_)

            if method == "SpectralClustering" :
                result1 = SpectralClustering(n_clusters=Listd[i], assign_labels='discretize', random_state=0).fit(data[i])
                result.append(result1.labels_)

            if method == "KMeans" :
                result1 =  KMeans(n_clusters=Listd[i], random_state=0).fit(data[i])
                result.append(result1.labels_)

            if method == "AffinityPropagation" :
                result1 = AffinityPropagation(damping = 0.7, random_state=50).fit(data[i])
                result.append(result1.labels_)

            if method == "DBSCAN" :
                result1 = DBSCAN(eps=epsilon, min_samples=min_point).fit(data[i])
                result.append(result1.labels_)


        return result



    def method(self):

        l = 3     # len(self._dim)  
        # the eigenvalue and eigenvector of each slice for each dimension
        # result use only k-means
        result = [] # contain all the index where is the gap of eigenvalue for each slice ( verification of the sinTheta theorem)
        store_C_prime = []
        first_C_prime = []   # store the C_prime for the top eigenvector
        r_ = []
        for i in range(3):   #  ONLY FOR THE FIRST DIMENSION
            if i == 0 :
                e0 = []
                n_i = []
                for k in range(self._dim[0]):
                    frontal =  self.normed_and_covariance(self._tensor[k,:,:])
                    w, v = la.eig(frontal)
                    transformer = SparsePCA(n_components=2, random_state=0)
                    transformer.fit(frontal)
                    pc = transformer.components_
                    w , v = w.real, v.real
                    p = heapq.nlargest(len(w), range(len(w)), w.take) # the index of the all eigenvalue in decreasing order
                    w = w[p]    # to make it decreasing order
                    v[:,:2] = pc.T
                    w[:2] = 1
                    e0.append([w[p], v[:,p]]) # e0 is a list, and each element of which is a pair s.t. 
                                                # the first element is the eigenvalue of frontal
                                                # the second is the eigenvector of frontal
                                                # w[1] < w[0], w is sorted in decreasing order

                    # determination of significant drop (determine n_i)
                    n_i.append(self.number_eigenspace(w))

                # determine r
                #r = mode(n_i)    # most frequent number in n_i
                r = max(n_i)
                r = 2  # only for mnist data
                r_.append(r)

                # build the matrix C prime
                V = np.zeros((len(e0[0][1][:,0]), len(e0)))  # initialization of matrix V
                C_prime = np.zeros((len(e0), len(e0)))

                k=0   # compteur of lambda_max and r
                lambda_max = [] 
                while (k < r):
                    Lambda = []
                    s = 0     # compteur of the column of matrix V
                    for j in e0:
                        V[:,s] = (j[0][k] * j[1][:,k] )  # product of eigenvalue and eigenvector of slice k
                        Lambda.append(j[0][k])
                        s +=1
    
                    lambda_max.append( np.max(Lambda) )  # the max among the lambda
                    # lambda_max.append( np.argmax(Lambda) )    # index of the maximum among the lambda
                    #V = V / lambda_max[k]
                    c = (V.T).dot(V) 
                    C_prime += np.abs(c)

                    #--the top eigenvector-----
                    if k==0 :
                        first_C_prime.append(C_prime / np.max(Lambda))
                    #--------------------------
                    k += 1

                lambda_max_square = [lam**2 for lam in lambda_max] 
                
                C_prime = C_prime / np.sum(lambda_max_square)
                result.append(self.clustering(C_prime,  self._number[i]))  # result of two methods of clustering
                store_C_prime.append(C_prime)      #store C_prime


            elif i == 1 :
                e1 = []
                n_i = []
                for k in range(self._dim[1]):
                    horizontale = self.normed_and_covariance(self._tensor[:,k,:])
                    w, v = la.eig(horizontale)
                    w , v = w.real, v.real
                    p = heapq.nlargest(len(w), range(len(w)), w.take) # the index of the top self._lim of the eigenvalue w, it is in decreasing order
                    w = w[p] 
                    e1.append([w[p], v[:,p]]) # the same structure as e0

                    # determination of significant drop (determine n_i)
                    n_i.append(self.number_eigenspace(w))

                # determine r
                r = max(n_i)  # most frequent number in n_i
                r_.append(r)

                V = np.zeros((len(e1[0][1][:,0]), len(e1))) 
                C_prime = np.zeros((len(e1), len(e1)))
                
                k = 0
                lambda_max = []
                while (k < r ):
                    Lambda = []
                    s = 0
                    for j in e1:
                        V[:,s] = (j[0][k] * j[1][:,k] )  # product of eigenvalue and eigenvector of slice k
                        Lambda.append(j[0][k])
                        s +=1

                    lambda_max.append(np.max(Lambda))
                    c   = (V.T).dot(V)
                    C_prime += np.abs(c)  # abs(C)

                    #--the top eigenvector-----
                    if k==0 :
                        first_C_prime.append(C_prime / np.max(Lambda))
                    #--------------------------
                    k += 1

                lambda_max_square = [lam**2 for lam in lambda_max]
                
                C_prime = C_prime / np.sum(lambda_max_square)
                result.append(self.clustering(C_prime, self._number[i]))
                store_C_prime.append(C_prime)

                    
            elif i==2 :
                e2 = []
                n_i = []
                for k in range(self._dim[2]):
                    laterale = self.normed_and_covariance(self._tensor[:,:,k])
                    w, v = la.eig(laterale)
                    w , v = w.real, v.real
                    p = heapq.nlargest(len(w), range(len(w)), w.take) # the index of the top self._lim of the eigenvalue w, it is in decreasing order
                    w = w[p]
                    e2.append([w[p], v[:,p]]) 

                    # determination of significant drop (determine n_i)
                    n_i.append(self.number_eigenspace(w))

                # determine r
                r = max(n_i)  # most frequent number in n_i
                r_.append(r)

                V = np.zeros((len(e2[0][1][:,0]), len(e2))) 
                C_prime = np.zeros((len(e2), len(e2)))
             
                k = 0
                lambda_max = []
                while (k < r):
                    Lambda = []
                    s = 0
                    for j in e2:
                        V[:,s] = (j[0][k] * j[1][:,k] )  # product of eigenvalue and eigenvector of slice k
                        Lambda.append(j[0][k])
                        s += 1

                    lambda_max.append(np.max(Lambda))
                    c = (V.T).dot(V) 
                    C_prime += np.abs(c)    # abs(C)

                    #--the top eigenvector-----
                    if k==0 :
                        first_C_prime.append(C_prime / np.max(Lambda))
                    #--------------------------
                    k += 1

                lambda_max_square = [lam**2 for lam in lambda_max]
                C_prime = C_prime / np.sum(lambda_max_square)
                result.append(self.clustering(C_prime, self._number[i]))
                store_C_prime.append(C_prime)


        return result, (first_C_prime, store_C_prime), r_  # we take C_prime in case we need the similarity matrix to test it with other algorithm


