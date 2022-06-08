
import scipy.linalg as la
from sklearn.preprocessing import normalize
import numpy as np
import sys
import heapq
from sklearn.cluster import SpectralClustering,AffinityPropagation

# Arguments
# tensor: the datasets 
# k : list of the number of clusters in each mode
# norm : normalization of the matrix slices


class Multiway_via_spectral():
    def __init__(self, tensor, k=[10,10,10], norm = "normalize"):
        self._number = k                            
        self._tensor = tensor
        self._dim = tensor.shape
        self._norm = norm
        self._result, self._c_prime, self._r_ = self.method()


    def get_result(self):    
        return self._result

    def get_c_prime(self):
        return self._c_prime

    def get_r_(self):
        return self._r_

    def normed_and_covariance(self, M):
        if self._norm == "centralize":
            m = np.mean(M, axis=0).reshape(1,-1)  # mean of each column
            M = M - m
            M = (1/len(M[0,:])) * ((M.T).dot(M))
        elif self._norm == "normalize":
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
        result_spectral = SpectralClustering(n_clusters=d,affinity='precomputed', assign_labels='discretize', random_state=0).fit(data)
        result_affinityProp = AffinityPropagation(damping=0.5,random_state=None).fit(data)
        #print(len(set(result_affinityProp.labels_)))
        return result_spectral.labels_, result_affinityProp.labels_




    def method(self):

        l = 3     # nomber of mode inside a 3-order tensor

        result = [] # contain the  clustering labels of the three modes
        store_C_prime = []  # list of the similarity matrices
        r_ = []      # r
        for i in range(3):  
            if i == 0 :
                e0 = []
                n_i = []
                for k in range(self._dim[0]):
                    frontal =  self.normed_and_covariance(self._tensor[k,:,:])
                    w, v = la.eig(frontal)
                    w , v = w.real, v.real
                    p = heapq.nlargest(len(w), range(len(w)), w.take) # the index of the all eigenvalue in decreasing order
                    w = w[p]    # to make it decreasing order
                    e0.append([w[p], v[:,p]]) # e0 is a list, and each element of which is a pair s.t. 
                                                # the first element is the eigenvalue of frontal
                                                # the second is the eigenvector of frontal
                                                # w[1] < w[0], w is sorted in decreasing order

                    # determination of significant drop (determine n_i)
                    n_i.append(self.number_eigenspace(w))

                # determine r

                r = max(n_i)

                r_.append(r)

                # build the matrix C prime
                V = np.zeros((len(e0[0][1][:,0]), len(e0)))  # initialization of the matrix V
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

                    c = (V.T).dot(V) 
                    C_prime += np.abs(c)

                    k += 1

                lambda_max_square = [lam**2 for lam in lambda_max] 
                
                C_prime = C_prime / np.sum(lambda_max_square)
                result.append(self.clustering(C_prime,  self._number[i]))  # result of two clustering  methods (Affinity Propagation and Spectral Clustering)
                store_C_prime.append(C_prime)      #store C_prime


            elif i == 1 :
                e1 = []
                n_i = []
                for k in range(self._dim[1]):
                    horizontale = self.normed_and_covariance(self._tensor[:,k,:])
                    w, v = la.eig(horizontale)
                    w , v = w.real, v.real
                    p = heapq.nlargest(len(w), range(len(w)), w.take) 
                    w = w[p] 
                    e1.append([w[p], v[:,p]]) 

                    n_i.append(self.number_eigenspace(w))

                # determination of r
                r = max(n_i)  
                r_.append(r)

                V = np.zeros((len(e1[0][1][:,0]), len(e1))) 
                C_prime = np.zeros((len(e1), len(e1)))
                
                k = 0
                lambda_max = []
                while (k < r ):
                    Lambda = []
                    s = 0
                    for j in e1:
                        V[:,s] = (j[0][k] * j[1][:,k] )  # product of eigenvalue and eigenvector of the k-th slice
                        Lambda.append(j[0][k])
                        s +=1

                    lambda_max.append(np.max(Lambda))
                    c   = (V.T).dot(V)
                    C_prime += np.abs(c)  

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
                r = max(n_i)  
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

                    k += 1

                lambda_max_square = [lam**2 for lam in lambda_max]
                C_prime = C_prime / np.sum(lambda_max_square)
                result.append(self.clustering(C_prime, self._number[i]))
                store_C_prime.append(C_prime)


        return result,  store_C_prime, r_  # we take C_prime in case we need the similarity matrix to test it with other algorithm


