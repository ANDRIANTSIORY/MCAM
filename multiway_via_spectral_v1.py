
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
        result_affinityProp = AffinityPropagation( damping=0.5, random_state=None, preference=None).fit(data)
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
                matrixV = []
                topEig = []

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

                    # store the largest eigenvalue for each slice

                # determine r
                r = max(n_i)
                r_.append(r)

                # we multiply the eigenvalue and eigenvector 

                c_prime = np.zeros((len(e0), len(e0)))

                for k in range(self._dim[0]):
                    v = np.zeros((len(e0[0][1][:,0]),r))
                    for y in range(r):
                        v[:,y] = e0[k][0][y] * e0[k][1][:,y]


                    # v contains the vectors necessary, r eigenvectors multiply with their corresponding eigenvalues
                    matrixV.append(v)

                for t in range(r):
                    Eig = []
                    for k in range(len(e0)):
                        Eig.append(e0[k][0][t])

                    topEig.append(np.max(Eig))   # we have the r max 


                # compute the entry (t1, t2) inside the similarity matrix
                for t1 in range(self._dim[0]):
                    for t2 in range(self._dim[0]):
                        a = np.abs( (matrixV[t1].T).dot(matrixV[t2]) )
                        c_prime[t1,t2] = np.sum(a)

                topEig = np.sum(topEig)

                # normalization of the similarity matrix
                C_prime = c_prime / (topEig**2)


                result.append(self.clustering(C_prime,  self._number[i]))

                store_C_prime.append(C_prime)
                
            

            elif i == 1 :
                e1 = []
                n_i = []
                matrixV = []
                topEig = []
                for k in range(self._dim[1]):
                    horizontale = self.normed_and_covariance(self._tensor[:,k,:])
                    w, v = la.eig(horizontale)
                    w , v = w.real, v.real
                    p = heapq.nlargest(len(w), range(len(w)), w.take) 
                    w = w[p] 
                    e1.append([w[p], v[:,p]]) 

                    # determination of significant drop (determine n_i)
                    n_i.append(self.number_eigenspace(w))

                    # store the largest eigenvalue for each slice

                # determine r
                r = max(n_i)
                r_.append(r)

                # we multiply the eigenvalue and eigenvector 
                c_prime = np.zeros((len(e1), len(e1)))

                for k in range(self._dim[1]):
                    v = np.zeros((len(e1[0][1][:,0]),r))
                    for y in range(r):
                        v[:,y] = e1[k][0][y] * e1[k][1][:,y]

                    # v contains the vectors necessary, r eigenvectors multiply with their corresponding eigenvalues
                    matrixV.append(v)

                for t in range(r):
                    Eig = []
                    for k in range(len(e1)):
                        Eig.append(e1[k][0][t])

                    topEig.append(np.max(Eig))   # we have the r max 


                # compute the entry (t1, t2) inside the similarity matrix
                for t1 in range(self._dim[1]):
                    for t2 in range(self._dim[1]):
                        c_prime[t1,t2] = np.sum( np.abs( (matrixV[t1].T).dot(matrixV[t2]) ) )

                topEig = np.sum(topEig)

                # normalization of the similarity matrix
                C_prime = c_prime / (topEig**2)
                #C_prime = c_prime / np.max(c_prime)

                result.append(self.clustering(C_prime,  self._number[i]))

                store_C_prime.append(C_prime)

                    
                    
            elif i==2 :
                e2 = []
                n_i = []
                matrixV = []
                topEig = []
                for k in range(self._dim[2]):
                    laterale = self.normed_and_covariance(self._tensor[:,:,k])
                    w, v = la.eig(laterale)
                    w , v = w.real, v.real
                    p = heapq.nlargest(len(w), range(len(w)), w.take) # the index of the top self._lim of the eigenvalue w, it is in decreasing order
                    w = w[p]
                    e2.append([w[p], v[:,p]]) 

                    # determination of significant drop (determine n_i)
                    n_i.append(self.number_eigenspace(w))

                    # store the largest eigenvalue for each slice
                    #topEig.append(w[0])

                # determine r
                r = max(n_i)
                r_.append(r)

                # we multiply the eigenvalue and eigenvector 
                c_prime = np.zeros((len(e2), len(e2)))

                for k in range(self._dim[2]):
                    v = np.zeros((len(e2[0][1][:,0]),r))
                    for y in range(r):
                        v[:,y] = e2[k][0][y] * e2[k][1][:,y]

                    # v contains the vectors necessary, r eigenvectors multiply with their corresponding eigenvalues
                    matrixV.append(v)

                for t in range(r):
                    Eig = []
                    for k in range(len(e1)):
                        Eig.append(e1[k][0][t])

                    topEig.append(np.max(Eig))   # we have the r max 


                # compute the entry (t1, t2) inside the similarity matrix
                for t1 in range(self._dim[2]):
                    for t2 in range(self._dim[2]):
                        c_prime[t1,t2] = np.sum( np.abs( (matrixV[t1].T).dot(matrixV[t2]) ) )

                topEig = np.sum(topEig)

                # normalization of the similarity matrix
                C_prime = c_prime / (topEig**2)
                #C_prime = c_prime / np.max(c_prime)

                result.append(self.clustering(C_prime,  self._number[i]))

                store_C_prime.append(C_prime)


        return result,  store_C_prime, r_  # we take C_prime in case we need the similarity matrix to test it with other algorithm


