import numpy as np
from tensorly.base import unfold
from sklearn.cluster import KMeans
import itertools
from tensorly.tenalg import mode_dot
from tensorly import norm


# input: the data and the rank of the cluster 
# Data has tensor structure
# rank is the nomber of block for each mode of the tensor
class MultiwayClusteringTBM:
    
    def __init__(self, data, R, iteration=15, sparse = False, rho = 0, ambda=1):
        self.data = data
        self.R = R
        self.iteration = iteration
        self.sparse = sparse
        self.rho = rho
        self.ambda = ambda
        self.dataShape = data.shape
        
        self.initiateMembershipMatrix()
        
        for i in range(self.iteration):
            self.updateCoreTensor()
            self.updateMembershipMatrix()

    
    def get_core(self):
        return self.core
    
    def get_membershipMatrix(self):
        return self.matrices

    # partition one mode n mumbership matrix to identify the cluster
    def get_cluster(self):
        C = []
        for c in range(len(self.R)):
            cluster = []
            for k in range(self.R[c]):
                cluster.append([i for i , j in enumerate(self.matrices[c][:,k]) if j==1])
                
            C.append(cluster)
        return C   # each element of C represent the mode_k cluster

    def initiateMembershipMatrix(self):

        self.matrices = []

        l =  len(self.dataShape)
        
        for i in range(l):
            # mode-i unfolding of the data
            A = unfold(self.data, i)
                # use Kmeans to initiate the membership matrix
            kmeans = KMeans(n_clusters=self.R[i], init='k-means++', max_iter=100, n_init = 10, random_state=0)
            self.matrices.append(kmeans.fit_predict(A).tolist())   # cluster structured as a list
        
        # Build the membership matrices from the result of the k-means
        for i in range(l):
            M = np.zeros((  self.dataShape[i]   , self.R[i]  ))
            
            for j, u in enumerate(self.matrices[i]):
                M[j,u] = 1
                
            self.matrices[i] = M
                
        
    
    # Update the core tensor C
    def updateCoreTensor(self):

        # if the data is a tensor
        if len(self.dataShape) == 3:
            self.core = np.zeros((self.R[0], self.R[1], self.R[2] ))

            for i in itertools.product(range(self.R[0]), range(self.R[1]), range(self.R[2])): 
                                   
                r0, r1, r2 = i    
                # Find the M^-1
                MInvr0 = np.nonzero(self.matrices[0][:,r0])[0].tolist()  # return 
                MInvr1 = np.nonzero(self.matrices[1][:,r1])[0].tolist()
                MInvr2 = np.nonzero(self.matrices[2][:,r2])[0].tolist()
            
                nr = len(MInvr0) * len(MInvr1) * len(MInvr2)
                # print(nr)
                                   
                # sum
                A = self.data[MInvr0,:,:]
                A = A[:, MInvr1,:]
                A = A[:, :, MInvr2]
                self.core[r0, r1, r2] =  np.sum(A) / nr
        
                if self.sparse == True and self.rho == 0:
                    # we need nr, self.core, and ambda
                    # define lambda by BIC (Bayesian information criterion)
                    if abs(self.core[r0,r1,r2]) >= np.sqrt(self.ambda/nr) :
                        pass
                    else:
                        self.core[r0,r1,r2] = 0
                elif self.sparse == True and self.rho == 1:
                    self.core[r0,r1,r2] = sign(self.core[r0,r1,r2])*max(abs(self.core[r0,r1,r2])-(self.ambda / (2*nr)),0)
            
        elif len(self.dataShape) == 2:
            self.core = np.zeros((self.R[0], self.R[1] ))

            for i in itertools.product(range(self.R[0]), range(self.R[1])):                             
                                   
                r0, r1 = i
                # Find the M^-1
                MInvr0 = np.nonzero(self.matrices[0][:,r0])[0].tolist()
                MInvr1 = np.nonzero(self.matrices[1][:,r1])[0].tolist()
            
                nr = len(MInvr0) * len(MInvr1) 
                # print(nr)
                                   
                # sum
                A = self.data[MInvr0,:]
                A = A[:, MInvr1]
                self.core[r0, r1] =  np.sum(A) / nr
        
                if self.sparse == True and self.rho == 0:
                    # we need nr, self.core, and ambda
                    # define lambda by BIC (Bayesian information criterion)
                    if abs(self.core[r0,r1]) >= np.sqrt(self.ambda/nr) :
                        pass
                    else:
                        self.core[r0,r1] = 0
                elif self.sparse == True and self.rho == 1:
                    self.core[r0,r1] = sign(self.core[r0,r1])*max(abs(self.core[r0,r1])-(self.ambda / (2*nr)),0)
            
            
    # Find all the mean slice 
    def mean_slice(self, k):
        # build the mean (or centered) of each cluster according to the mode k
        
        if len(self.dataShape) == 2:
            if k==0:
                # we need the length of each block of mode-1
                j=45
        return j
        
    # Update the mode-k membership matrices
    def updateMembershipMatrix(self):
        
        for k in range(len(self.dataShape)):
            # select a random index in one dimension
            a = np.random.randint(0, self.dataShape[k])

            # for a in range(self.dataShape[k]):   # this is to test all elements of the tensor

            # we focus in the membership matrix M_k
            # Min is a list on which we take the min for the cluster label M_k(a)
            Min = []
            for r in range(self.R[k]):
                # Construction of the set I_k
                S = 0
                if (len(self.dataShape) == 3 and k == 0 ):
                    for i1, i2 in itertools.product(range(self.dataShape[1]), range(self.dataShape[2])):
                        ic1 = np.nonzero(self.matrices[1][i1,:])[0][0]
                        ic2 = np.nonzero(self.matrices[2][i2,:])[0][0]
                        S = S + (self.core[r,ic1,ic2] - self.data[a,i1,i2])**2
                            
                elif(len(self.dataShape) == 3 and k == 1):
                    for i0, i2 in itertools.product(range(self.dataShape[0]), range(self.dataShape[2])):
                        ic0 = np.nonzero(self.matrices[0][i0,:])[0][0]
                        ic2 = np.nonzero(self.matrices[2][i2,:])[0][0]
                        S = S + (self.core[ic0,r,ic2] - self.data[i0,a,i2])**2
                        
                elif(len(self.dataShape) == 3 and k == 2):
                    for i0, i1 in itertools.product(range(self.dataShape[0]), range(self.dataShape[1])):
                        ic0 = np.nonzero(self.matrices[0][i0,:])[0][0]
                        ic1 = np.nonzero(self.matrices[1][i1,:])[0][0]
                        S = S + (self.core[ic0,ic1,r] - self.data[i0,i1,a])**2
                    
                # ----------------------------------------------------------
                # If the input data is a matrix
                if (len(self.dataShape) == 2 and k == 0 ):
                    for i1 in range(self.dataShape[1]):
                        ic1 = np.nonzero(self.matrices[1][i1,:])[0][0]
                        S = S + (self.core[r,ic1] - self.data[a,i1])**2
                            
                elif(len(self.dataShape) == 2 and k == 1):
                    for i0 in range(self.dataShape[0]):
                        ic0 = np.nonzero(self.matrices[0][i0,:])[0][0]
                        S = S + (self.core[ic0,r] - self.data[i0,a])**2
                       
                Min.append(S)
                
                
            # new cluster : cluster index who has S minimal
            new = np.argmin(Min)
            # change the Membership matrix M_k(a)
            self.matrices[k][a,:] = np.zeros(( self.R[k] ))     
            self.matrices[k][a,new] = 1
                
    
        
                        
    def sign(self, a):
        if a < 0:
            return -1
        elif a > 0:
            return 1