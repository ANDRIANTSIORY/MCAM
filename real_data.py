#http://www.models.life.ku.dk/Flow_Injection

from sklearn import metrics
import numpy as np
import multiway_via_spectral_v1 as multiwayvs1  # MCAM-I
import multiway_via_spectral_v2 as multiwayvs2  # MCAM-II
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import SpectralClustering
import functions_v1  # MCAM-I
import functions_v2  # MCAM-II
from pymatreader import read_mat

if __name__ == "__main__":
	source = read_mat('../../data/Flow_Injection/fia.mat')
	dataShape = source['DimX']
	data = source['X']
	shape = []
	for i in range(3):
		shape.append(int(dataShape[i]) )
	data = np.reshape(data, shape, order="F")
	print("data shape : ", data.shape) 

	# We do not know yet the number of cluster
	# We just need the similarity matrices

	core_size= [2,2,2]    # the user can guess here because we onle need the similarity matrices C_Prime
	result_ariSLICE_hac, result_ariSLICEk_means = [], []
	multiway = multiwayvs1.Multiway_via_spectral(data, k=core_size) 
	estimation = multiway.get_result()
	C_prime = multiway.get_c_prime()  # list of three similarity matrices (from the three mode)

	# ---------------Determination of the number of the clusters ----------------

	### The silhouette score of the mode-1,  of the mode-2, and of the mode-3.
	# Plot
	#functions.silhouette_tensor( (2,10,1),(2,30,1),(2,30,1),data,C_prime)

	### The Davies-Bouldin index of the mode-1, of the mode-2,  and of the mode-3.
	# plot
	#functions.DaviesBouldinIndex( (2,10,1),(2,30,1),(2,30,1),data,C_prime)

	# ---------------------------------------------------------------------------

	# core tensor can be interpreted as the number of cluster in each mode
	# We use the RMSE to evaluate the clustering quality of the output
	# MCAM : - first component : MCAM-SC
	#        - second component : MCAM-AP

	core_size = [2,2,2] 
	real = []   # The partition of the elements in each cluster is unknown
	cp_rank = [2,2,2]
	#cluster = functions_v1.multiway_via_spec_dec(data, core_size, cp_rank, real, cp_tucker=True, tbm=True, mse=True)
	#for i in cluster:
	#	print(i[2])
	# ------- MCAM-I ------------------  
	print("MCAM-I (SC, AP), Tucker+k-means, CP+k-means, TBM : \n")
	cluster = functions_v1.multiway_via_spec_dec(data, core_size, cp_rank, real, cp_tucker=False, tbm=False, mse=True)
	for i in cluster:
		print(i[2])
    
	# ------- MCAM-II ------------------  
	print("\nMCAM-II : \n")
	cluster = functions_v2.multiway_via_spec_dec(data, core_size, cp_rank, real, cp_tucker=False, tbm=False, mse=True)
	for i in cluster:
		print(i[2])