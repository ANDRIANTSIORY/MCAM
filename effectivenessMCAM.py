import generate_data as gdata
from sklearn import metrics
import numpy as np
import multiway_via_spectral_v1 as multiwayvs1  # MCAM-I
import multiway_via_spectral_v2 as multiwayvs2  # MCAM-II
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import SpectralClustering
from tensorly.decomposition import CP, Tucker
from sklearn.metrics.cluster import adjusted_rand_score
import functions_v1  # MCAM-I
import functions_v2  # MCAM-II
from pymatreader import read_mat
from sklearn.cluster import KMeans
import math


if __name__=="__main__":
	# v1 refers to MCAM-I
	# v2 refers to MCAM-II
	m, k, size = 100, 1, 12
	#--------real cluster---------
	real = []
	for i in range(k):
		real = real + [i for p in range(size)]
	real  = real + [k for i in range(m - (k*size))]

	#----------- MCAM for a rank-one tensor ----------------
	std_cluster_spectralClustering_v1, std_cluster_spectralClustering_v2 = [], []
	std_aff_v1, std_aff_v2 = [], []
	affinity_propagation_v1, affinity_propagation_v2 = [], []
	cluster_spectralClustering_v1, cluster_spectralClustering_v2 = [], [] # will contain list of tuple ( result_spectralClsutering, result_AffinityPropagation)
	for gamma in range(30, 81, 5):
		spectral_v1, spectral_v2 = [], []
		aff_v1, aff_v2 = [], []
		for _ in range(2):
			data = gdata.Generate_tensor(m=m, n1=m,n2=m,k1=size,k2=size,k3=size, cluster=k, gamma=gamma)
			data = data.first_data()

			core_size = [k+1,k+1,k+1]
			cp_rank = [1,1,1]   # we do not use it in this experiment but the class needs it as a parameter
			# ------------------ MCAM-I ---------------------------
			cluster = functions_v1.multiway_via_spec_dec(data, core_size, cp_rank, real,cp_tucker=False,tbm=False)
			ari_s_v1 = cluster[0][0]
			aff_s_v1 = cluster[1][0]
			# ------------------- MCAM-II ------------------------------
			cluster = functions_v2.multiway_via_spec_dec(data, core_size, cp_rank, real,cp_tucker=False,tbm=False)
			ari_s_v2 = cluster[0][0]
			aff_s_v2 = cluster[1][0]
			spectral_v1.append(np.mean(ari_s_v1))
			spectral_v2.append(np.mean(ari_s_v2))
			aff_v1.append(np.mean(aff_s_v1))
			aff_v2.append(np.mean(aff_s_v2))

		cluster_spectralClustering_v1.append(np.mean(spectral_v1))
		std_cluster_spectralClustering_v1.append(np.std(spectral_v1))
		cluster_spectralClustering_v2.append(np.mean(spectral_v2))
		std_cluster_spectralClustering_v2.append(np.std(spectral_v2))
		affinity_propagation_v1.append(np.mean(aff_v1))
		affinity_propagation_v2.append(np.mean(aff_v2))
		std_aff_v1.append(np.std(aff_v1))
		std_aff_v2.append(np.std(aff_v2))

	
	# -------------------plot--------------
	# Plot MCAM-I and MCAM-II with Spectral Clustering

	x = [i for i in range(30,81,5)]
	plt.plot(x,cluster_spectralClustering_v1, c ='g',label="MCAM-I-SC")
	plt.errorbar(x, cluster_spectralClustering_v1, std_cluster_spectralClustering_v1, linestyle='None', marker='')
	plt.plot(x,cluster_spectralClustering_v2, c ='r',label="MCAM-II-SC")
	plt.errorbar(x, cluster_spectralClustering_v2, std_cluster_spectralClustering_v2, linestyle='None', marker='')
	new_list = range(math.floor(min(x)), math.ceil(max(x)+1),5)
	plt.xticks(new_list)
	plt.legend(bbox_to_anchor=(1, 1), fontsize=14)
	plt.ylabel("ARI",  fontsize=14)
	plt.xlabel("signal strength",  fontsize=14)
	#plt.savefig('./image/SNR_D100_G30_80_C1_SC.png',bbox_inches='tight')
	plt.show()