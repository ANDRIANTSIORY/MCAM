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


# 2 - First synthetical dataset
##  2.a - Comparison of the effectiveness of MCAM and the other algorithms for a n rank-one tensor
# n = 8
# **functions_v1** corresponds to MCAM-I
# **functions_v2** corresponds to MCAM-II
def mcam_and_others():
	m, k, size = 100, 8, 12
	#--------real cluster---------
	real = []
	for i in range(k):
		real = real + [i for p in range(size)]
	real  = real + [k for i in range(m - (k*size))]

	#-----------Sum of n rank-one tensor ----------------
	std_cluster_spectralClustering1, std_cluster_affinityPropagation1 = [], []
	cluster_spectralClustering1, cluster_affinityPropagation1 = [], [] # will contain list of tuple ( result_spectralClsutering, result_AffinityPropagation)
	std_cluster_spectralClustering2, std_cluster_affinityPropagation2 = [], []
	cluster_spectralClustering2, cluster_affinityPropagation2 = [], []
	cluster_tucker, cluster_cp, cluster_tbm = [], [], []
	std_tucker, std_cp, std_tbm = [], [], []
	NMI_spectral1, NMI_aff1, NMI_tucker, NMI_cp, NMI_tbm =[], [], [], [], []
	STD_spectral1, STD_aff1, STD_tucker, STD_cp, STD_tbm = [], [], [], [], []
	NMI_spectral2, NMI_aff2, STD_spectral2, STD_aff2 = [],[],[],[]

	cp_tucker = True
	tbm_method = True 

	for gamma in range(30, 81, 5):
		spectral1, aff1, spectral2, aff2,tucker, cp, tbm = [], [], [], [], [],[],[]
		nmi_spectral1, nmi_aff1,nmi_spectral2, nmi_aff2, nmi_tucker, nmi_cp, nmi_tbm = [], [], [], [], [],[],[]
		for _ in range(2):
			data = gdata.Generate_tensor(m=m, n1=m,n2=m,k1=size,k2=size,k3=size, cluster=k, gamma=gamma)
			data = data.first_data()

			# --------------------MCAM-I --------------------------------------------
			core_size = [k+1,k+1,k+1]
			cp_rank = [9,9,9]  
			cluster = functions_v1.multiway_via_spec_dec(data,core_size,cp_rank,real,cp_tucker=False,tbm=False)
			ari_s = cluster[0][0]
			ari_a = cluster[1][0]
			nmi_s = cluster[0][1]
			nmi_a = cluster[1][1]

			spectral1.append(np.mean(ari_s))
			aff1.append(np.mean(ari_a))
			nmi_spectral1.append(np.mean(nmi_s))
			nmi_aff1.append(np.mean(nmi_a))

			# ---------------------------------------------------------------------------
			# --------------MCAM-II - Tucker+k-means - CP+k-means - TBM-----------------
			cluster = functions_v2.multiway_via_spec_dec(data,core_size,cp_rank,real,cp_tucker=cp_tucker,tbm=tbm_method)
			ari_s = cluster[0][0]
			ari_a = cluster[1][0]
			nmi_s = cluster[0][1]
			nmi_a = cluster[1][1]

			spectral2.append(np.mean(ari_s))
			aff2.append(np.mean(ari_a))
			nmi_spectral2.append(np.mean(nmi_s))
			nmi_aff2.append(np.mean(nmi_a))

			# --------------------------------------------------------------------------
			if cp_tucker == True :
				ari_tucker = cluster[2][0]
				ari_cp = cluster[3][0]
				nmi_tuck = cluster[2][1]
				nmi_c = cluster[3][1]
				tucker.append(np.mean(ari_tucker))
				cp.append(np.mean(ari_cp))
				nmi_tucker.append(np.mean(nmi_tuck))
				nmi_cp.append(np.mean(nmi_c))

			if tbm_method == True :
				ari_tbm = cluster[4][0]
				nmi_tb = cluster[4][1]
				tbm.append(np.mean(ari_tbm))
				nmi_tbm.append(np.mean(nmi_tb))

		cluster_spectralClustering1.append(np.mean(spectral1))
		std_cluster_spectralClustering1.append(np.std(spectral1))
		cluster_affinityPropagation1.append(np.mean(aff1))
		std_cluster_affinityPropagation1.append(np.std(aff1))

		NMI_spectral1.append(np.mean(nmi_spectral1))
		NMI_aff1.append(np.mean(nmi_aff1))
		STD_spectral1.append(np.std(nmi_spectral1))
		STD_aff1.append(np.std(nmi_aff1))
		# ---------------------------------------------------------------------
		cluster_spectralClustering2.append(np.mean(spectral2))
		std_cluster_spectralClustering2.append(np.std(spectral2))
		cluster_affinityPropagation2.append(np.mean(aff2))
		std_cluster_affinityPropagation2.append(np.std(aff2))

		NMI_spectral2.append(np.mean(nmi_spectral2))
		NMI_aff2.append(np.mean(nmi_aff2))
		STD_spectral2.append(np.std(nmi_spectral2))
		STD_aff2.append(np.std(nmi_aff2))


		if cp_tucker == True :
			cluster_tucker.append(np.mean(tucker))
			std_tucker.append(np.std(tucker))
			cluster_cp.append(np.mean(cp))
			std_cp.append(np.std(cp))
			NMI_tucker.append(np.mean(nmi_tucker))
			NMI_cp.append(np.mean(nmi_cp))
			STD_tucker.append(np.std(nmi_tucker))
			STD_cp.append(np.std(nmi_cp))
		if tbm_method == True :
			cluster_tbm.append(np.mean(tbm))
			std_tbm.append(np.std(tbm))
			NMI_tbm.append(np.mean(nmi_tbm))
			STD_tbm.append(np.std(nmi_tbm))

  # ARI
	x = [i for i in range(30,81,5)]
	plt.plot(x,cluster_spectralClustering1, c ='g',label="MCAM-I-SP")
	plt.errorbar(x, cluster_spectralClustering1, std_cluster_spectralClustering1, linestyle='None', marker='')
	plt.plot(x,cluster_affinityPropagation1, c ='r',label="MCAM-I-AP")
	plt.errorbar(x, cluster_affinityPropagation1, std_cluster_affinityPropagation1, linestyle='None', marker='')
	# ---------------------------------
	plt.plot(x,cluster_spectralClustering2, c ='k',label="MCAM-II-SP")
	plt.errorbar(x, cluster_spectralClustering2, std_cluster_spectralClustering2, linestyle='None', marker='')
	plt.plot(x,cluster_affinityPropagation2, c ='m',label="MCAM-II-AP")
	plt.errorbar(x, cluster_affinityPropagation2, std_cluster_affinityPropagation2, linestyle='None', marker='')
	# -------------------------------------
	plt.plot(x, cluster_tucker, c='b', label="Tucker+k-means")
	plt.errorbar(x, cluster_tucker, std_tucker, linestyle='None', marker='')
	plt.plot(x, cluster_cp, c='y', label="CP+k-means")
	plt.errorbar(x, cluster_cp, std_cp, linestyle='None', marker='')
	plt.plot(x, cluster_tbm, c='c', label="TBM")
	plt.errorbar(x, cluster_tbm, std_tbm, linestyle='None', marker='')

	new_list = range(math.floor(min(x)), math.ceil(max(x)+1),5)
	plt.xticks(new_list)
	plt.legend(bbox_to_anchor=(1, 1), fontsize=14)
	plt.ylabel("ARI",  fontsize=14)
	plt.xlabel("signal strength",  fontsize=14)
	#plt.savefig('./image/mcam_and_other_tfs_data_ari.png',bbox_inches='tight')
	plt.show()


## 2.b - The ARI and NMI of the MCAM algorithm for r varies from 1 to 10 and for $\gamma=60$.
import mcam_for_fix_r_v1 as mcam_r        # for MCAM-I

#import mcam_for_fix_r_v2 as mcam_r       # for MCAM-II

def diff_value_r():
	m, k, size = 100, 8, 12
	cp_tucker_rank = 8
	#--------real cluster---------
	real = []
	for i in range(k):
		real = real + [i for p in range(size)]
	real  = real + [k for i in range(m - (k*size))]

	#-----------first data ----------------
	er =[]
	ari_cp, nmi_cp = [], []
	ari_tucker, nmi_tucker = [], []
	experiments = 10
	mean_ari_spec, mean_nmi_spec, mean_ari_ap, mean_nmi_ap = [], [], [], []
	for R in range(1,11): 
		ari_spectral, nmi_spectral, ari_AffProp, nmi_AffProp = [],[],[],[]
		for i in range(experiments):
			data = gdata.Generate_tensorBiclustering(m=m, n1=m,n2=m,k1=size,k2=size, k3=size, cluster=k, gamma=55)
			data = data.first_data()

		    # --------------------------------
		    multiway = mcam_r.Multiway_via_spectral(data, k=[k+1,k+1,k+1], r=R)  
		    cluster = multiway.get_result()  
		    spectral = [a[0] for a in cluster]
		    AffProp = [a[1] for a in cluster]
		    res_ari_spectral, res_nmi_spectral = [], []
		    res_ari_AffProp, res_nmi_AffProp = [], []
		    for j in range(3):
		    	res_ari_spectral.append(metrics.adjusted_rand_score(real, spectral[j]))
		    	res_nmi_spectral.append(metrics.adjusted_mutual_info_score(real, spectral[j]))
		    	#---------------------------------------
		    	res_ari_AffProp.append(metrics.adjusted_rand_score(real, AffProp[j]))
		    	res_nmi_AffProp.append(metrics.adjusted_mutual_info_score(real, AffProp[j]))


		    ari_spectral.append(np.mean(res_ari_spectral))
		    nmi_spectral.append(np.mean(res_nmi_spectral))
		    ari_AffProp.append(np.mean(res_ari_AffProp))
		    nmi_AffProp.append(np.mean(res_nmi_AffProp))

		mean_ari_spec.append(np.mean(ari_spectral))
		mean_nmi_spec.append( np.mean(nmi_spectral))
		mean_ari_ap.append(np.mean(ari_AffProp))
		mean_nmi_ap.append( np.mean(nmi_AffProp))


	# boxplot 
	plt.boxplot([mean_ari_spec, mean_nmi_spec, mean_ari_ap, mean_nmi_ap],widths=0.5, labels=["ARI-SC","NMI-SC","ARI-AP","NMI-AP"])
	plt.title("MCAM-I")
	#plt.savefig('./image/boxplot_tfs_algo1.png',bbox_inches='tight')
	plt.show()

## 2.c - Performance of CP+k-means and Tucker+k-means with different rank of decomposition  

# We only consider the clustering result from mode-1 of the 3-order tensor dataset
def cp_tucker_depend_of_rank():
	m, k, size = 100, 8, 12
	#--------real cluster---------
	real = []
	for i in range(k):
		real = real + [i for p in range(size)]
	real  = real + [k for i in range(m - (k*size))]
	cp, tucker = [], []
	# ----------------------------------
	experiments = 10
	rank_max = 10
	# ----------------------------------
	for j in range(experiments):
		# ---------second data-------------------
		data = gdata.Generate_tensorBiclustering(m=m, n1=m,n2=m,k1=size,k2=size,k3=size, cluster=k, gamma=55)
		data = data.first_data()

		#------- CP and Tucker -----------------------------
		res_ari_cp ,res_ari_tucker = [], []
		for i in range(1,rank_max+1):
			a = CP(rank=i).fit_transform(data) 
			b = Tucker(rank=[i,i,i]).fit_transform(data) 
			#  ----------------- k-means ------------------
			label_cp = KMeans(n_clusters=k+1, random_state=0).fit(a[1][0])
			label_tucker = KMeans(n_clusters=k+1, random_state=0).fit(b[1][0])
			res_ari_cp.append(metrics.adjusted_rand_score(real, label_cp.labels_))
			res_ari_tucker.append(metrics.adjusted_rand_score(real, label_tucker.labels_))

		cp.append(res_ari_cp)
		tucker.append(res_ari_tucker)

	mean_cp, std_cp, mean_tucker, std_tucker = [], [], [], []
	for i in range(rank_max):
		intermediate_cp, intermediate_tucker = [], []
		for j in range(experiments):
			intermediate_cp.append(cp[j][i])
			intermediate_tucker.append(tucker[j][i])
		mean_cp.append(np.mean(intermediate_cp))
		std_cp.append(np.std(intermediate_cp))
		mean_tucker.append(np.mean(intermediate_tucker))
		std_tucker.append(np.std(intermediate_tucker))


	# ---------- plot -----------------------

	lim = rank_max
	x = [i for i in range(1,lim +1)]
	plt.plot(x,mean_cp,c='y', label="CP+k-means")
	plt.errorbar(x, mean_cp, std_cp, linestyle='None', marker='')

	plt.plot(x,mean_tucker, c ='b',label="Tucker+k-means")
	plt.errorbar(x, mean_tucker, std_tucker, linestyle='None', marker='')
	new_list = range(math.floor(min(x)), math.ceil(max(x))+1,2)
	plt.xticks(new_list)
	plt.legend(loc="lower right", fontsize=14)
	plt.ylabel("ARI",  fontsize=14)
	plt.xlabel("rank of tensor decomposition",  fontsize=14)
	#plt.savefig('./image/cp_tucker_one_mode_55.png',bbox_inches='tight')
	plt.show()


if __name__=="__main__":
	diff_value_r()