# silhouette for each nbr of cluster
from sklearn.metrics import davies_bouldin_score
from sklearn import metrics
import numpy as np
import multiway_via_spectral_v2 as multiwayvs
import matplotlib.pyplot as plt
from sklearn.cluster import  SpectralClustering
from tensorly.decomposition import CP, Tucker
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
import math
import rmse_data_estimation as diffMse
from sklearn.metrics import silhouette_samples, silhouette_score
from tensorly import unfold
import multiway_Clustering_TBM as tbm_method
import tensorly as tl
import itertools



def multiway_via_spec_dec(data, core_size, cp_rank, real, cp_tucker = False, tbm = False, mse=False):    
    result_ariSLICE_hac, result_ariSLICEk_means = [], []
    multiway = multiwayvs.Multiway_via_spectral(data, k=core_size) 
    estimation = multiway.get_result()

    result = []
    #----------------------------------------
    spectral, affProp = [], []
    ari_s, ari_a = [], []
    nmi_s, nmi_a = [], []
    for i in range(3):
        spectral.append(estimation[i][0])
        affProp.append(estimation[i][1])

        if real != []:
            ari_s.append(metrics.adjusted_rand_score(real, spectral[i]))
            nmi_s.append(metrics.adjusted_mutual_info_score(real, spectral[i]))
            ari_a.append(metrics.adjusted_rand_score(real, affProp[i]))
            nmi_a.append(metrics.adjusted_mutual_info_score(real, affProp[i]))

    # ----------------------------------------
    C_prime = multiway.get_c_prime()  # three element, c_prime of three mode, liste with three elements

    # evaluation
    # build the membershipe matrices
    if mse == True:
        difference_s = diffMse.Mse_multiway_evaluation(data, spectral)

        difference_a = diffMse.Mse_multiway_evaluation(data, affProp)

        result.append( (ari_s, nmi_s, difference_s.result) )
        result.append( (ari_a, nmi_a, difference_a.result) )
    else :
        result.append( (ari_s, nmi_s) )
        result.append( (ari_a, nmi_a) )


    # CP and Tucker 
    if cp_tucker == True:
        res_cp, res_tucker = [], []
        cp = CP(rank=cp_rank[0]).fit_transform(data)  # sum of rank-one
        tucker = Tucker(rank=cp_rank).fit_transform(data) # I take more rank as the core_size
        ari_cp, ari_tucker = [], []
        nmi_cp, nmi_tucker = [], []
        for i in range(3):
            cluster_tucker = KMeans(n_clusters=core_size[i], random_state=0).fit(tucker[1][i])
            res_tucker.append(cluster_tucker.labels_)
            cluster_cp = KMeans(n_clusters=core_size[i], random_state=0).fit(cp[1][i])
            res_cp.append(cluster_cp.labels_)
            if real != []:
                ari_cp.append(metrics.adjusted_rand_score(real, cluster_cp.labels_))
                nmi_cp.append(metrics.adjusted_mutual_info_score(real, cluster_cp.labels_))
                ari_tucker.append(metrics.adjusted_rand_score(real, cluster_tucker.labels_))
                nmi_tucker.append(metrics.adjusted_mutual_info_score(real, cluster_tucker.labels_))

        if mse == True :
    
            difference_tucker = diffMse.Mse_multiway_evaluation(data, res_tucker)
            difference_cp = diffMse.Mse_multiway_evaluation(data, res_cp)

            result.append( (ari_tucker,nmi_tucker, difference_tucker.result) )
            result.append( (ari_cp, nmi_cp, difference_cp.result) )
        else :
            result.append( (ari_tucker,nmi_tucker) )
            result.append( (ari_cp, nmi_cp) )

    if tbm == True :
        # TBM
        res_TBM = []
        tbm1 = tbm_method.MultiwayClusteringTBM(data, core_size, iteration=50)
        cluster = tbm1.get_cluster()
        #---------- ARI ---------------
        ari = []
        nmi_tbm = []
        for mode in range(3):
            a = np.zeros((data.shape[mode]), dtype=int)
            for i in range(core_size[mode]):
                for j in cluster[mode][i]:
                    a[j]=int(i)
            res_TBM.append(a)
            if real != []:
                ari.append(metrics.adjusted_rand_score(real, a))
                nmi_tbm.append(metrics.adjusted_mutual_info_score(real, a))

        if mse == True :
            difference_tbm = diffMse.Mse_multiway_evaluation(data, res_TBM)

            result.append( (ari, nmi_tbm , difference_tbm.result) )
        else :
            result.append( (ari, nmi_tbm ) )
        

    return result

    # MCAM-SP , MCAM-AP, Tucker+kmeans, cp+kmeans, TBM  / each result is the (ARI , NMI, RMSE between estimation and the real value)


# silhouette for each nbr of cluster
def silhouette_tensor(max_lista, max_listb, max_listc, data, cPrime):
    # C_prime is a list of 3 elements, c_prime for mode1, mode2 and mode3
    
    listea = [i for i in range(max_lista[0], max_lista[1],max_lista[2])]
    listeb = [i for i in range(max_listb[0], max_listb[1],max_listb[2])]
    listec = [i for i in range(max_listc[0], max_listc[1],max_listc[2])]
    range_n_clusters = [listea,listeb,listec]

    res_silhouette_avg = []
    for i in range(3):
        silhouette_avg = []
        matrice = unfold(data, i)  
        for n_clusters in range_n_clusters[i]:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(cPrime[i])
    
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg.append(silhouette_score(matrice, cluster_labels))
            #print("For n_clusters =", n_clusters,"The average silhouette_score is :", silhouette_avg)
        res_silhouette_avg.append(silhouette_avg)
        # plot the silhouette
        if i == 0:
            plt.plot(listea,silhouette_avg, label="mode-1")
        elif i==1:    
            plt.plot(listeb,silhouette_avg, label="mode-2")
        else :
            plt.plot(listec,silhouette_avg, label="mode-3")
            
    max_abc = max(max_lista[1], max_listb[1], max_listc[1])
    min_abc = min(max_lista[0], max_listb[0], max_listc[0])
    new_list = range(min_abc,max_abc,4)
    plt.xticks(new_list)
    plt.xlabel("number of clusters", fontsize=13)
    plt.ylabel("Silhouette score", fontsize=13)
    plt.legend(fontsize=13)
    plt.savefig('./image/silhouette.png')
    plt.show()
 





# DaviesBouldinIndex
def DaviesBouldinIndex(max_lista, max_listb, max_listc, data, cPrime):
    # C_prime is a list of 3 elements, c_prime for mode1, mode2 and mode3
    listea = [i for i in range(max_lista[0], max_lista[1],max_lista[2])]
    listeb = [i for i in range(max_listb[0], max_listb[1],max_listb[2])]
    listec = [i for i in range(max_listc[0], max_listc[1],max_listc[2])]
    range_n_clusters = [listea,listeb,listec]

    davis_bouldin = []
    for i in range(3):
        d_b = []
        matrice = unfold(data, i)  
        for n_clusters in range_n_clusters[i]:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(cPrime[i])
    
            d_b.append(davies_bouldin_score(matrice, cluster_labels))
        davis_bouldin.append(d_b)
        # plot the Davies bouldin
        if i == 0:
            plt.plot(listea,d_b, label="mode-1")
        elif i==1:    
            plt.plot(listeb,d_b, label="mode-2")
        else :
            plt.plot(listec,d_b, label="mode-3")
            
    max_abc = max(max_lista[1], max_listb[1], max_listc[1])
    min_abc = min(max_lista[0], max_listb[0], max_listc[0])
    new_list = range(min_abc,max_abc,4)
    plt.xticks(new_list)
    plt.xlabel("number of clusters", fontsize=13)
    plt.ylabel("Davies Bouldin Index", fontsize=13)
    plt.legend( fontsize=13)
    plt.savefig('./image/Daviesbouldin.png')
    plt.show()
        

def membership_matrix_from_cluster(cluster ):
    membership_matrices = []
    for i in range(3):
        m = np.zeros(( len(cluster[i]), len(set(cluster[i])) ))
        for j,k in enumerate(cluster[i]) : 
            m[j,k] = 1
        membership_matrices.append(m)
    return membership_matrices

def build_core_tensor(tensor, matrices):  # mean of each block
    c0, c1, c2 = matrices[0].shape[1], matrices[1].shape[1], matrices[2].shape[1]
    core = np.zeros((c0, c1, c2 ))
    
    for r0, r1, r2  in itertools.product(range(c0), range(c1), range(c2)):
        # Find the M^-1
        MInvr0 = np.nonzero(matrices[0][:,r0])[0].tolist()  
        MInvr1 = np.nonzero(matrices[1][:,r1])[0].tolist()
        MInvr2 = np.nonzero(matrices[2][:,r2])[0].tolist()
            
        nr = len(MInvr0) * len(MInvr1) * len(MInvr2)
        A = tensor[MInvr0,:,:]
        A = A[:, MInvr1,:]
        A = A[:, :, MInvr2]
        core[r0, r1, r2] =  np.sum(A) / nr
    return core


