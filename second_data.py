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



##- The ARI and NMI of the different algorithms (MCAM, CP+k-means, Tucker+k-means, TBM)
def mcam_and_others():
    # m : tensor of shape (m, m, m)
    # k : number of the cluster in each mode
    # cp_tucker_rank : rank decomposition of the tensor
    m, k = 200, 20   
    cp_tucker_rank = 20
    #-------------------------------
    r, real = [i for i in range(m)], [i for i in range(m)]
    r = np.array_split(r, k)
    r = [list(i) for i in r]
    for i in range(k):
        for j in r[i]:
            real[j] = i
    #-----------data TBM----------------
    ari_spectral1, nmi_spectral1, ari_AffProp1, nmi_AffProp1 = [], [], [], []
    ari_spectral2, nmi_spectral2, ari_AffProp2, nmi_AffProp2 = [], [], [], []
    ari_cp, nmi_cp = [], []
    ari_tucker, nmi_tucker = [], []
    ari_tbm, nmi_tbm = [], []
    experiments = 2
    r = []

    for i in range(experiments):
        generatedata = gdata.Generate_tensor(m = m, k=k)
        data = generatedata.second_data(-2,2,1)
        # ----------- MCAM-I - Tucker+k-means - CP+k-means - TBM ----------------
        core_size = [k,k,k]
        cp_rank = [cp_tucker_rank,cp_tucker_rank,cp_tucker_rank]
        cluster = functions_v1.multiway_via_spec_dec(data, core_size, cp_rank, real, cp_tucker=True, tbm=True)
        ari_spectral1.append(np.mean(cluster[0][0]))
        ari_AffProp1.append(np.mean(cluster[1][0]))
        nmi_spectral1.append(np.mean(cluster[0][1]))
        nmi_AffProp1.append(np.mean(cluster[1][1]))

        ari_tucker.append(np.mean(cluster[2][0]))
        ari_cp.append(np.mean(cluster[3][0]))
        ari_tbm.append(np.mean(cluster[4][0]))

        nmi_tucker.append(np.mean(cluster[2][1]))
        nmi_cp.append(np.mean(cluster[3][1]))
        nmi_tbm.append(np.mean(cluster[4][1]))
        # ------------------------------ MCAM-II -------------------
        cluster = functions_v2.multiway_via_spec_dec(data, core_size, cp_rank, real, cp_tucker=False, tbm=False)
        ari_spectral2.append(np.mean(cluster[0][0]))
        ari_AffProp2.append(np.mean(cluster[1][0]))
        nmi_spectral2.append(np.mean(cluster[0][1]))
        nmi_AffProp2.append(np.mean(cluster[1][1]))
        # -------------------------------------------------------------


    print("MCAM1-SC ARI : ", np.mean(ari_spectral1)," +/- ", np.std(ari_spectral1))
    print("MCAM1-AP ARI : ", np.mean(ari_AffProp1)," +/- ", np.std(ari_AffProp1))
    print("MCAM2-SC ARI : ", np.mean(ari_spectral2)," +/- ", np.std(ari_spectral2))
    print("MCAM2-AP ARI : ", np.mean(ari_AffProp2)," +/- ", np.std(ari_AffProp2))
    print("ARI Tucker: ", np.mean(ari_tucker)," +/- ", np.std(ari_tucker))
    print("ARI CP: ", np.mean(ari_cp)," +/- ", np.std(ari_cp))
    print("ARI TBM : ",np.mean(ari_tbm), "+/- ", np.std(ari_tbm))
    print("\n")
    print("MCAM1-SC NMI : ", np.mean(nmi_spectral1), " +/- ", np.std(nmi_spectral1))
    print("MCAM1-AP NMI : ", np.mean(nmi_AffProp1), " +/- ", np.std(nmi_AffProp1))
    print("MCAM2-SC NMI : ", np.mean(nmi_spectral2), " +/- ", np.std(nmi_spectral2))
    print("MCAM2-AP NMI : ", np.mean(nmi_AffProp2), " +/- ", np.std(nmi_AffProp2))
    print("NMI Tucker: ", np.mean(nmi_tucker), " +/- ", np.std(nmi_tucker))
    print("NMI CP: ", np.mean(nmi_cp), " +/- ", np.std(nmi_cp))
    print("NMI TBM : ",np.mean(nmi_tbm), "+/- ", np.std(nmi_tbm))


# The ARI and NMI of the MCAM algorithm for r varies from 1 to 10.
import mcam_for_fix_r_v1 as mcam_r        # for MCAM-I
#import mcam_for_fix_r_v2 as mcam_r       # for MCAM-II

def diff_value_r():
    # MCAM with different value of r
    m, k = 200, 20   # m is the dimension of data, k is the number of clusters
    er = []
    #-------------------------------
    r, real = [i for i in range(m)], [i for i in range(m)]
    r = np.array_split(r, k)
    r = [list(i) for i in r]
    for i in range(k):
        for j in r[i]:
            real[j] = i

    #-----------data TBM----------------
    ari_spectral, nmi_spectral, ari_AffProp, nmi_AffProp = [], [], [], []
    ari_cp, nmi_cp = [], []
    ari_tucker, nmi_tucker = [], []
    experiments = 1
    r = []
    mean_ari_spec, mean_nmi_spec, mean_ari_ap, mean_nmi_ap = [], [], [], []
    for R in range(1,11): 
        for i in range(experiments):
            generatedata = gdata.Generate_tensor(m = m, k=k)
            data = generatedata.second_data(-2,2,1)
            # --------------------------------
            multiway = mcam_r.Multiway_via_spectral(data, k=[20,20,20], r=R)  # k is the number of clusters
            cluster = multiway.get_result()  
            # separation of the spectral clustering and the affinity propagation
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
    #plt.title("MCAM-II")
    plt.title("MCAM-I")
    #plt.savefig('./image/boxplot_tbm_algo2.png',bbox_inches='tight')
    plt.show()


if __name__=="__main__":
    #run()

    diff_value_r()
