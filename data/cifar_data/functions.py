import tensorly as tl
import numpy as np
import intertools


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


def rmse(tensor1, core, membership_matrix):
    # build the estimator tensor
    estim = tl.tucker_to_tensor((core, membership_matrix))
    a = tensor1 - estim 
    return (np.sum(a * a))**0.5

def eval_clutering(tensor, cluster):
    matrices = membership_matrix_from_cluster(cluster)
    core = build_core_tensor(tensor, matrices)
    return rmse(tensor, core, matrices)