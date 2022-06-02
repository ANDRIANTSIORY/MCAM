
# Build the membership matrices from the result of the k-means
import numpy as np
import tensorly as tl
import itertools
import sys

class Mse_multiway_evaluation():
	def __init__(self, data, clusterListe):
		shape = data.shape
		membership = self.cluster_to_membership(shape, clusterListe)
		core = self.mean_block_Core(data, membership)
		data_approx = self.tensor_from_core_and_membershipMatrices(core, membership)
		self.result = self.rmse_difference_of_two_tensors(data, data_approx)



	def cluster_to_membership(self, tensorShape, clusterList):   # each element of the list is the cluster in one mode
		# number of cluster in each mode
		result = []
		clusterNumber = []
		for i in range(len(tensorShape)):
			clusterNumber.append(len(set(clusterList[i])))

		for i in range(len(tensorShape)):
			M = np.zeros(( tensorShape[i], clusterNumber[i]))

			for j, u in enumerate(clusterList[i]):
				M[j,u]=1

			result.append(M)


		return result

	def mean_block_Core(self, tensor, membershipMatrices):
		c0 = len(membershipMatrices[0][0,:])
		c1 = len(membershipMatrices[1][0,:])
		c2 = len(membershipMatrices[2][0,:])
		core = np.zeros((c0,c1,c2))

		for i in itertools.product(range(c0), range(c1), range(c2)): 
			r0, r1, r2 = i

			# Find the M^-1
			MInvr0 = np.nonzero(membershipMatrices[0][:,r0])[0].tolist()  
			MInvr1 = np.nonzero(membershipMatrices[1][:,r1])[0].tolist()
			MInvr2 = np.nonzero(membershipMatrices[2][:,r2])[0].tolist()

			nr = len(MInvr0) * len(MInvr1) * len(MInvr2)

			# the mean of the block
			A = tensor[MInvr0,:,:]
			A = A[:, MInvr1,:]
			A = A[:, :, MInvr2]
			core[r0, r1, r2] =  np.sum(A) / nr

		return core

	def tensor_from_core_and_membershipMatrices(self, core, membershipMatrices):
		return tl.tucker_to_tensor((core, membershipMatrices))

	def rmse_difference_of_two_tensors(self, T1, T2):
		r1 = T1.shape
		r2 = T2.shape
		if r1 == r2 :
			T1 = T1 - T2
			T1 = T1 * T1
			sum = np.sum(T1) / (r1[0] * r1[1] *r1[2])
		else :
			print("The tensors must have the same shape")
			sys.exit(1)
		return np.sqrt(sum)







