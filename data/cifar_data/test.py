from sklearn.decomposition import SparsePCA
from keras.datasets import mnist, cifar10
import numpy as np
import multiway_via_spectral as mvs

from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from tensorly.decomposition import CP, Tucker

# load the data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)

#the train data
limite = 500
data = x_train[:limite,:,:,:]
data = data / 255
label = y_train[:limite]

#chaque couche donne un donnée de tenseur
data1, data2, data3 = np.zeros((limite, 32,32)), np.zeros((limite, 32,32)), np.zeros((limite, 32,32))
for i in range(limite):
    data1[i,:,:] = data[i,:,:,0]
    data2[i,:,:] = data[i,:,:,1]
    data3[i,:,:] = data[i,:,:,2]


label = label.reshape(-1)
label = label.tolist()

# the mode-1 should be divided in 10 clusters (0-9)
for l in range(3):
    if l==0:
        method = mvs.Multiway_via_spectral(data1, k=[10,5,5])
    if l==1:
        method = mvs.Multiway_via_spectral(data2, k=[10,5,5])
    if l==2:
        method = mvs.Multiway_via_spectral(data3, k=[10,5,5])

    result = method.get_result()
    a, b = [], []
    for i in range(2):
        a.append(result[i][0])
        b.append(result[i][1])

    print("ARI MCAM : ",l, " -  ", adjusted_rand_score(a[0], label))

data4 = data1 + data2 + data3
method = mvs.Multiway_via_spectral(data4, k=[10,5,5])
result = method.get_result()
a, b = [], []
for i in range(2):
    a.append(result[i][0])
    b.append(result[i][1])

print("ARI MCAM 4: ", adjusted_rand_score(a[0], label))

# tucker + kmeans
factors= Tucker(rank=[10,5,5]).fit_transform(data1)
# k means
print(len(factors))
kmeans = KMeans(n_clusters=10, random_state=0).fit(factors[1][0])
res = kmeans.labels_

# evaluation
print("ARI Tucker : ", adjusted_rand_score(label, res))