# source : https://www.askpython.com/python/examples/load-and-plot-mnist-dataset-in-python

from keras.datasets import mnist
import numpy as np
import multiway_via_spectral as multiwayvs
from sklearn import metrics


(train_X, train_y), (test_X, test_y) = mnist.load_data()

#print('X_train: ' + str(train_X.shape))
#print('Y_train: ' + str(train_y.shape))
#print('X_test:  '  + str(test_X.shape))
#print('Y_test:  '  + str(test_y.shape))

# extract 1000 samples
data = train_X[:100,:,:]
target_y = train_y[:100]
print(" shape of data : ", data.shape, " and ", len(target_y))
# add noise to the data
data = data.astype(float)
#data += np.random.normal(0, 1,  size= (100, 28, 28))

# normaliser le donnée

multiway = multiwayvs.Multiway_via_spectral(data, k=[10,2,2], norm="centralized")  # k is the number of the eigenvalue considered


cluster = multiway.get_result()  
cluster = cluster[0][0]
print("output : ", set(cluster), "number of elements :", len(cluster))


# uniq target
print("target : ", set(target_y))

# quality
print("Ri quality : ", metrics.adjusted_rand_score(target_y, cluster))




# clustering step
#https://colab.research.google.com/github/goodboychan/goodboychan.github.io/blob/main/_notebooks/2020-10-26-01-K-Means-Clustering-for-Imagery-Analysis.ipynb

