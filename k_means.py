import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


path1 = os.path.join(os.path.expanduser('~'),'Desktop','Dataset for Programming Exam on Unsupervised Learning','MFCC_N_2500.npy')
path2 = os.path.join(os.path.expanduser('~'),'Desktop','Dataset for Programming Exam on Unsupervised Learning','MFCC_S_2500.npy')


#loading the data set to N & S array
N = np.load(path1)
S = np.load(path2)
print(N.shape)
print(S.shape)


#Appending these two arrays to form the Dataset D=[N,S]
D = np.concatenate((N,S))
print(D.shape)

# K_mean clustering with k=100

Kmean = KMeans(n_clusters=100)
y_mean = Kmean.fit(D)
print(Kmean.cluster_centers_)
print(Kmean.labels_)


# question number 4

N_lebel = Kmean.labels_[:2500]
S_label = Kmean.labels_[2500:]

map_label_to_datapoint_N = {}
for i in range(100):
    map_label_to_datapoint_N[i] = 0

for j in N_lebel:
    map_label_to_datapoint_N[j] += 1


map_label_to_datapoint_S = {}
for i in range(100):
    map_label_to_datapoint_S[i] = 0

for j in S_label:
    map_label_to_datapoint_S[j] += 1

PS = [0 for i in range(100)]
PN = [0 for i in range(100)]
for i in range(100):
    PS[i] = (map_label_to_datapoint_S[i] / (map_label_to_datapoint_S[i]+map_label_to_datapoint_N[i]))*100
    PN[i] = 100 - PS[i]


plt.scatter([i for i in range(100)], PN , color= 'b', marker=".",label='non shout')
plt.scatter([i for i in range(100)], PS , color= 'r', marker="*",label='non shout')
plt.ylabel('Value in percentage')
plt.xlabel('cluster number')
plt.legend()
plt.show()
