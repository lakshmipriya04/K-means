import numpy as np
from scipy.spatial import distance_matrix
import os
import sys

filename=sys.argv[1]
k=int(sys.argv[2])


a=np.loadtxt(filename)
n,d=a.shape

init_centroid=np.random.choice(n,k,replace=False)
centroid=a[init_centroid]
c_old=np.zeros((k,d))
cluster_assign=np.zeros(n)

while (c_old!=centroid).any():
    c_old=centroid.copy()
    d_matrix=distance_matrix(a,centroid,p=2)
    for i in np.arange(n):
        d=d_matrix[i]
        close_centroid=(np.where(d==np.min(d)))[0][0]
        cluster_assign[i]=close_centroid
    for j in np.arange(k):
        ak=a[cluster_assign==j]
        centroid[j]=np.apply_along_axis(np.mean,axis=0,arr=ak)

np.savetxt('clusters.txt',centroid)
print(centroid)
