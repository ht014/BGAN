import numpy as np
import scipy.io as sio
import os

data = np.load('cifar_KNN.npz')['topK']# cifar_KNN.npz is the first KNN rank ,the top K nearest neighbor of  each image
size = data.shape[0]
K1 = 20
K2 = 30
S = np.ones((size,size))*-1 
for i in xrange(size):
    top = set(data[i][:K1+1]) #exclude ifself so add one
    nums=[]
    idx = []
    for j in xrange(size):
         si = set(data[j][:K1+1])
         both = si & top
         nums.append(len(both))
         idx.append(j)
    nums = np.array(nums)
    indx = np.argsort(nums*-1)[:K2]
    sp=set()
    for ii in indx:
        idx_ = idx[ii]
        sp = sp|set(data[idx_][:K1+1])
    for  j in sp:
         S[i][j] = 1.
         S[j][i] = 1.
sio.savemat('S_K1_'+str(K1)+'_K2_'+str(K2)+'.mat',{'S':S})

