import math
import tensorflow as tf
import prettytensor as pt
import numpy as np
from deconv import deconv2d
import IPython.display
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed
import skimage
import skimage.io
import skimage.transform
import tqdm # making loops prettier
import h5py
import scipy
import cv2
import scipy.io as sio

def cal_map(rel,K=None):
    shot = 0.
    ans = 0.
    if K is not  None:
        rel= rel[:K]
    pos = 1.
    for i in rel:
        shot += i
        ans +=(shot/pos)*i
        pos += 1.0
    return ans / shot


#binary data dataset
feat = np.load('codes/32_00020_beta.npz')



mat = sio.loadmat('cifar-10.mat')
dataset = np.sign(feat['dataset'])
dataset_L = mat['dataset_L']
test_L = mat['test_L']
test = np.sign(feat['test'])
'''
nms= 'codes/48_3_00030'#"SphericalH_12_resnet_cifar"
data = np.load(nms+'.npz')
dataset=data['B_dataset']
dataset_L=data['dataset_L']
test_L=data['test_L']
test = data['B_test']
'''
print dataset.shape,test.shape,dataset_L.shape,test_L.shape,'DSDD'
s = 0.
for i in xrange(len(test_L)):
    query = test[i]
    sim=[]
    query_L=test_L[i]
    for j in xrange(len(dataset)):
        a = np.sum(query*dataset[j]) / (np.sqrt(np.sum(query**2))*np.sqrt(np.sum(dataset[j]**2)))
        sim.append(a)
    sim = np.array(sim)
    indx = np.argsort(-sim)

    rel=[]
    for j in indx:
        if query_L==dataset_L[j]:
            rel.append(1)
        else:
            rel.append(0)

    mp1 = cal_map(rel,len(dataset))
    s +=mp1
    print i,'ap=',mp1,"Map=",s/(i+1.)




