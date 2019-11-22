#import h5py
import numpy as np
from PIL import Image
#from matplotlib import pyplot as plt
import os
from sklearn.cluster import KMeans

import scipy.io as sio


def get3AttributeMaskFromLabels(labels,LengthAttributs,N):

    mask11=np.zeros((3*LengthAttributs))
    mask21=np.ones((3*LengthAttributs))
    for k in range(LengthAttributs*2,LengthAttributs*3):
        mask11[k]=1  #001
        mask21[k]=0  #110

    mask12=np.zeros((3*LengthAttributs))
    mask22=np.ones((3*LengthAttributs))
    for k in range(LengthAttributs,LengthAttributs*2):
        mask12[k]=1  #010
        mask22[k]=0  #101

    mask13=np.zeros((3*LengthAttributs))
    mask23=np.ones((3*LengthAttributs))
    for k in range(0,LengthAttributs):
        mask13[k]=1  #100
        mask23[k]=0  #011

    mask1Array=np.zeros((N,3*LengthAttributs))
    mask2Array=np.zeros((N,3*LengthAttributs))
    for k in range(N):
        if labels[k]==0:
            mask1Array[k,:]=mask11
            mask2Array[k,:]=mask21 
        elif labels[k]==1:
            mask1Array[k,:]=mask12
            mask2Array[k,:]=mask22 
        elif labels[k]==2 or labels[k]==3 :
            mask1Array[k,:]=mask13
            mask2Array[k,:]=mask23
    return  mask1Array,mask2Array 



def getMask5FromLabels(labels,LengthAttributs,N):

    mask11=np.zeros((5*LengthAttributs))
    mask21=np.ones((5*LengthAttributs))
    for k in range(LengthAttributs*4,LengthAttributs*5):
        mask11[k]=1  #00001
        mask21[k]=0  #11110

    mask12=np.zeros((5*LengthAttributs))
    mask22=np.ones((5*LengthAttributs))
    for k in range(LengthAttributs*3,LengthAttributs*4):
        mask12[k]=1  #00010
        mask22[k]=0  #11101

    mask13=np.zeros((5*LengthAttributs))
    mask23=np.ones((5*LengthAttributs))
    for k in range(LengthAttributs*2,LengthAttributs*3):
        mask13[k]=1 #00100
        mask23[k]=0 #11011

    mask14=np.zeros((5*LengthAttributs))
    mask24=np.ones((5*LengthAttributs))
    for k in range(LengthAttributs*1,LengthAttributs*2):
        mask14[k]=1 #01000
        mask24[k]=0 #10111

    mask15=np.zeros((5*LengthAttributs))
    mask25=np.ones((5*LengthAttributs))
    for k in range(0,LengthAttributs*1):
        mask15[k]=1 #10000
        mask25[k]=0 #01111

    mask1Array=np.zeros((N,5*LengthAttributs))
    mask2Array=np.zeros((N,5*LengthAttributs))
    for k in range(N):
        if labels[k]==0:
            mask1Array[k,:]=mask11
            mask2Array[k,:]=mask21 
        elif labels[k]==1:
            mask1Array[k,:]=mask12
            mask2Array[k,:]=mask22 
        elif labels[k]==2:
            mask1Array[k,:]=mask13
            mask2Array[k,:]=mask23
        elif labels[k]==3:
            mask1Array[k,:]=mask14
            mask2Array[k,:]=mask24
        elif labels[k]==4:
            mask1Array[k,:]=mask15
            mask2Array[k,:]=mask25

    return  mask1Array,mask2Array 
