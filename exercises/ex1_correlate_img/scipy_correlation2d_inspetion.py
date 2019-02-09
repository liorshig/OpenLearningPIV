# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 17:51:10 2019

@author: lior
"""
# inspection cross corelation function
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import correlate2d

a=plt.imread('B005_1.tif')
b=plt.imread('B005_2.tif')
ia=a[0:32,0:32].copy()
ib=b[0:32,0:32].copy()
c_0_0=np.sum((ia-ia.mean())*(ib-ib.mean()))
c_0_1=np.sum((ia-ia.mean())*(np.roll(ib,1,axis=1)-ib.mean()))
c_0_ng1=np.sum((ia-ia.mean())*(np.roll(ib,-1,axis=1)-ib.mean()))
c_1_1=np.sum((ia-ia.mean())*(np.roll(ib,(1,1),axis=(0,1))-ib.mean()))
#for row in range
c1=correlate2d(ib-ib.mean(),ia-ia.mean())
c2=correlate2d(ia-ia.mean(),ib-ib.mean())
c1[31,31]
c1[31,32]
c1[31,30]
c2[31,31]
c2[31,32]
c2[31,30]
c_0_1=np.sum((ia[:,1::]-ia.mean())*(np.roll(ib,1,axis=1)[:,1::]-ib.mean()))
c_15_0=np.sum((ia[30::,:]-ia.mean())*(np.roll(ib,30,axis=0)[30::,:]-ib.mean()))
c2[31+30,31]
c2[31+15,31]