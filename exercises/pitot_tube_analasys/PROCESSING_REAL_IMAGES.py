# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 21:09:49 2019

@author: lior
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import correlate2d

# load the images: 
im1 = plt.imread('pitot_yoav000716.T000.D000.P000.H000.LA.TIF')
im2 = plt.imread('pitot_yoav000716.T000.D000.P000.H000.LB.TIF')
plt.imshow(im1,cmap='gray',vmin=0,vmax=250)
plt.plot(1600*np.ones(1750-1350),np.arange(1350,1750),'r')
plt.plot(2100*np.ones(1750-1350),np.arange(1350,1750),'r')
plt.plot(np.arange(1600,2100), 1350*np.ones(2100-1600),'r')
plt.plot(np.arange(1600,2100), 1750*np.ones(2100-1600),'r')
plt.imshow(im1[1350:1750,1600:2100],cmap='gray',vmin=0,vmax=250)
plt.imshow(im2,cmap='gray',vmin=0,vmax=250)
plt.plot(1600*np.ones(1750-1350),np.arange(1350,1750),'r')
plt.plot(2100*np.ones(1750-1350),np.arange(1350,1750),'r')
plt.plot(np.arange(1600,2100), 1350*np.ones(2100-1600),'r')
plt.plot(np.arange(1600,2100), 1750*np.ones(2100-1600),'r')
plt.imshow(im2[1350:1750,1600:2100],cmap='gray',vmin=0,vmax=250)
a=im1[1350:1750,1600:2100]
b=im2[1350:1750,1600:2100]
from my_cor_fun import RMS2D_1, RMS2D_2, my_crosscor2d
iw=32
pixtomm=(100./1650)*1e-3 #[m/pixels]
dt=15e-6 # time between to frames [sec]
calib=pixtomm/dt
x,y,u_rms,v_rms,u_cor,v_cor,u_my,v_my = [],[],[],[],[],[],[],[]
for k in range(0,a.shape[0],iw):
    for m in range(0,a.shape[1],iw):
        ia = a[k:k+iw,m:m+iw].copy()
        ib = b[k:k+iw,m:m+iw].copy()
        cor1 = RMS2D_1(ia-ia.mean(),ib-ib.mean())
        row,col = np.unravel_index(cor1.argmin(), cor1.shape)
        y.append(k+iw/2.-1)
        x.append(m+iw/2.-1)
        u_rms.append(col-8)
        v_rms.append(row-8)
        c = correlate2d(ib-ib.mean(),ia-ia.mean())
        c=c[31-8:32+8,31-8:32+8]
        i,j = np.unravel_index(c.argmax(), c.shape)
        u_cor.append(j -8)
        v_cor.append(i- 8)
        cor_my = my_crosscor2d(ia-ia.mean(),ib-ib.mean())
        row,col = np.unravel_index(cor_my.argmax(), cor_my.shape)
        u_my.append(col -8)
        v_my.append(row- 8)
plt.figure(figsize=(12,10))
M_rms = np.sqrt(pow(calib*np.array(u_rms), 2) + pow(calib*np.array(v_rms), 2))
plt.quiver(x,y,u_rms,v_rms,M_rms)
plt.figure(figsize=(12,10))
M_cor = np.sqrt(pow(calib*np.array(u_cor), 2) + pow(calib*np.array(v_cor), 2))
plt.quiver(x,y,u_cor,v_cor,M_cor)

plt.figure(figsize=(12,10))
M_my = np.sqrt(pow(calib*np.array(u_my), 2) + pow(calib*np.array(v_my), 2))
plt.quiver(x,y,u_my,v_my,M_my)