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
#lets find the displacemant
def match_template(img, template,maxroll=8):
    mindist = float('inf')
    idx = (0,0)
    cor=np.zeros((maxroll*2+1,maxroll*2+1))
    #first calculate the correaltion without shift
    #cor[maxroll,maxroll]=np.sqrt(np.sum(np.square(np.roll(template,(0,0),axis=(0,1)) - np.roll(img,(0,0),axis=(0,1)))))
     #calculate Euclidean distance:
    for row in range(-maxroll,maxroll+1):
        for col in range(-maxroll,maxroll+1):
            cor[maxroll+row,maxroll+col]=np.sqrt(np.sum(np.square(np.roll(img,(row,col),axis=(0,1)) - template)))                  
    return cor
#lets find the displacemant
def my_correlate2d(img, template, maxroll=8):
    cor=np.zeros((maxroll*2+1,maxroll*2+1))
    for row in range(maxroll+1):
        for col in range(maxroll+1):
            if col==0 and row==0:
                cor[maxroll,maxroll]=np.sqrt(np.sum(np.square(template - img)))
            elif col==0 and row!=0:
                cor[maxroll+row,maxroll]=np.sqrt(np.sum(np.square(template[row::,:] - np.roll(img,row,axis=0)[row::,:])))
                cor[maxroll-row,maxroll]=np.sqrt(np.sum(np.square(img[row::,:] - np.roll(template,row,axis=0)[row::,:])))
            elif row==0 and col!=0:
                cor[maxroll,maxroll+col]=np.sqrt(np.sum(np.square(template[:,col::] - np.roll(img,col,axis=1)[:,col::])))
                cor[maxroll,maxroll-col]=np.sqrt(np.sum(np.square(img[:,col::] - np.roll(template,col,axis=1)[:,col::])))
            else:
                cor[maxroll+row,maxroll+col]=np.sqrt(np.sum(np.square(template[row::,col::] - np.roll(img,(row,col),axis=(0,1))[row::,col::])))
                cor[maxroll-row,maxroll-col]=np.sqrt(np.sum(np.square(img[row::,col::] - np.roll(template,(row,col),axis=(0,1))[row::,col::])))
                cor[maxroll+row,maxroll-col]=np.sqrt(np.sum(np.square(np.roll(template,col,axis=1)[row::,col::] - np.roll(img,row,axis=0)[row::,col::])))
                cor[maxroll-row,maxroll+col]=np.sqrt(np.sum(np.square(np.roll(template,row,axis=0)[row::,col::] - np.roll(img,col,axis=1)[row::,col::])))
    return cor
iw=32
x,y,u_dif,v_dif,u_cor,v_cor,u_my,v_my = [],[],[],[],[],[],[],[]
for k in range(0,a.shape[0],iw):
    for m in range(0,a.shape[1],iw):
        ia = a[k:k+iw,m:m+iw].copy()
        ib = b[k:k+iw,m:m+iw].copy()
        cor_dif = match_template(ia-ia.mean(),ib-ib.mean())
        row,col = np.unravel_index(cor_dif.argmin(), cor_dif.shape)
        y.append(k+iw/2.-1)
        x.append(m+iw/2.-1)
        u_dif.append(col-8)
        v_dif.append(row-8)
        c = correlate2d(ib-ib.mean(),ia-ia.mean())
        c=c[31-8:32+8,31-8:32+8]
        i,j = np.unravel_index(c.argmax(), c.shape)
        u_cor.append(j -8)
        v_cor.append(i- 8)
        c_my = my_correlate2d(ia-ia.mean(),ib-ib.mean())
        row,col = np.unravel_index(c_my.argmin(), c_my.shape)
        u_my.append(col -8)
        v_my.append(row- 8)
plt.figure(figsize=(12,10))
M_dif = np.sqrt(pow(np.array(u_dif), 2) + pow(np.array(v_dif), 2))
plt.quiver(x,y,u_dif,v_dif,M_dif)
q_dif=(100./1650)*M_dif/(15.*10**(-6))
plt.figure(figsize=(12,10))
M_cor = np.sqrt(pow(np.array(u_cor), 2) + pow(np.array(v_cor), 2))
q_cor=(100./1650)*M_cor/(15.*10**(-6))
plt.quiver(x,y,u_cor,v_cor,M_cor)

plt.figure(figsize=(12,10))
M_my = np.sqrt(pow(np.array(u_my), 2) + pow(np.array(v_my), 2))
q_my=(100./1650)*M_my/(15.*10**(-6))
plt.quiver(x,y,u_my,v_my,M_my)