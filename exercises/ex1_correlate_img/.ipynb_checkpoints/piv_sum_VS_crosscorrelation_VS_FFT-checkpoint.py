
# coding: utf-8

# # PIV basics 1 

#  start analasys of two picture from piv challenge going through the steps in PIV basics

# In[121]:


import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import correlate2d
# load the images: 
a = plt.imread('B005_1.tif')
b = plt.imread('B005_2.tif')

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


# let's test that it works indeed by manually rolling (shifting circurlarly) the same image
cor =match_template(ia-ia.mean(),np.roll(ia,(4,-7),axis=(0,1))-ia.mean())
i,j = np.unravel_index(cor.argmin(), cor.shape)
print('cor2d:%d,%d'%(i-8,j-8))
cor=my_correlate2d(ia-ia.mean(),np.roll(ia,(4,-7),axis=(0,1))-ia.mean())
i,j = np.unravel_index(cor.argmin(), cor.shape)
print('cor2d:%d,%d'%(i-8,j-8))
c = correlate2d(np.roll(ia,(4,-7),axis=(0,1))-ia.mean(),np.roll(ia,(0,0),axis=(0,1))-ia.mean())
i,j = np.unravel_index(c[31-8:32+8,31-8:32+8].argmax(), c[31-8:32+8,31-8:32+8].shape)
print('cor2d:%d,%d'%(i-8,j-8))
cor= match_template(ia-ia.mean(),ib-ib.mean())
i,j = np.unravel_index(cor.argmin(), cor.shape)
print('cor2d:%d,%d'%(i-8,j-8))
cor = my_correlate2d(ia-ia.mean(),ib-ib.mean())
i,j = np.unravel_index(cor.argmin(), cor.shape)
print('cor2d:%d,%d'%(i-8,j-8))
c = correlate2d(ib-ib.mean(),ia-ia.mean())
i,j = np.unravel_index(c.argmax(), c.shape)
print('cor2d:%d,%d'%(i-31,j-31))
iw = 32

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

plt.figure(figsize=(12,10))
M_cor = np.sqrt(pow(np.array(u_cor), 2) + pow(np.array(v_cor), 2))
plt.quiver(x,y,u_cor,v_cor,M_cor)

plt.figure(figsize=(12,10))
M_my = np.sqrt(pow(np.array(u_my), 2) + pow(np.array(v_my), 2))
plt.quiver(x,y,u_my,v_my,M_my)
#the results are abit different lets explore it
q_dif_my= np.array(M_dif)==np.array(M_my)
problem_i=np.where(q_dif_my==False)
k=int(y[problem_i[0][1]]-(iw/2.-1))
m=int(x[problem_i[0][1]]-(iw/2.-1))
ia = a[k:k+iw,m:m+iw].copy()
ib = b[k:k+iw,m:m+iw].copy()
mindist, idx =match_template(ia-ia.mean(),ib-ib.mean())
print('Minimal distance = %f' % mindist)
print('idx = %d, %d' % idx)
c = correlate2d(ib-ib.mean(),ia-ia.mean())
i,j = np.unravel_index(c.argmax(), c.shape)
print('cor2d:%d,%d'%(i-31,j-31))
iw = 32
plt.imshow(c,cmap='gray')
plt.plot(j,i,'ro')
plt.plot(31+idx[1],31+idx[0],'bo')
c[31+idx[0],31+idx[1]]
c.max()
dist_diff = np.sqrt(np.sum(np.square(ib - np.roll(ia,(idx[0],idx[1]),axis=(0,1)))))
dist_cor= np.sqrt(np.sum(np.square(ib - np.roll(ia,(i-31,j-31),axis=(0,1)))))

#second problematic point
k=int(y[problem_i[0][1]]-(iw/2.-1))
m=int(x[problem_i[0][1]]-(iw/2.-1))
ia = a[k:k+iw,m:m+iw].copy()
ib = b[k:k+iw,m:m+iw].copy()
mindist, idx = match_template(ia-ia.mean(),ib-ib.mean())
print('Minimal distance = %f' % mindist)
print('idx = %d, %d' % idx)
c = correlate2d(ib-ib.mean(),ia-ia.mean())
i,j = np.unravel_index(c.argmax(), c.shape)
print('cor2d:%d,%d'%(i-31,j-31))
iw = 32
plt.imshow(c,cmap='gray')
plt.plot(j,i,'ro')
plt.plot(31+idx[1],31+idx[0],'bo')
#checklist
c[31+idx[0],31+idx[1]] #mindif check in crosscorelation
c.max() #max crosscorraltion 
np.sum((ib-ib.mean())*(np.roll(ia,(i-31,j-31),axis=(0,1))-ia.mean())) #check if crosscorralatio is miltiplicATION
np.sum((ib-ib.mean())*(np.roll(ia,(idx[0],idx[1]),axis=(0,1))-ia.mean()))
dist_diff = np.sqrt(np.sum(np.square(ib - np.roll(ia,(idx[0],idx[1]),axis=(0,1)))))#CHECK MINDIF
dist_cor= np.sqrt(np.sum(np.square(ib - np.roll(ia,(i-31,j-31),axis=(0,1)))))#CHECK WHAT THE DIF FROM THE CROSSCORRELATION