
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
    for row in range(-maxroll,maxroll+1):
        for col in range(-maxroll,maxroll+1):
        #calculate Euclidean distance
            dist = np.sqrt(np.sum(np.square(template - np.roll(img,(row,col),axis=(0,1)))))
            if dist < mindist:
                mindist = dist
                idx = (row,col)
    return [mindist, idx]
#lets find the displacemant
def my_correlate2d(img, template):
    cor=np.zeros(2*np.shape(img)[0],2*np.shape(img)[0])
    for row in range(-maxroll,maxroll+1):
        for col in range(-maxroll,maxroll+1):
            mindist=-float('inf')
        #calculate Euclidean distance
            #dist =np.abs(np.sum(template*np.roll(img,(y,x),axis=(0,1))))/((np.sqrt(np.sum(template**2)))*(np.sqrt(np.sum(np.roll(img,(y,x),axis=(0,1))**2))))
            cor[row,col]=np.sum(template*np.roll(img,(row,col),axis=(0,1)))
#            if cor > mindist:
#                mindist = cor
#                idx = (row,col)
                
    return cor


# let's test that it works indeed by manually rolling (shifting circurlarly) the same image
mindist, idx =match_template(ia-ia.mean(),np.roll(ia,(3,7),axis=(0,1))-ia.mean())
print('Minimal distance = %f' % mindist)
print('idx = %d, %d' % idx)
cor=my_correlate2d(ia-ia.mean(),np.roll(ia,(2,3),axis=(0,1))-ia.mean())
i,j = np.unravel_index(cor.argmax(), cor.shape)
print('cor2d:%d,%d'%(i,j))
c = correlate2d(np.roll(ia,(1,8),axis=(0,1))-ia.mean(),np.roll(ia,(0,0),axis=(0,1))-ia.mean())
i,j = np.unravel_index(c[31-8:32+8,31-8:32+8].argmax(), c[31-8:32+8,31-8:32+8].shape)
print('cor2d:%d,%d'%(i-8,j-8))
mindist, idx = match_template(ia,ib)
print('Minimal distance = %f' % mindist)
print('idx = %d, %d' % idx)
mindist, idx = my_correlate(ia,ib)
print('Minimal distance = %f' % mindist)
print('idx = %d, %d' % idx)
c = correlate2d(ib-ib.mean(),ia-ia.mean())
i,j = np.unravel_index(c.argmax(), c.shape)
print('cor2d:%d,%d'%(i-31,j-31))
iw = 32

x,y,u_dif,v_dif,u_cor,v_cor = [],[],[],[],[],[]
for k in range(0,a.shape[0],iw):
    for m in range(0,a.shape[1],iw):
        ia = a[k:k+iw,m:m+iw].copy()
        ib = b[k:k+iw,m:m+iw].copy()
        cor, (i,j) = match_template(ia-ia.mean(),ib-ib.mean())
        y.append(k+iw/2.-1)
        x.append(m+iw/2.-1)
        u_dif.append(i)
        v_dif.append(j)
        c = correlate2d(ib-ib.mean(),ia-ia.mean())
        c=c[31-8:32+8,31-8:32+8]
        i,j = np.unravel_index(c.argmax(), c.shape)
        u_cor.append(i -8)
        v_cor.append(j- 8)

plt.figure(figsize=(12,10))
M_dif = np.sqrt(pow(np.array(u_dif), 2) + pow(np.array(v_dif), 2))
plt.quiver(x,y,v_dif,u_dif,M)

plt.figure(figsize=(12,10))
M_cor = np.sqrt(pow(np.array(u_cor), 2) + pow(np.array(v_cor), 2))
plt.quiver(x,y,v_cor,u_cor,M)
#the results are abit different les find why
q= np.array(M_dif)==np.array(M_cor)
problem_i=np.where(q==False)
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