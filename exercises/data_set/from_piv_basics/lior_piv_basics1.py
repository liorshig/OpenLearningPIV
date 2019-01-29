# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 22:54:24 2019

@author: lior
"""

import numpy as np
from matplotlib import pyplot as plt

# load the images: 
a = plt.imread('B005_1.tif')
b = plt.imread('B005_2.tif')
plt.figure(figsize=(12,10))
plt.subplot(1,2,1)
plt.imshow(a,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(b,cmap="gray")
#interrogation windows. 
#typically we can start with the size of 128 x 128 pixels or smaller.
size=32
ia = a[:size,:size].copy()
ib = b[:size,:size].copy()
plt.figure()
plt.subplot(1,2,1)
plt.imshow(ia,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(ib,cmap='gray')
#lets start with simple substraction to find the displacemants between a to b
plt.imshow(ib-ia,cmap='gray')
plt.title('Without shift')
# lets shift ia by 1 pixle down
plt.imshow(ib-np.roll(ia,1,axis=0),cmap='gray')
plt.title('Difference when IA has been shifted by 1 pixel')
#lets find the displacemant
def match_template(img, template,maxroll=8):
    mindist = float('inf')
    idx = (-1,-1)
    for y in range(maxroll):
        for x in range(maxroll):
        #calculate Euclidean distance
            dist = np.sqrt(np.sum(np.square(template - np.roll(img,(x,y),axis=(0,1)))))
            if dist < mindist:
                mindist = dist
                idx = (x,y)
                
    return [mindist, idx]
# let's test that it works indeed by manually rolling (shifting circurlarly) the same image
match_template(ia,np.roll(ia,2,axis=0))
# indeed, when we find the correct shift, we got zero distance. it's not so in real images:
mindist, idx = match_template(ia,ib)
print('Minimal distance = %f' % mindist)
print('idx = %d, %d' % idx)
#lets check if it looks the same
plt.figure()
plt.subplot(1,2,1)
plt.imshow(np.roll(ia,idx,(0,1)),cmap='gray')
plt.subplot(1,2,2)
plt.imshow(ib,cmap='gray')
#lets try cross coralation
from scipy.signal import correlate2d
c = correlate2d(ia-ia.mean(),ib-ib.mean())
# not it's twice bigger than the original windows, as we can shift ia by maximum it's size horizontally and vertically
print('Size of the correlation map %d x %d' % c.shape)
# let's see how the correlation map looks like:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cx,cy = np.meshgrid(range(c.shape[0]), range(c.shape[1]))

ax.plot_surface(cx,cy,c,cmap='jet', linewidth=0.2)
plt.title('Correlation map - peak is the most probable shift')
# let's see the same correlation map, from above
fig = plt.figure()
plt.imshow(c, cmap='gray')

i,j = np.unravel_index(c.argmax(), c.shape)

print('i = {}, j= {}'.format(i,j))

plt.plot(j,i,'ro')
iw = 32

x,y,u,v = [],[],[],[]
for k in range(0,a.shape[0],iw):
    for m in range(0,a.shape[1],iw):
        ia = a[k:k+iw,m:m+iw]
        ib = b[k:k+iw,m:m+iw]
        c = correlate2d(ia-ia.mean(),ib-ib.mean())
        i,j = np.unravel_index(c.argmax(), c.shape)
        x.append(k-iw/2.)
        y.append(m-iw/2.)
        u.append(i - iw/2.-1)
        v.append(j - iw/2.-1)
plt.figure(figsize=(12,10))
M = np.sqrt(pow(np.array(u), 2) + pow(np.array(v), 2))
plt.quiver(x,y,u,v,M)
