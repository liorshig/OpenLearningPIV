
# coding: utf-8

# # PIV basics 1 

#  start analasys of two picture from piv challenge going through the steps in PIV basics

# In[26]:


import numpy as np
from matplotlib import pyplot as plt


# In[27]:



# load the images: 
a = plt.imread('B005_1.tif')
b = plt.imread('B005_2.tif')
plt.figure(figsize=(12,10))
plt.subplot(1,2,1)
plt.imshow(a,cmap="gray")
plt.subplot(1,2,2)
plt.imshow(b,cmap="gray")


# In[28]:


#interrogation windows. 
size=32 #size 32X32
ia = a[:size,:size].copy()
ib = b[:size,:size].copy()
plt.figure()
plt.subplot(1,2,1)
plt.imshow(ia,cmap='gray')
plt.subplot(1,2,2)
plt.imshow(ib,cmap='gray')


# let's look at the sustraction of both sub images: $im_a-im_b$ in order to spot the differences

# In[29]:


#lets start with simple substraction to find the displacemants between a to b
plt.imshow(ib-ia,cmap='gray')
plt.title('Without shift')


# So the images are defenetley not the same. lets try to shift $image_a$ by 1 pixel 

# In[30]:


# lets shift ia by 1 pixle down
plt.imshow(ib-np.roll(ia,1,axis=0),cmap='gray')
plt.title('Difference when IA has been shifted by 1 pixel')


# let's try to find the displacemant by going through all the combinations considering the maximun posible diplacemant is 8 pixels in each axis.

# In[31]:


#lets find the displacemant
def match_template(img, template,maxroll=8):
    mindist = float('inf')
    idx = (-1,-1)
    for y in range(-maxroll,maxroll+1):
        for x in range(-maxroll,maxroll+1):
        #calculate Euclidean distance
            dist = np.sqrt(np.sum(np.square(template - np.roll(img,(x,y),axis=(0,1)))))
            if dist < mindist:
                mindist = dist
                idx = (x,y)
                
    return [mindist, idx]


# In[32]:


# let's test that it works indeed by manually rolling (shifting circurlarly) the same image
match_template(ia,np.roll(ia,2,axis=0))


# it worked! we got zero distance and a displacemant of 2 pixels exactly what we expected.
# let's apply it on the sub pictures.

# In[33]:


# indeed, when we find the correct shift, we got zero distance. it's not so in real images:
mindist, idx = match_template(ia,ib)
print('Minimal distance = %f' % mindist)
print('idx = %d, %d' % idx)


# firstly, the minimal distance is defentely not zero. But maybe it's good enough lets see the substraction on $im_b(i,j)-im_a(i,j+1)$

# In[34]:


#lets check if it looks the same
plt.figure()
plt.subplot(1,2,1)
plt.imshow(np.roll(ia,idx,(0,1)),cmap='gray')
plt.subplot(1,2,2)
plt.imshow(ib,cmap='gray')
plt.figure()
plt.imshow(np.roll(ia,idx,(0,1))-ib,cmap='gray')


# Well it's not the exact shift indeed, but it is closer than before.
# let's find all the shifts and time it using the simple algoritm we intreduced here.

# In[67]:


i,j=np.shape(a)
print(i)
print(j)
size=32
Xwin_num=i/size #number of windows given size of sizeXsize pixels of each window
Ywin_num=j/size # "---"
cor, dispx, dispy=-10*np.ones([Xwin_num,Ywin_num]), -10*np.ones([Xwin_num,Ywin_num]), -10*np.ones([Xwin_num,Ywin_num])
for X in range(Xwin_num):
    for Y in range(Ywin_num):
        #print(X,Y)
        ia=a[X*size:(X+1)*size,Y*size:(Y+1)*size].copy()
        ib=b[X*size:(X+1)*size,Y*size:(Y+1)*size].copy()
        cor[X,Y], (dispx[X,Y],dispy[X,Y]) = match_template(ia-np.mean(ia),ib-np.mean(ib))
        

        
    


# In[68]:


plt.figure(figsize=(12,10))
x = np.arange(size/2-1, i, size)
y = np.arange(size/2-1, i, size)
Y, X = np.meshgrid(y, x)
M = np.sqrt(pow(dispx, 2) + pow(dispy, 2))
q = plt.quiver(X, Y, dispx, dispy,M)
plt.show()

q = plt.quiver(X, Y, np.ones([i,j]), np.zeros([i,j]),M)
# ## lets try using cross corelation

# In[17]:


from scipy.signal import correlate2d
ia = a[:size,:size].copy()
ib = b[:size,:size].copy()
c = correlate2d(ia-ia.mean(),ib-ib.mean())
# not it's twice bigger than the original windows, as we can shift ia by maximum it's size horizontally and vertically
print('Size of the correlation map %d x %d' % c.shape)


# In[18]:


# let's see how the correlation map looks like:
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cx,cy = np.meshgrid(range(c.shape[0]), range(c.shape[1]))

ax.plot_surface(cx,cy,c,cmap= "jet", linewidth=0.2)
plt.title('Correlation map - peak is the most probable shift')


# In[19]:


plt.imshow(c, cmap='gray')

i,j = np.unravel_index(c.argmax(), c.shape)

print('i = {}, j= {}'.format(i,j))

plt.plot(j,i,'ro')


# In[69]:


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
        u.append(i - 31)
        v.append(j - 31)


# In[70]:


plt.figure(figsize=(12,10))
M = np.sqrt(pow(np.array(u), 2) + pow(np.array(v), 2))
plt.quiver(x,y,u,v,M)


# In[65]:

iw = 32

dispx,dispy,dispu,dispv = [],[],[],[]
for k in range(0,a.shape[0],iw):
    for m in range(0,a.shape[1],iw):
        ia = a[k:k+iw,m:m+iw]
        ib = b[k:k+iw,m:m+iw]
        cor, (ix,iy) = match_template(ia-np.mean(ia),ib-np.mean(ib))
        dispx.append(k+iw/2.)
        dispy.append(m+iw/2.)
        dispu.append(ix)
        dispv.append(iy)
plt.figure(figsize=(12,10))
dispM = np.sqrt(pow(np.array(dispu), 2) + pow(np.array(dispv), 2))
plt.quiver(dispx,dispy,dispu,dispv,dispM)