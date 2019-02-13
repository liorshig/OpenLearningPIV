
# coding: utf-8

# # PIV basics 1 

#  start analasys of two picture from piv challenge going through the steps in PIV basics

# In[121]:

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import correlate2d
from skimage.feature import match_template
# load the images: 
a = plt.imread('B005_1.tif')
b = plt.imread('B005_2.tif')
iw = 16
ia = a[:iw,:iw].copy()
ib = b[:iw,:iw].copy()
ia=ia.astype(np.float)
ib=ib.astype(np.float)
#lets find the displacemant
def RMS2D_1(img, template,maxroll=8):
    #CALCULATES the RMS of the difference between img and template at different shifts of img
    #img-2d array of gray scale value of image at time t 
    #template-2d array of gray scale value of image at time t+dt
    #img and template should be same size
    #maxroll- maximun pixel shift of the img 8 pixels is the defult
    #last row/clumn that goes out of array bounds become the first/last row/column
    #i.e the shifted img stays at the same size
    #the function returns a 2d array C at size maxroll+1Xmaxroll+1 wherers
    #C[maxroll,maxroll]-Rms of the difference withount shifts img[i,j] tem[i,j]
    #C[maxroll+x,maxroll+y]-Rms of the difference with shifts img[i+x,j+y] tem[i,j]
    
    cor=np.zeros((maxroll*2+1,maxroll*2+1))
    for row in range(-maxroll,maxroll+1):
        for col in range(-maxroll,maxroll+1):
            norm=np.mean(img)*np.size(img)+1
            cor[maxroll+row,maxroll+col]=np.sqrt(np.sum(np.square(np.roll(img,(row,col),axis=(0,1)) - template))/norm)                  
    return cor
#lets find the displacemant
def RMS2D_2(img, template, maxroll=8):
    #CALCULATES the RMS of the difference between img and template at different shifts of img
    #img-2d array of gray scale value of image at time t 
    #template-2d array of gray scale value of image at time t+dt
    #img and template should be same size
    #maxroll- maximun pixel shift of the img 8 pixels is the defult
    #last row/clumn that goes out of array bounds is thrown away
    #i.e the shifted img becomes smaller with each shift
    #the function returns a 2d array C at size maxroll+1Xmaxroll+1 wherers
    #C[maxroll,maxroll]-Rms of the difference withount shifts img[i,j] tem[i,j]
    #C[maxroll+x,maxroll+y]-Rms of the difference with shifts img[i+x,j+y] tem[i,j]
    cor=np.zeros((maxroll*2+1,maxroll*2+1))
    for row in range(maxroll+1):
        for col in range(maxroll+1):
            if col==0 and row==0:
                norm=np.mean(img)*np.size(img)+1
                cor[maxroll,maxroll]=np.sqrt(np.sum(np.square(template - img))/norm)
            elif col==0 and row!=0:
                norm1=np.mean(np.roll(img,row,axis=0)[row::,:])*np.size(img[row::,:])+1
                norm2=np.mean(img[row::,:])*np.size(img[row::,:])+1
                cor[maxroll+row,maxroll]=np.sqrt(np.sum(np.square(template[row::,:] - np.roll(img,row,axis=0)[row::,:]))/norm1)
                cor[maxroll-row,maxroll]=np.sqrt(np.sum(np.square(img[row::,:] - np.roll(template,row,axis=0)[row::,:]))/norm2)
            elif row==0 and col!=0:
                norm1=np.mean(np.roll(img,col,axis=1)[:,col::])*np.size(img[:,col::])+1
                norm2=np.mean(img[:,col::])*np.size(img[:,col::])+1
                cor[maxroll,maxroll+col]=np.sqrt(np.sum(np.square(template[:,col::] - np.roll(img,col,axis=1)[:,col::]))/norm1)
                cor[maxroll,maxroll-col]=np.sqrt(np.sum(np.square(img[:,col::] - np.roll(template,col,axis=1)[:,col::]))/norm2)
            else:
                norm1=np.mean(np.roll(img,(row,col),axis=(0,1))[row::,col::])*np.size(img[row::,col::])+1
                norm2=np.mean(img[row::,col::])*np.size(img[row::,col::])+1
                norm3=np.mean(np.roll(img,row,axis=0)[row::,col::])*np.size(img[row::,col::])+1
                norm4=np.mean(np.roll(img,col,axis=1)[row::,col::])*np.size(img[row::,col::])+1
                cor[maxroll+row,maxroll+col]=np.sqrt(np.sum(np.square(template[row::,col::] - np.roll(img,(row,col),axis=(0,1))[row::,col::]))/norm1)
                cor[maxroll-row,maxroll-col]=np.sqrt(np.sum(np.square(img[row::,col::] - np.roll(template,(row,col),axis=(0,1))[row::,col::]))/norm2)
                cor[maxroll+row,maxroll-col]=np.sqrt(np.sum(np.square(np.roll(template,col,axis=1)[row::,col::] - np.roll(img,row,axis=0)[row::,col::]))/norm3)
                cor[maxroll-row,maxroll+col]=np.sqrt(np.sum(np.square(np.roll(template,row,axis=0)[row::,col::] - np.roll(img,col,axis=1)[row::,col::]))/norm4)
    return cor
def my_crosscor2d(img, template, maxroll=8):
    #CALCULATES the cross corelation  between img and template at different shifts of img
    #img-2d array of gray scale value of image at time t 
    #template-2d array of gray scale value of image at time t+dt
    #img and template should be same size
    #maxroll- maximun pixel shift of the img 8 pixels is the defult
    #last row/clumn that goes out of array bounds is thrown away
    #i.e the shifted img becomes smaller with each shift
    #the function returns a 2d array C at size maxroll+1Xmaxroll+1 wherers
    #C[maxroll,maxroll]-cross corelation  withount shifts img[i,j] X tem[i,j]
    #C[maxroll+x,maxroll+y]-cross corelation with shifts img[i+x,j+y] X tem[i,j]    
    cor=np.zeros((maxroll*2+1,maxroll*2+1))
    for row in range(maxroll+1):
        for col in range(maxroll+1):
            if col==0 and row==0:
                cor[maxroll,maxroll]=np.sum(template * img)
            elif col==0 and row!=0:
                cor[maxroll+row,maxroll]=np.sum(template[row::,:] * np.roll(img,row,axis=0)[row::,:])
                cor[maxroll-row,maxroll]=np.sum(img[row::,:] * np.roll(template,row,axis=0)[row::,:])
            elif row==0 and col!=0:
                cor[maxroll,maxroll+col]=np.sum(template[:,col::] * np.roll(img,col,axis=1)[:,col::])
                cor[maxroll,maxroll-col]=np.sum(img[:,col::] * np.roll(template,col,axis=1)[:,col::])
            else:
                cor[maxroll+row,maxroll+col]=np.sum(template[row::,col::] * np.roll(img,(row,col),axis=(0,1))[row::,col::])
                cor[maxroll-row,maxroll-col]=np.sum(img[row::,col::] * np.roll(template,(row,col),axis=(0,1))[row::,col::])
                cor[maxroll+row,maxroll-col]=np.sum(np.roll(template,col,axis=1)[row::,col::] * np.roll(img,row,axis=0)[row::,col::])
                cor[maxroll-row,maxroll+col]=np.sum(np.roll(template,row,axis=0)[row::,col::] * np.roll(img,col,axis=1)[row::,col::])
    return cor


# let's test that it works indeed by manually rolling (shifting circurlarly) the same image
cor =RMS2D_1(ia,np.roll(ia,(4,-7),axis=(0,1)))
i,j = np.unravel_index(cor.argmin(), cor.shape)
print('cor2d:%d,%d'%(i-8,j-8))
cor=RMS2D_2(ia,np.roll(ia,(4,-7),axis=(0,1)))
i,j = np.unravel_index(cor.argmin(), cor.shape)
print('cor2d:%d,%d'%(i-8,j-8))
cor=my_crosscor2d(ia-ia.mean(),np.roll(ia,(4,-7),axis=(0,1))-ia.mean())
i,j = np.unravel_index(cor.argmax(), cor.shape)
print('cor2d:%d,%d'%(i-8,j-8))
c = correlate2d(np.roll(ia,(4,-7),axis=(0,1))-ia.mean(),np.roll(ia,(0,0),axis=(0,1))-ia.mean())
i,j = np.unravel_index(c[iw-1-8:iw+8,iw-1-8:iw+8].argmax(), c[iw-1-8:iw+8,iw-1-8:iw+8].shape)
print('cor2d:%d,%d'%(i-8,j-8))
cor1= RMS2D_1(ia,ib)
i,j = np.unravel_index(cor1.argmin(), cor1.shape)
print('RMS1:%d,%d'%(i-8,j-8))
cor2 =RMS2D_2(ia,ib)
i,j = np.unravel_index(cor2.argmin(), cor2.shape)
print('RMS2:%d,%d'%(i-8,j-8))
cor3 = my_crosscor2d(ia-ia.mean(),ib-ib.mean())
i,j = np.unravel_index(cor3.argmax(), cor3.shape)
print('my cross cor:%d,%d'%(i-8,j-8))
cor4 = correlate2d(ib-ib.mean(),ia-ia.mean())
i,j = np.unravel_index(cor4.argmax(), cor4.shape)
print('scipy cross cor:%d,%d'%(i-15,j-15))
fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
x=np.arange(8*2+1)
y=np.arange(8*2+1)
X,Y=np.meshgrid(x,y)
surf1 = ax1.plot_surface(X, Y, cor1, cmap='jet')
ax2 = fig.add_subplot(2, 2, 2, projection='3d')
surf2 = ax2.plot_surface(X, Y, cor2, cmap='jet')
ax3 = fig.add_subplot(2,2, 3, projection='3d')
surf3 = ax3.plot_surface(X, Y, cor3, cmap='jet')
ax4 = fig.add_subplot(2,2, 4, projection='3d')
surf4 = ax4.plot_surface(X, Y, cor4[iw-1-8:iw+8,iw-1-8:iw+8], cmap='jet')
#ax.w_zaxis.set_major_locator(LinearLocator(10))
#ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.03f'))
#fig.colorbar(surf, shrink=0.5, aspect=5)

x,y,u_rms1,v_rms1,u_cor,v_cor,u_rms2,v_rms2,u_mycross,v_mycross = [],[],[],[],[],[],[],[],[],[]
for k in range(0,a.shape[0],iw):
    for m in range(0,a.shape[1],iw):
        ia = a[k:k+iw,m:m+iw].copy()
        ib = b[k:k+iw,m:m+iw].copy()
        ia=ia.astype(np.float)
        ib=ib.astype(np.float)
        cor_rms1 = RMS2D_1(ia,ib)
        row,col = np.unravel_index(cor_rms1.argmin(), cor_rms1.shape)
        y.append(k+iw/2.-1)
        x.append(m+iw/2.-1)
        u_rms1.append(col-8)
        v_rms1.append(row-8)
        cor = correlate2d(ib-ib.mean(),ia-ia.mean())
        cor=cor[iw-1-8:iw+8,iw-1-8:iw+8]
        row,col = np.unravel_index(cor.argmax(), cor.shape)
        u_cor.append(col-8)
        v_cor.append(row-8)
        c_rms2 = RMS2D_2(ia,ib)
        row,col = np.unravel_index(c_rms2.argmin(), c_rms2.shape)
        u_rms2.append(col -8)
        v_rms2.append(row- 8)
        c_mycross = my_crosscor2d(ia-ia.mean(),ib-ib.mean())
        row,col = np.unravel_index(c_mycross.argmax(), c_mycross.shape)
        u_mycross.append(col -8)
        v_mycross.append(row- 8)

plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
M_rms1 = np.sqrt(pow(np.array(u_rms1), 2) + pow(np.array(v_rms1), 2))
plt.quiver(x,y,u_rms1,v_rms1,M_rms1)
plt.title('rms1')

plt.subplot(2,2,2)
M_cor = np.sqrt(pow(np.array(u_cor), 2) + pow(np.array(v_cor), 2))
plt.quiver(x,y,u_cor,v_cor,M_cor)
plt.title('scipy cross cor')

plt.subplot(2,2,3)
M_rms2 = np.sqrt(pow(np.array(u_rms2), 2) + pow(np.array(v_rms2), 2))
plt.quiver(x,y,u_rms2,v_rms2,M_rms2)
plt.title('rms2')

plt.subplot(2,2,4)
M_mycross = np.sqrt(pow(np.array(u_mycross), 2) + pow(np.array(v_mycross), 2))
plt.quiver(x,y,u_mycross,v_mycross,M_mycross)
plt.title('my cross cor')


#the results are abit different lets explore it
q_rms1_rms2= M_rms1==M_rms2
problem_rms1_rms2=np.where(q_rms1_rms2==False)
q_mycrosscor_corsscor= M_mycross==M_cor
problem_mycrosscor_corsscor=np.where(q_mycrosscor_corsscor==False)
q_rms1_corsscor= M_rms1==M_cor
problem_rms1_corsscor=np.where(q_rms1_corsscor==False)
q_rms2_corsscorr= M_rms2==M_cor
problem_rms2_corsscor=np.where(q_rms2_corsscorr==False)
print('number of different points comparing rms1 to rms2:%d'%np.size(problem_rms1_rms2[0]))
print('number of different points coparing my cross cor to scipys:%d'%np.size(problem_mycrosscor_corsscor[0]))
print('number of different points comparing rms1 to crosscor:%d'%np.size(problem_rms1_corsscor[0]))
print('number of different points comparing rms2 to crosscor:%d'%np.size(problem_rms2_corsscor[0]))

#
#k=int(y[problem_i[0][1]]-(iw/2.-1))
#m=int(x[problem_i[0][1]]-(iw/2.-1))
#ia = a[k:k+iw,m:m+iw].copy()
#ib = b[k:k+iw,m:m+iw].copy()
#mindist, idx =RMS2D_1(ia-ia.mean(),ib-ib.mean())
#print('Minimal distance = %f' % mindist)
#print('idx = %d, %d' % idx)
#c = correlate2d(ib-ib.mean(),ia-ia.mean())
#i,j = np.unravel_index(c.argmax(), c.shape)
#print('cor2d:%d,%d'%(i-31,j-31))
#iw = 32
#plt.imshow(c,cmap='gray')
#plt.plot(j,i,'ro')
#plt.plot(31+idx[1],31+idx[0],'bo')
#c[31+idx[0],31+idx[1]]
#c.max()
#dist_diff = np.sqrt(np.sum(np.square(ib - np.roll(ia,(idx[0],idx[1]),axis=(0,1)))))
#dist_cor= np.sqrt(np.sum(np.square(ib - np.roll(ia,(i-31,j-31),axis=(0,1)))))
#
##second problematic point
#k=int(y[problem_i[0][1]]-(iw/2.-1))
#m=int(x[problem_i[0][1]]-(iw/2.-1))
#ia = a[k:k+iw,m:m+iw].copy()
#ib = b[k:k+iw,m:m+iw].copy()
#mindist, idx = RMS2D_1(ia-ia.mean(),ib-ib.mean())
#print('Minimal distance = %f' % mindist)
#print('idx = %d, %d' % idx)
#c = correlate2d(ib-ib.mean(),ia-ia.mean())
#i,j = np.unravel_index(c.argmax(), c.shape)
#print('cor2d:%d,%d'%(i-31,j-31))
#iw = 32
#plt.imshow(c,cmap='gray')
#plt.plot(j,i,'ro')
#plt.plot(31+idx[1],31+idx[0],'bo')
##checklist
#c[31+idx[0],31+idx[1]] #mindif check in crosscorelation
#c.max() #max crosscorraltion 
#np.sum((ib-ib.mean())*(np.roll(ia,(i-31,j-31),axis=(0,1))-ia.mean())) #check if crosscorralatio is miltiplicATION
#np.sum((ib-ib.mean())*(np.roll(ia,(idx[0],idx[1]),axis=(0,1))-ia.mean()))
#dist_diff = np.sqrt(np.sum(np.square(ib - np.roll(ia,(idx[0],idx[1]),axis=(0,1)))))#CHECK MINDIF
#dist_cor= np.sqrt(np.sum(np.square(ib - np.roll(ia,(i-31,j-31),axis=(0,1)))))#CHECK WHAT THE DIF FROM THE CROSSCORRELATION