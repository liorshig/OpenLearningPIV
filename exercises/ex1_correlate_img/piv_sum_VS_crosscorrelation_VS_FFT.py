
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
iw = 32
ia = a[:iw,:iw].copy()
ib = b[:iw,:iw].copy()

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
    mindist = float('inf')
    idx = (0,0)
    cor=np.zeros((maxroll*2+1,maxroll*2+1))
    for row in range(-maxroll,maxroll+1):
        for col in range(-maxroll,maxroll+1):
            cor[maxroll+row,maxroll+col]=np.sqrt(np.sum(np.square(np.roll(img,(row,col),axis=(0,1)) - template)))                  
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
cor =RMS2D_1(ia-ia.mean(),np.roll(ia,(4,-7),axis=(0,1))-ia.mean())
i,j = np.unravel_index(cor.argmin(), cor.shape)
print('cor2d:%d,%d'%(i-8,j-8))
cor=RMS2D_2(ia-ia.mean(),np.roll(ia,(4,-7),axis=(0,1))-ia.mean())
i,j = np.unravel_index(cor.argmin(), cor.shape)
print('cor2d:%d,%d'%(i-8,j-8))
cor=my_crosscor2d(ia-ia.mean(),np.roll(ia,(4,-7),axis=(0,1))-ia.mean())
i,j = np.unravel_index(cor.argmax(), cor.shape)
print('cor2d:%d,%d'%(i-8,j-8))
c = correlate2d(np.roll(ia,(4,-7),axis=(0,1))-ia.mean(),np.roll(ia,(0,0),axis=(0,1))-ia.mean())
i,j = np.unravel_index(c[31-8:32+8,31-8:32+8].argmax(), c[31-8:32+8,31-8:32+8].shape)
print('cor2d:%d,%d'%(i-8,j-8))
cor= RMS2D_1(ia-ia.mean(),ib-ib.mean())
i,j = np.unravel_index(cor.argmin(), cor.shape)
print('cor2d:%d,%d'%(i-8,j-8))
cor =RMS2D_2(ia-ia.mean(),ib-ib.mean())
i,j = np.unravel_index(cor.argmin(), cor.shape)
print('cor2d:%d,%d'%(i-8,j-8))
c1 = my_crosscor2d(ia-ia.mean(),ib-ib.mean())
i,j = np.unravel_index(c.argmax(), c.shape)
print('cor2d:%d,%d'%(i-8,j-8))
c2 = correlate2d(ib-ib.mean(),ia-ia.mean())
i,j = np.unravel_index(c.argmax(), c.shape)
print('cor2d:%d,%d'%(i-31,j-31))

x,y,u_dif,v_dif,u_cor,v_cor,u_my,v_my,u_mycross,v_mycross = [],[],[],[],[],[],[],[],[],[]
for k in range(0,a.shape[0],iw):
    for m in range(0,a.shape[1],iw):
        ia = a[k:k+iw,m:m+iw].copy()
        ib = b[k:k+iw,m:m+iw].copy()
        cor_dif = RMS2D_1(ia-ia.mean(),ib-ib.mean())
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
        c_my = RMS2D_2(ia-ia.mean(),ib-ib.mean())
        row,col = np.unravel_index(c_my.argmin(), c_my.shape)
        u_my.append(col -8)
        v_my.append(row- 8)
        c_mycross = my_crosscor2d(ia-ia.mean(),ib-ib.mean())
        row,col = np.unravel_index(c_mycross.argmax(), c_mycross.shape)
        u_mycross.append(col -8)
        v_mycross.append(row- 8)

plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
M_dif = np.sqrt(pow(np.array(u_dif), 2) + pow(np.array(v_dif), 2))
plt.quiver(x,y,u_dif,v_dif,M_dif)
plt.title('rms1')

plt.subplot(2,2,2)
M_cor = np.sqrt(pow(np.array(u_cor), 2) + pow(np.array(v_cor), 2))
plt.quiver(x,y,u_cor,v_cor,M_cor)
plt.title('rms2')

plt.subplot(2,2,3)
M_my = np.sqrt(pow(np.array(u_my), 2) + pow(np.array(v_my), 2))
plt.quiver(x,y,u_my,v_my,M_my)
plt.title('my cross cor')

plt.subplot(2,2,4)
M_mycross = np.sqrt(pow(np.array(u_mycross), 2) + pow(np.array(v_mycross), 2))
plt.quiver(x,y,u_mycross,v_mycross,M_mycross)
plt.title('scipy cross cor')

#the results are abit different lets explore it
q_rms1_rms2= np.array(M_dif)==np.array(M_my)
problem_rms1_rms2=np.where(q_rms1_rms2==False)
q_mycrosscor_corsscor= np.array(M_mycross)==np.array(M_cor)
problem_mycrosscor_corsscor=np.where(q_mycrosscor_corsscor==False)
print('number of different points comparing rms1 to rms2:%d'%np.size(problem_rms1_rms2[0]))
print('number of different points coparing my cross cor to scipys:%d'%np.size(problem_mycrosscor_corsscor[0]))

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