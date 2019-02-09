# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 00:19:41 2019

@author: lior
"""
import numpy as np
#lets find the displacemant
def RMS2D_1(img, template,maxroll=8):
    """
    #CALCULATES the RMS of the difference between img and template at different shifts of img
    #img-2d array of gray scale value of image at time t 
    #template-2d array of gray scale value of image at time t+dt
    #img and template should be same size
    #maxroll- maximun pixel shift of the img 8 pixels is the defult
    #last row/clumn that goes out of array bounds become the first/last row/column
    #i.e the shifted img stays at the same size
    #the function returns a 2d array C at size maxroll+1Xmaxroll+1 wherers
    #C[maxroll,maxroll]-Rms of the difference withount shifts img[i,j] tem[i,j]
    #C[maxroll+x,maxroll+y]-Rms of the difference with shifts img[i+x,j+y] tem[i,j]"""
    mindist = float('inf')
    idx = (0,0)
    cor=np.zeros((maxroll*2+1,maxroll*2+1))
    for row in range(-maxroll,maxroll+1):
        for col in range(-maxroll,maxroll+1):
            cor[maxroll+row,maxroll+col]=np.sqrt(np.sum(np.square(np.roll(img,(row,col),axis=(0,1)) - template)))                  
    return cor
#lets find the displacemant
def RMS2D_2(img, template, maxroll=8):
    """
    #CALCULATES the RMS of the difference between img and template at different shifts of img
    #img-2d array of gray scale value of image at time t 
    #template-2d array of gray scale value of image at time t+dt
    #img and template should be same size
    #maxroll- maximun pixel shift of the img 8 pixels is the defult
    #last row/clumn that goes out of array bounds is thrown away
    #i.e the shifted img becomes smaller with each shift
    #the function returns a 2d array C at size maxroll+1Xmaxroll+1 wherers
    #C[maxroll,maxroll]-Rms of the difference withount shifts img[i,j] tem[i,j]
    #C[maxroll+x,maxroll+y]-Rms of the difference with shifts img[i+x,j+y] tem[i,j]"""
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
    """
    #CALCULATES the cross corelation  between img and template at different shifts of img
    #img-2d array of gray scale value of image at time t 
    #template-2d array of gray scale value of image at time t+dt
    #img and template should be same size
    #maxroll- maximun pixel shift of the img 8 pixels is the defult
    #last row/clumn that goes out of array bounds is thrown away
    #i.e the shifted img becomes smaller with each shift
    #the function returns a 2d array C at size maxroll+1Xmaxroll+1 wherers
    #C[maxroll,maxroll]-cross corelation  withount shifts img[i,j] X tem[i,j]
    #C[maxroll+x,maxroll+y]-cross corelation with shifts img[i+x,j+y] X tem[i,j] """   
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
