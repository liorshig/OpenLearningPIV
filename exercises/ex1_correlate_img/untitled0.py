# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 02:06:11 2019

@author: lior
"""
mindist = float('inf')
idx = (0,0)
for row in range(6,maxroll+1):
        for col in range(5,maxroll+1):
            #calculate Euclidean distance
            dist = np.sqrt(np.sum(np.square(template - np.roll(img,(row,col),axis=(0,1)))))
            if dist < mindist:
                mindist = dist
                idx = (row,col)