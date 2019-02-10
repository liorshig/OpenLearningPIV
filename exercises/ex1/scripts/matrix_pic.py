# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 20:20:42 2019

@author: lior
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
row = np.arange(4)
col = np.arange(4)
mat = np.array([[1,2,3,4],
             [5,6,7,8],
             [9,10,11,12],
             [13,14,15,16]])
fig, (ax,ax1) = plt.subplots(1, 2)
im = ax.imshow(mat,cmap='jet')
# We want to show all ticks...
ax.set_xticks(col)
ax.set_yticks(row)
# ... and label them with the respective list entries
ax.set_xticklabels(col)
ax.set_yticklabels(row)
# Loop over data dimensions and create text annotations.
for i in range(len(col)):
    for j in range(len(row)):
        text = ax.text(j, i,mat[i, j],
                       ha="center", va="center", color="w")
ax.set_title("original image [0,0]")
fig.tight_layout()
plt.show()

mat=np.roll(mat,(0,1),axis=(0,1))
#fig, ax = plt.subplots()
im = ax1.imshow(mat,cmap='jet')
# We want to show all ticks...
ax1.set_xticks(col)
ax1.set_yticks(row)
# ... and label them with the respective list entries
ax1.set_xticklabels(np.roll(col,1))
ax1.set_yticklabels(np.roll(row,0))
# Loop over data dimensions and create text annotations.
for i in range(len(col)):
    for j in range(len(row)):
        text = ax1.text(j, i,mat[i, j],
                       ha="center", va="center", color="w")
ax1.set_title("shifted image [0,+1]")
fig.tight_layout()
plt.show()
# rms 2:
fig, (ax,ax1) = plt.subplots(1, 2)
im = ax.imshow(mat[:,1::],cmap='jet')
# We want to show all ticks...
ax.set_xticks(col[1::])
ax.set_yticks(row[1::])
# ... and label them with the respective list entries
ax.set_xticklabels(col[1::])
ax.set_yticklabels(row[1::])
# Loop over data dimensions and create text annotations.
for i in range(len(col[1::])):
    for j in range(len(row[1::])):
        text = ax.text(j, i,mat[i,j],
                       ha="center", va="center", color="w")
ax.set_title("template image [0,0]")
fig.tight_layout()
plt.show()

mat=np.roll(mat,(0,1),axis=(0,1))
#fig, ax = plt.subplots()
im = ax1.imshow(mat[:,1::],cmap='jet')
# We want to show all ticks...
ax1.set_xticks(col[::-1])
ax1.set_yticks(row)
# ... and label them with the respective list entries
ax1.set_xticklabels(np.roll(col[1::],0))
ax1.set_yticklabels(np.roll(row,0))
# Loop over data dimensions and create text annotations.
for i in range(len(col[1::])):
    for j in range(len(row)):
        text = ax1.text(j, i,mat[i, j],
                       ha="center", va="center", color="w")
ax1.set_title("shifted image [0,+1]")
fig.tight_layout()
plt.show()