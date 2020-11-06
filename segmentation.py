# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:35:59 2020

@author: Gene764
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt 

def parti_x(seq):
    part_index=[]
    start=0
    end=0
    for i in range(len(seq)-3):
        if seq[i]==0 and seq[i+1]!=0 :
            start=i
        if seq[i]!=0 and seq[i+1]==0 :
            end=i
        if start!=0 and end!=0:
#            print(start,end)
            part_index.append((start,end))
            start=0
            end=0
    return part_index 

def parti_y(seq):
    part_index=[]
    start=0
    end=0
    for i in range(len(seq)-3):
        if seq[i]==0 and seq[i+1]!=0 and seq[i+2]!=0 :
            start=i
        if seq[i]!=0 and seq[i+1]==0 and seq[i+2]==0 :
            end=i
        if start!=0 and end!=0:
#            print(start,end)
            part_index.append((start,end))
            start=0
            end=0
    return part_index        

img_gray = cv2.imread("E:/project/text_rotation/music.jpg",cv2.IMREAD_GRAYSCALE)

#img = cv2.medianBlur(img_gray, 5)
grayNot = cv2.bitwise_not(img_gray)
threImg = cv2.threshold(grayNot,90,255,cv2.THRESH_BINARY,)[1]
#img_blur=cv2.GaussianBlur(img_gray,(5,5),5)
##自适应阈值分割
#threImg=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
#threImg = cv2.bitwise_not(threImg)
threImg = threImg/255
#threImg = cv2.bitwise_not(threImg)
cv2.imshow("threImg", threImg )
cv2.waitKey()


y=np.sum(threImg[:,:100],axis=1)
y_threshold=(np.max(y)-np.min(y))/3

for i in range(len(y)):
    if y[i] < y_threshold:
        y[i]=0

cA, cD = pywt.dwt(y, 'db2') 
x2 = pywt.idwt(cA, cD, 'db2') 
a=parti_y(y)

c=range(len(y))
plt.scatter(c,y)
plt.show()

for p in a:
    z=np.sum(threImg[p[0]-8:p[1]+8,:],axis=0)
    z_threshold=(np.max(z)-np.min(z))/3

    for i in range(len(z)):
        if z[i] < z_threshold:
            z[i]=0
    az=parti_x(z)        
    x=range(len(z))
#    plt.scatter(x,z)
#    plt.show()

last = a[0]
#for ii in a:
for i in az:
    cv2.imshow("threImg", img_gray[last[1]-25:last[0]+25,i[0]-5:i[1]+5])
    cv2.waitKey()

