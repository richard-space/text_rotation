# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 10:55:30 2020

@author: Gene764
"""

import numpy as np
import os
import cv2
import math

def rotate(image,angle,center=None,scale=1.0):
    (w,h) = image.shape[0:2]
    if center is None:
        center = (w//2,h//2)   
    wrapMat = cv2.getRotationMatrix2D(center,angle,scale)    
    return cv2.warpAffine(image,wrapMat,(h,w))
#使用矩形框
def getCorrect():
    #读取图片，灰度化
    src = cv2.imread("F:/images.png")
    cv2.imshow("src",src)
    cv2.waitKey()
    gray = cv2.imread("F:/images.png",cv2.IMREAD_GRAYSCALE)
    cv2.imshow("gray",gray)
    cv2.waitKey()
    #图像取非
    grayNot = cv2.bitwise_not(gray)
    cv2.imshow("grayNot",grayNot)
    cv2.waitKey()
    #二值化
    threImg = cv2.threshold(grayNot,100,255,cv2.THRESH_BINARY,)[1]
    cv2.imshow("threImg",threImg)
    cv2.waitKey()
    #获得有文本区域的点集,求点集的最小外接矩形框，并返回旋转角度
    coords = np.column_stack(np.where(threImg>0))
    rect = cv2.minAreaRect(coords)
#    print(rect)
    angle = rect[-1]
    
#    box = cv2.boxPoints(rect) # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
##    
#    box = np.int0(box)
#    # 画出来
#    cv2.drawContours(src, [box], 0, (255, 0, 0), 1)
#    cv2.imwrite('contours.png', src)
#    
    if angle < -45:
        angle = -(angle + 90)
    else:
        angle = -angle

    #仿射变换，将原图校正
#    dst = rotate(src,angle)
#    cv2.imshow("dst",dst)
#    cv2.waitKey()
#    print(angle)
    
    imgRotation = rotate_bound_white_bg(src, angle)
    cv2.imwrite('contours.png', imgRotation)
    cv2.imshow("imgRotation",imgRotation)
    cv2.waitKey()
    return angle
   

def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(image, M, (nW, nH),borderValue=(255,255,255))
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))
    

if __name__ == "__main__":              
    angle=getCorrect()
    