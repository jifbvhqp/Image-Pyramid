# -*- coding: utf-8 -*-
import cv2
import numpy as np
from module import FilterMatrix,convolution,reduce,extend,pyramid,scaleSpectrum,sleSpe,ShowImg

picture_name = '6_makeup_before.jpg'
img = cv2.imread('data/'+ picture_name , cv2.IMREAD_COLOR)

sigma = 1
Gaussian,Laplacian = pyramid(img,sigma,5)
G_scaleSpectrum,L_scaleSpectrum = sleSpe(Gaussian,Laplacian)
ShowImg(Gaussian)
ShowImg(Laplacian)
#ShowImg(G_scaleSpectrum)
#ShowImg(L_scaleSpectrum)

