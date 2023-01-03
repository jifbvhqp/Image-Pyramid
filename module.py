import cv2
import numpy as np
import matplotlib.pyplot as plt
def FilterMatrix(p,q,sigma):
	center_x = int(p/2)
	center_y = int(q/2)
	s = 2*sigma*sigma
	H = np.zeros((p,q))
	for i in range(p):
		for j in range(q):
			x = (i-center_x)**2
			y = (j-center_y)**2
			H[i,j] = (1.0/(np.pi*s)) * np.exp(-float(x+y)/s)
	return H

def convolution(img,kernel):
	img_arr = np.array(img)
	x,y,z = img.shape
	kernel_h,kernel_w = kernel.shape
	
	cov = np.zeros((x,y,z))
	#global paddind_row
	paddind_row = np.zeros((int((kernel_h-1)/2),y))
	#global paddind_col
	paddind_col = np.zeros(((x+paddind_row.shape[0]*2),int((kernel_w-1)/2)))
	
	for h in range(z):
		temp_img_arr = np.vstack([np.vstack([paddind_row,img_arr[:,:,h]]),paddind_row])
		paddind_img_arr = np.hstack([np.hstack([paddind_col,temp_img_arr]),paddind_col])
		for r in range(x):
			for c in range(y):
				val = (paddind_img_arr[r:r+kernel_h,c:c+kernel_w]*kernel).sum()
				cov[r][c][h] = max(min(val,255),0)
	return cov

def reduce(img):
	m,n,c = img.shape
	res = np.zeros((round(m/2),round(n/2),c))
	for i in range(res.shape[0]):
		for j in range(res.shape[1]):
			res[i,j,:] = img[i*2,j*2,:]
	return res

def extend(img):
	x,y,z = img.shape
	extend_img = np.zeros((2*x,2*y,z))
	for i in range(x):
		for j in range(y):
			extend_img[2*i,2*j,:] = img[i,j,:]
	return extend_img
	
def pyramid(img,sigma,layer):
	Filter = FilterMatrix(5,5,sigma)
	Gaussian = []
	Gaussian.append(img)
	Extend = []
	for i in range(layer-1):
		cov_img = convolution(Gaussian[-1],Filter)
		reduce_img = reduce(cov_img)
		Gaussian.append(reduce_img)
		extend_img = extend(reduce_img)
		cov1_img = convolution(extend_img,4*Filter)
		Extend.append(cov1_img)

	Laplacian = []
	for i in range(layer-1):
		m = min(Gaussian[i].shape[0],Extend[i].shape[0])
		n = min(Gaussian[i].shape[1],Extend[i].shape[1])
		Gaussian[i] = Gaussian[i][0:m,0:n]
		Extend[i] = Extend[i][0:m,0:n]
		Laplacian.append(Gaussian[i]-Extend[i])
	Laplacian.append(Gaussian[-1])
		
	return Gaussian,Laplacian
  
def scaleSpectrum(img):
	res_img = np.zeros(img.shape)
	for i in range(img.shape[-1]):
		f = np.fft.fft2(img[:,:,i])
		fshift = np.fft.fftshift(f)
		magnitude_spectrum = 20*np.log(np.abs(fshift))
		res_img[:,:,i] = magnitude_spectrum
	return res_img

def sleSpe(Gaussian,Laplacian):
	G_scaleSpectrum=[]
	L_scaleSpectrum=[]
	for i in range(len(Gaussian)):
		G_scaleSpectrum.append(scaleSpectrum(Gaussian[i]))
		L_scaleSpectrum.append(scaleSpectrum(Laplacian[i]))
	return G_scaleSpectrum,L_scaleSpectrum
	
def ShowImg(img_arr):
	plt.figure(figsize=(20,75))
	L = len(img_arr)
	for i in range(L):
		img = img_arr[i]
		img = img.astype(int)
		img = np.clip(img,0,255)
		img = img[:,:,::-1]
		plt.subplot(1,L,i+1)
		plt.axis('off')
		plt.imshow(img/255)
	plt.show()