# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 20:41:35 2020

@author: Fuso

Beatriz Campos de Almeida de Castro Monteiro
NUSP: 9778619
Bacharel em Ciência da Computação - ICMC, USP São Carlos
SCC0251/5830— Prof. Moacir Ponti
2020.01

github link: https://github.com/biamont/PDI/tree/master/Image%20Enhancement%20and%20Filtering

Short Assignment 2 :  Image Restoration

"""

import numpy as np
import imageio, math, scipy

#CREATES A BORDER OF n/2 0s around the img n=3 t=3 a = b = som = 1
def padding(f, kernel):
    
    n, m = f.shape
    
    kr, kc = kernel.shape
    
    centerR = int((kr-1)/2) # find middle row of kernel
    centerC = int((kc-1)/2) #find middle collun of kernel

    pad_filter = np.zeros((n+(kr-1),m+(kc-1)), dtype=np.float)

    for i in range(centerR, n+centerR):
        for j in range(centerC,m+centerC):
            pad_filter[i][j] = f[i-centerR][j-centerC]
    
    return pad_filter

def gaussian_filter(k, sigma):
   arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
   x, y = np.meshgrid(arx, arx)
   filt = np.exp( -(1/2)*(np.square(x) + np.square(y))/np.square(sigma) )
   return filt/np.sum(filt)

def convolution_point(f, kernel, x, y):

    n, m = kernel.shape #dimensions of w
    a = int((n-1)/2)
    b = int((m-1)/2)
    
    #copy img region centered at x, y
    region_f = np.zeros((n,m), dtype=np.float)
    region_f = f[(x-a):(x+(a+1)), (y-b):(y+(b+1))]
    
    If = 0.0
    s1, s2 = region_f.shape 

    for i in range(n):
        for j in range(m):
            Ii = region_f[i][j]
            If = If + (Ii * kernel[i, j])   
    
    return If

def unsharp_mask(img, k, kernel, sigma, final_img):
   
    #defining kernel filters
    t1, t2 = img.shape
    
    nr,nc = kernel.shape# gets kernel's shape
    centerR = int((nr-1)/2) # find middle row of kernel
    centerC = int((nc-1)/2) #find middle collun of kernel
    

    #creating padding with +centerR 0's on top and bottom of img and +centerC on right and left of img
    pad_img = padding(img, kernel)
    
    #applying the convolution with the chosed kernel filter
    for i in range(centerR, t1+centerR):
        for j in range(centerC, t2+centerC):
            final_img[i-centerR][j-centerC] = convolution_point(pad_img, kernel, i, j)
    
    #scaling and adding
    min_f_i = np.min(final_img)
    max_f_i = np.max(final_img)
   
    for i in range(t1):
        for j in range(t2):
            final_img[i,j] = ((final_img[i,j]-min_f_i)*255)/(max_f_i - min_f_i)
            
            final_img[i,j] = (final_img[i,j]*sigma) + img[i,j]

    #scaling again
    min_f_i = np.min(final_img)
    max_f_i = np.max(final_img)
	
    for i in range(t1):
        for j in range(t2):
            final_img[i,j] = ((final_img[i,j]-min_f_i)*255)/(max_f_i - min_f_i)
            
    return final_img


#Reads inputs
filename = str(input()).rstrip()#reads Image File
k = int(input()) #reads Gaussian Filter h Size 
sigma = float(input()) #reads standard deviation sigma 
gamma = float(input())
#save = int(input()) #1(save final_img) 0(dont save)

#Sets input image
input_img = imageio.imread(filename)
img = np.array(input_img)
img = img.astype(np.int32) #casting para realizar as funcoes

#creates mold for final image
t1, t2 = img.shape
final_img = np.zeros((t1,t2), dtype=np.float)

#creates gaussian filter
gfilter = gaussian_filter(k, sigma)

# applying fast fourier transform in img for denoising
img_fft = scipy.fftpack.fft2(img)
# applying fast fourier transform in kernel for denoising
gfilter_fft = scipy.fftpack.fft2(gfilter)

final_img = unsharp_mask(img, k, gfilter, sigma, final_img)

min_img = np.min(img)
max_img = np.max(img)

#prints standard deviation of image after restoration
print(np.std(final_img)) 

