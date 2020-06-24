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
import imageio, math, warnings
import scipy.ndimage, scipy.fftpack

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


def constrained_least_squares(Hu, Gu, Pu, gamma, final_img):
   Hu_abs = np.abs(Hu) 
   Pu_abs = np.abs(Pu)
   
   div = np.power(Hu_abs,2)+(gamma*(np.power(Pu_abs,2)))
   kernel_matrix = np.conjugate(Hu)/div
   
   final_img = np.multiply(kernel_matrix, Gu)
            
   return final_img

#Reads input
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
tranformed_img = np.zeros((t1,t2), dtype=np.float)

#creates gaussian filter and laplacian matrix
gfilter = gaussian_filter(k, sigma)
laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

#sets padding for filter
padding_filter = (img.shape[0]//2)-gfilter.shape[0]//2
gfilter_pad = np.pad(gfilter, (padding_filter, padding_filter-1), "constant",  constant_values=0)

# applying fast fourier transform in img and gaussian filter for denoising 
img_fft = np.fft.ifft2(img)
gfilter_fft = np.fft.ifft2(gfilter_pad)
transformed_img = np.multiply(gfilter_fft, img_fft)

#transforms back to space domain
transformed_img_ifft = np.real(scipy.fftpack.fftshift(scipy.fftpack.ifftn(transformed_img)))

#normalizing
max_img = np.amax(img)
min_t = np.amin(transformed_img)

transformed_img = (transformed_img_ifft-min_t)*max_img/(max_img-min_t)
transformed_img = scipy.fftpack.fftn(transformed_img)

#sets padding for laplacian filter
padding_laplace = (img.shape[0]//2)-laplacian.shape[0]//2
laplacian_pad = np.pad(laplacian, (padding_laplace, padding_laplace-1), "constant",  constant_values=0)
laplacian_pad = np.fft.ifft2(laplacian_pad)

final_img = constrained_least_squares(gfilter_fft, transformed_img, laplacian_pad, gamma, final_img)

#inverse transform and normalize
final_img = np.real(scipy.fftpack.fftshift(scipy.fftpack.ifftn(final_img)))

max_f = np.amax(final_img)
min_f = np.amin(final_img)
final_img = (final_img-min_f)*max_img/(max_f-min_f)

        
#prints standard deviation of image after restoration
print("%.1f" % np.std(final_img)) 
warnings.filterwarnings("ignore")

final_img = final_img.astype(np.uint8) #transforms image bacj to its original format (uint8)          
imageio.imwrite('output_img.png', final_img)

