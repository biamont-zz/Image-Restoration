# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 20:41:35 2020

@author: Fuso

Beatriz Campos de Almeida de Castro Monteiro
NUSP: 9778619
Bacharel em Ciência da Computação - ICMC, USP São Carlos
SCC0251/5830— Prof. Moacir Ponti
2020.01

github link: https://github.com/biamont/Image-Restoration

Short Assignment 2 :  Image Restoration

"""

import numpy as np
import imageio, warnings
import scipy.ndimage, scipy.fftpack

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

#DENOISING
def denoising(transformed_img, img, gfilter_pad):
    # applying fast fourier transform in img and gaussian filter for denoising 
    img_fft = np.fft.ifft2(img)
    gfilter_fft = np.fft.ifft2(gfilter_pad)
    transformed_img = np.multiply(gfilter_fft, img_fft)

    #transforms back to space domain
    transformed_img_ifft = np.real(scipy.fftpack.fftshift(scipy.fftpack.ifftn(transformed_img)))

    #normalizing
    max_img = np.max(img)
    min_t = np.min(transformed_img)

    transformed_img = (transformed_img_ifft-min_t)*max_img/(max_img-min_t)
    transformed_img = scipy.fftpack.fftn(transformed_img)
    #end of denoising
    
    return transformed_img

#DEBLURING
def debluring(final_img, max_img, gfilter_pad):
   
    gfilter_fft = np.fft.ifft2(gfilter_pad)# applies fast fourier transform in gaussian filter
    
    #sets padding for laplacian filter -> used in constrained least squares
    padding_laplace = (img.shape[0]//2)-laplacian.shape[0]//2
    laplacian_pad = np.pad(laplacian, (padding_laplace, padding_laplace-1), "constant",  constant_values=0)
    laplacian_pad = np.fft.ifft2(laplacian_pad)
 
    #resets blur using contraied least squares method
    final_img = constrained_least_squares(gfilter_fft, transformed_img, laplacian_pad, gamma, final_img)

    #inverse transform and normalize
    final_img = np.real(scipy.fftpack.fftshift(scipy.fftpack.ifftn(final_img)))

    #normalizing
    max_f = np.max(final_img)
    min_f = np.min(final_img)
    final_img = (final_img-min_f)*max_img/(max_f-min_f)

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
transformed_img = np.zeros((t1,t2), dtype=np.float)

#creates gaussian filter and laplacian matrix
gfilter = gaussian_filter(k, sigma)
laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

#sets padding for filter
'''atention: when running in my computer, i had to set the the padding as:
gfilter_pad = np.pad(gfilter, (padding_filter, padding_filter), "constant",  constant_values=0)
but in run.codes i got an error saying the border was big, so I had to subtract 1'''
    
padding_filter = (img.shape[0]//2)-gfilter.shape[0]//2
gfilter_pad = np.pad(gfilter, (padding_filter, padding_filter-1), "constant",  constant_values=0)

#RESTORATION:
transformed_img = denoising(transformed_img, img, gfilter_pad) #call denoising funtion
final_img = debluring(final_img, np.max(img), gfilter_pad)
      
#prints standard deviation of image after restoration
print("%.1f" % np.std(final_img)) 
warnings.filterwarnings("ignore")

final_img = final_img.astype(np.uint8) #transforms image bacj to its original format (uint8)          
imageio.imwrite('output_img.png', final_img)

