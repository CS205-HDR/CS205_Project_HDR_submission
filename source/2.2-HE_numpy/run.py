from helper import *

import pyopencl as cl
import numpy as np
import pylab

import scipy
import PIL
import PIL.Image as im
from scipy import ndimage
from PIL import ImageEnhance
import random
import os
import matplotlib.pyplot as plt
import timeit

show = lambda img: plt.imshow(img.astype(int))
gshow = lambda img: plt.imshow(img.astype(int), cmap = plt.get_cmap('gray'))
rgb2gray = lambda rgb: np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

im_orig, im_small = read_img_large_small('../testing_images/result_large.jpg')

def HE_numpy(im_input,nbr_bins=256):
    im = im_input.astype(int)
    #get image histogram
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    cdf = imhist.cumsum() #cumulative distribution function
    cdf = 255 * cdf / cdf[-1] #normalize

    #use linear interpolation of cdf to find new pixel values
    im2 = np.interp(im.flatten(),bins[:-1],cdf)

    return im2.reshape(im.shape)

if __name__ == '__main__':
	print 'Started... 2. HE numpy'

	start_time = timeit.default_timer()
	out_orig = HE_numpy(im_orig)
	elapsed = timeit.default_timer() - start_time
	print 'On original image: ', elapsed, 'seconds'

	gshow(out_orig)
	plt.show()
	hist_np(out_orig)
	cumhist_np(out_orig)

	start_time = timeit.default_timer()
	out_small = HE_numpy(im_small)
	elapsed = timeit.default_timer() - start_time
	print 'On small image:', elapsed, 'seconds'