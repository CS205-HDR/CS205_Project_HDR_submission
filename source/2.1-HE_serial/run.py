from helper import *

import pyopencl as cl
import numpy as np

import scipy
from scipy import ndimage
import random
import os
import matplotlib.pyplot as plt
import timeit

show = lambda img: plt.imshow(img.astype(int))
gshow = lambda img: plt.imshow(img.astype(int), cmap = plt.get_cmap('gray'))
rgb2gray = lambda rgb: np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

im_orig, im_small = read_img_large_small('../testing_images/result_large.jpg')

def HE_serial(im_in):
    im = im_in.astype(int).copy()
    histogram = [0]*256
    height, width = im.shape
    for i in im:
        for j in i:
            histogram[j] += 1
    histogram = np.array(histogram, dtype=float)/(width*height)
    cum_hist = np.cumsum(histogram)
    equal_hist = (cum_hist*256).astype(int)
    mapfunc = dict(zip(range(256), equal_hist))
    new_im = np.zeros_like(im)
    for i in range(height):
        for j in range(width):
            new_im[i,j] = mapfunc[im[i,j]]
    
    return new_im

if __name__ == '__main__':
	print 'Started... 1. HE serial'

	start_time = timeit.default_timer()
	he_serial_orig = HE_serial(im_orig)
	elapsed = timeit.default_timer() - start_time
	print 'On original image: ', elapsed, 'seconds'

	gshow(he_serial_orig)
	plt.show()
	hist_np(he_serial_orig)
	cumhist_np(he_serial_orig)

	start_time = timeit.default_timer()
	he_serial_small = HE_serial(im_small)
	plt.show()
	elapsed = timeit.default_timer() - start_time
	print 'On small image:', elapsed, 'seconds'

	hist_np(he_serial_orig)
	cumhist_np(he_serial_orig)