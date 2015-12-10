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

def read_img_large_small(imgpath = '../result_large.JPG'):
    # Read in images, im_orig is a large image, im is a small one.
    im_orig = scipy.misc.imread(imgpath,flatten=True)
    # Use a small image to test (by 12*12)
    im_small = np.array([im_orig[i][::12] for i in range(len(im_orig)) if i%12==0])
    return im_orig, im_small

def histogram(im):
    histogram = [0]*256
    height, width = im.shape
    for i in im:
        for j in i:
            if j>255:
                j=255
            histogram[j] += 1
    histogram = np.array(histogram, dtype=float)/(width*height)
    plt.bar(range(256), histogram)
    plt.xlim([0,256])
    plt.show()
    #return histogram

def hist_np(im_input,nbr_bins=256):
    im = im_input.astype(int)
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    plt.bar(bins[:-1], imhist)
    plt.xlim([0,256])
    plt.show()

def cumhist(im_input):
    im = im_input.astype(int)
    histogram = [0]*256
    height, width = im.shape
    for i in im:
        for j in i:
            if j<0:
                j=0
            if j>255:
                j=255
            histogram[j] += 1
    histogram = np.array(histogram, dtype=float)/(width*height)
    plt.bar(range(256), np.cumsum(histogram))
    plt.xlim([0,256])
    plt.ylim([0,1])
    plt.show()
    
def cumhist_np(im_input,nbr_bins=256):
    im = im_input.astype(int)
    imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
    plt.bar(bins[:-1], imhist.cumsum())
    plt.xlim([0,256])
    plt.ylim([0,1])
    plt.show()
    
def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r

def normalize(img):
    return img/(np.max(img)-np.min(img))*256