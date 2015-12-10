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

def AHE_serial(im_input, windowwidth=21):
    im = im_input.astype(int).copy()
    height, width = im.shape
    new = np.zeros([height,width])
    d = windowwidth/2
    
    for h in range(height):
        for w in range(width):
            cur = im[h,w]
            window = im[h-d if h-d>-1 else 0:h+d+1 if h+d+1<height else height,
                        w-d if w-d>-1 else 0:w+d+1 if w+d+1<width else width]
            flat = [i for sub in window for i in sub]
            flat = flat + random.sample(flat+flat+flat+flat, windowwidth*windowwidth-len(flat))
            idx = np.sort(np.array(flat)).tolist().index(cur)
            
            new[h,w] = int(idx*1.0/windowwidth/windowwidth*256)
    return new

if __name__ == '__main__':
	print 'Started... 4. AHE serial'
	print 'On original image: tested, slow (>1 min)'

	start_time = timeit.default_timer()
	out_small = AHE_serial(im_small)
	elapsed = timeit.default_timer() - start_time
	print 'On small image:', elapsed, 'seconds'
	gshow(out_small)
	plt.show()
	hist_np(out_small)
	cumhist_np(out_small)