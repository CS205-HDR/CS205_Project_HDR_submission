from helper import *

import pyopencl as cl
import numpy as np

import scipy
from scipy import ndimage
import random
import os
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import timeit

show = lambda img: plt.imshow(img.astype(int))
gshow = lambda img: plt.imshow(img.astype(int), cmap = plt.get_cmap('gray'))
rgb2gray = lambda rgb: np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

im_orig, im_small = read_img_large_small('../testing_images/result_large.jpg')

def run_AHE_Interpolation_buffer_cl(img,clfile = 'AHE_interp_cl_buf.cl',segmentationsize=12,suppressprint=True):
    '''
    1). Setting up OpenCL environment with the specified cl file.
    '''
    #################################
    # Setting up environment
    #################################
    # List our platforms
    platforms = cl.get_platforms()


    # Create a context with all the devices
    devices = platforms[0].get_devices()
    context = cl.Context(devices)


    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    
    if(not suppressprint):
        print 'The platforms detected are:'
        print '---------------------------'
        for platform in platforms:
            print platform.name, platform.vendor, 'version:', platform.version

        print 'This context is associated with ', len(context.devices), 'devices'
        # List devices in each platform
        for platform in platforms:
            print 'The devices detected on platform', platform.name, 'are:'
            print '---------------------------'
            for device in platform.get_devices():
                print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
                print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
                print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
                print 'Maximum work group size', device.max_work_group_size
                print '---------------------------'
        print 'The queue is using the device:', queue.device.name

    program = cl.Program(context, open(clfile).read()).build(options='')
    
    width = np.int32(img.shape[1])
    height = np.int32(img.shape[0])
    
    
    n_seg = width*height/segmentationsize/segmentationsize
    seg_width = np.int32(width/segmentationsize)
    seg_height = np.int32(height/segmentationsize)
    # every segmentation requires 256 bins
    #hist_mapping = np.array([0]*(n_seg*256))
    hist_mapping = np.zeros([segmentationsize,256*segmentationsize]).astype(np.int32)
    

    gpu_im = cl.Buffer(context, cl.mem_flags.READ_ONLY, img.size * 4)
    # buffer histogram storage
    gpu_hist = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, hist_mapping.size * 4)
    

    local_size = (16, 32)  # 64 pixels per work group
    global_size = tuple([round_up(g, l) for g, l in zip(img.shape[::-1], local_size)])
    
    
    
    cl.enqueue_copy(queue, gpu_im, img, is_blocking=False)
    
    
    event1 = program.AHE_interpolation_hist(queue, global_size, local_size,
                                           gpu_im, gpu_hist,
                                           width, height, seg_width,seg_height)
    
    cl.enqueue_copy(queue, hist_mapping, gpu_hist, is_blocking=True)
    
    time1 = (event1.profile.end - event1.profile.start) / 1e9

    img_out = np.zeros_like(img, dtype=np.float32)
    gpu_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, img_out.size * 4)
    print hist_mapping.shape
    hist_norm = (hist_mapping.astype(np.float32)/(im_orig.size/144)).reshape([segmentationsize, segmentationsize, 256])
    cumsumhist = np.array([[i.cumsum()*256 for i in sub] for sub in hist_norm ], dtype=np.int32)
    
    gpu_cumhist = cl.Buffer(context, cl.mem_flags.READ_ONLY, cumsumhist.size * 4)
    
    local_memory = cl.LocalMemory(256*4*4+2*4) # store the mapping histogram into the buffer
    
    
    cl.enqueue_copy(queue, gpu_cumhist, cumsumhist, is_blocking=False)
    event2 = program.AHE_interpolation_transform_buffer(queue, global_size, local_size,
                                           gpu_im, gpu_cumhist, gpu_out, local_memory,
                                           width, height,seg_width,seg_height)
    
    cl.enqueue_copy(queue, img_out, gpu_out, is_blocking=True)
    
    time2 = (event2.profile.end - event2.profile.start) / 1e9
    
    gshow(normalize(img_out).astype(int))
    print "------------------------------------------"
    print "AHE serial on image with size: ", img.shape
    print "Segmentation size (on each side): ", segmentationsize
    print "Used time: 1. histogram calculation: ", time1 *1000, ' ms'
    print "Used time: 2. transformation: ", time2 *1000, ' ms'
    print "Used time: 3. total: ", time1*1000+time2 *1000, ' ms'
    print "------------------------------------------"
    
    return img_out

if __name__ == '__main__':
    print 'Started... 7. AHE interpolation, opencl, buffer'

    out_orig = run_AHE_Interpolation_buffer_cl(im_orig)
    gshow(out_orig)
    plt.show()
    hist_np(out_orig)
    cumhist_np(out_orig)

    out_small = run_AHE_Interpolation_buffer_cl(im_small)