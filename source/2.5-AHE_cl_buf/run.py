from helper import *

import pyopencl as cl
import numpy as np
import pylab

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

def run_AHE_cl(img,clfile = 'AHE_cl_buf.cl',windowsize=21,suppressprint=True):
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

        for platform in platforms:
            print 'The devices detected on platform', platform.name, 'are:'
            print '---------------------------'
            for device in platform.get_devices():
                print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
                print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
                print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
                print 'Maximum work group size', device.max_work_group_size
                print '---------------------------'
        print 'This context is associated with ', len(context.devices), 'devices'
        print 'The queue is using the device:', queue.device.name



    program = cl.Program(context, open(clfile).read()).build(options='')
    
    AHE_out = np.zeros_like(img)

    gpu_im = cl.Buffer(context, cl.mem_flags.READ_ONLY, img.size * 4)
    gpu_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, AHE_out.size * 4)

    local_size = (8, 8)  # 64 pixels per work group
    global_size = tuple([round_up(g, l) for g, l in zip(img.shape[::-1], local_size)])


    pad_size = np.int32(windowsize/2)
    # Set up a (N+8 x N+8) local memory buffer.
    # +2 for 1-pixel halo on all sides, 4 bytes for float.
    local_memory = cl.LocalMemory(4 * (local_size[0] + pad_size) * (local_size[1] + pad_size))
    # Each work group will have its own private buffer.
    buf_width = np.int32(local_size[0] + 2*pad_size)
    buf_height = np.int32(local_size[1] + 2*pad_size)
    halo = np.int32(pad_size)

    width = np.int32(img.shape[1])
    height = np.int32(img.shape[0])

    #max_iters = np.int32(1024)

    cl.enqueue_copy(queue, gpu_im, img, is_blocking=False)

    event = program.AHE_buffer(queue, global_size, local_size,
                           gpu_im, gpu_out, local_memory,
                           width, height,
                           buf_width, buf_height, halo,2*pad_size+1)

    cl.enqueue_copy(queue, AHE_out, gpu_out, is_blocking=True)

    seconds = (event.profile.end - event.profile.start) / 1e9

    gshow(AHE_out.astype(int))
    print "------------------------------------------"
    print "AHE serial image with size: ", img.shape
    print "Window size: ", windowsize
    print "Used time: ", seconds
    print "------------------------------------------"
    
    return AHE_out

if __name__ == '__main__':
	print 'Started... 5. AHE, opencl, buffer (no interpolation, thus not good result for large image.)'

	out_orig = run_AHE_cl(im_orig)

	out_small = run_AHE_cl(im_small)
	gshow(out_small)
	plt.show()
	hist_np(out_small)
	cumhist_np(out_small)