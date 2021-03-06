
# improvement of convolution

# This is the mask adding version with halo

from __future__ import division
import pyopencl as cl
import numpy as np
import pylab
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage


def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r


if __name__ == '__main__':
    # List our platforms
    platforms = cl.get_platforms()
    print 'The platforms detected are:'
    print '---------------------------'
    for platform in platforms:
        print platform.name, platform.vendor, 'version:', platform.version

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

    # Create a context with all the devices
    devices = platforms[0].get_devices()
    context = cl.Context(devices)
    print 'This context is associated with ', len(context.devices), 'devices'

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    program = cl.Program(context, open('mask_imp.cl').read()).build(options='')

    im0 = scipy.misc.imread('../testing_images/result_enhanced.jpg', flatten=True)
    him0 = im0.copy()
    him0 = np.array(him0, dtype=np.float32)

    # mask that takes 1/2 for each pixal
    #mask = 0.5*np.ones(shape=(499392, 3)).astype(np.float32)

    #lumR = 0.2125
    # lumG = 0.7154
    # lumB = 0.0721
    # # saturation parameter
    # s = 1.4
    # sr = (1 - s) * lumR
    # sg = (1 - s) * lumG
    # sb = (1 - s) * lumB
    # create mask matrix
    #mask = [[sr+s, sr, sr], [sg, sg+s, sg], [sb, sb, sb+s]].astype(np.float32)


    # Original
    #mask = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).astype(np.float32)
    # sharpen
    #mask = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).astype(np.float32)
    # Box blur
    mask = (1/9) * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.float32)
    # Edge detection
    #mask = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).astype(np.float32)
    # Edge detection2
    #mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).astype(np.float32)
    # Gaussian blur
    #mask = (1/16)*np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]).astype(np.float32)



    #print mask
    #print 'mask shape: ', mask.shape

    out = np.zeros_like(him0).astype(np.float32)

    gpu_0 = cl.Buffer(context, cl.mem_flags.READ_ONLY, him0.size * 4)
    gpu_out = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, out.size * 4)
    # gpu of mask
    gpu_mask = cl.Buffer(context, cl.mem_flags.READ_ONLY, mask.size * 4)

    cl.enqueue_copy(queue, gpu_0, him0, is_blocking=False)
    cl.enqueue_copy(queue, gpu_mask, mask, is_blocking=False)



    local_size = (8, 8)  # 64 pixels per work group
    global_size = tuple([round_up(g, l) for g, l in zip(him0.shape[::-1], local_size)])

    width = np.int32(him0.shape[1]) # 3
    height = np.int32(him0.shape[0]) # 499392
    halo = np.int32(1)

    # Set up a (N+2 x N+2) local memory buffer.
    # +2 for 1-pixel halo on all sides, 4 bytes for float.
    buf_size = (np.int32(local_size[0] + 2 * halo), np.int32(local_size[1] + 2 * halo))
    buf_w = np.int32(local_size[0] + 2)
    buf_h = np.int32(local_size[1] + 2)
    local_memory = cl.LocalMemory(4 * buf_size[0] * buf_size[1])
    # Each work group will have its own private buffer.

    event = program.mask(queue, global_size, local_size,
                               gpu_0, gpu_mask, gpu_out, local_memory,
                                width, height, buf_size[0], buf_size[1], halo)



    cl.enqueue_copy(queue, out, gpu_out, is_blocking=True)

    seconds = (event.profile.end - event.profile.start) / 1e9

    print out

    print '-----------Result---------'
    print 'Seconds: ', seconds
    print '--------------------------'


    pylab.imshow(out, cmap=plt.cm.gray)
    pylab.show()