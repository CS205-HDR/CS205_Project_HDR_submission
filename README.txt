# CS205_Project_HDR_submission

Team: Xinyan Han, Haosu Tang, Qing Zhao

website: http://cs205-hdr.weebly.com/
video: https://www.youtube.com/watch?v=mfCvMTj94Pg

## Running Instructions
In the **"source"** folder, all **.py** files except helper.py are runnable. To note, the image/plots may come behind the terminal.

## Some details
1. required packages (non-standard) needed before execution:
    1) PIL (for part 1)
    2) pyopencl (for parts 1,2,3)
    3) pylab (for part 3)

2. Part 0: HDR Serial Implementation 
	(Reference: https://sites.google.com/site/bpowah/hdrandpythonpil)

    Program Folder/Files: 0-HDR-serial/HDR_serial.py

    For default testing images, you can run the program directly. If you would like to try different images set, replace the 'case'
    and 'cur_dir' values in line: "proj = hdr(case='3_lake', resize=1.0, img_type='jpg', cur_dir=r'../testing_images')" under main function, where 'case' is the folder name where the images are stored and 'cur_dir' is the file path of 'case'.

3. Part 1: HDR Image Processing
    Program Folders(5 steps of optimization): 1.1 xxx through 1.5 xxx
    Program Files:
        HDR.py: python driver for opencl
        hdr.cl: opencl implementation for parallel HDR image processing

    For default testing images, you can run the program directly. All the testing images are stored under '/testing_images' directory.
    If you want to test other images instead of the default 'lake', be sure to change all the file path in HDR.py file.

    Note: the execution may take several minutes because of loading large images.
    
4. Part 2: Local Contrast Enhancement
    2.1 through 2.10 folders contain executable run.py for testing different conditions.

    The plots may come out behind the terminal..

5. Part 3: Convolution Mask
    3.1 through 3.3 .py files are all runnable.

## Result Summary
For much more details, refer to our website: http://cs205-hdr.weebly.com/

Part 1:
We see OpenCL parallelism increases speed by over 40 times. Reading into buffer slightly increased the time but moving into constant memory decreased time by 5%.
![](/imgs/part1_result.png)

Part 2:
To enhance the local contrast, we used histogram equalization. We started from simple global HE, then speeded it up by calling numpy. Going from global to local (to have artistic effect), we first applied LHE, but the outcome image quality is not satisfactory. Therefore we moved on to AHE, pixel by pixel, but it runs too slow on a large image in serial. We ran it on opencl with buffer ¨C the speed increased a lot. We further optimized the algorithm using interpolation (more than 10,000 times than serial), and found this to be the fastest. In an effort to make it faster, we added buffer and found it to be impeding the speed. Further changing the data structure and avoiding bank conflict and grabbing simultaneously to optimize the buffer, the speed has increased a lot comparing to the one with buffer.
![](/imgs/part2_result1.png)
![](/imgs/part2_result2.jpg)

Part 3:
This chart shows that the execution time of convolution adding  buffer for image is roughly the same with convolution per pixel, because accessing mask is still globally and cost running time. However, convolution adding buffer for both  image and mask improves speed by 3 more times because both image pixel values and mask values can be accessed from local buffer. 
![](/imgs/part3_result.png)