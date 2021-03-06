//This function calculates the mapping histogram in all segments
__kernel void
AHE_interpolation_hist(__global __read_only float *in_values,
           __global __write_only int *histmapping,
           int w, int h,
           int seg_width, int seg_height){

    const int x = get_global_id(0);
    const int y = get_global_id(1);

    // Number of the segmentations in the width and height
    const int w_seg = w/seg_width;
    const int h_seg = h/seg_height;

    if((y<h) && (x<w)){
        float currentpixel = in_values[y*w+x];
        // Atomic add to the histogram
        atomic_inc(&histmapping[256*(int)(x/seg_width)+(int)(y/seg_height)*w_seg*256+(int)(currentpixel)]);
    }
}

__kernel void
AHE_interpolation_transform_buffer_small_256_bank_mod(__global __read_only float *in_values,
           __global __read_only int *cumhist,
           __global __write_only float *out_values,
           __local uchar *buffer,
           int w, int h,
           int seg_width, int seg_height){

    const int x = get_global_id(0);
    const int y = get_global_id(1);

    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    bool within = false;    //if local group is not upon boundary

    //check if it is possible to use buffer:
    if((x-lx+seg_width/2)/seg_width == (x-lx+seg_width/2+get_local_size(0))/seg_width 
        && (y-ly+seg_height/2)/seg_height == (y-ly+seg_height/2+get_local_size(1))/seg_height)
        within=true;

    float currentpixel = in_values[y*w+x];

    int totalworkers = get_local_size(0)*get_local_size(1); // Evenly distribute buffer creation to all workers
    int workeridx = ly * get_local_size(0) + lx;    // Get 1d index of the worker
    int workerload = 1024/totalworkers; //4 histograms, each one has 256 bins

    if(within){
        const int w_seg = w/seg_width;
        const int h_seg = h/seg_height;

        int tile = workeridx * workerload / 256;
        int tilex = 0;
        int tiley = 0;
        if(tile==0){
            tilex = (x+seg_width/2)/seg_width-1;
            tiley = (y+seg_height/2)/seg_height-1;
        }
        if(tile==1){
            tilex = (x+seg_width/2)/seg_width;
            tiley = (y+seg_height/2)/seg_height-1;
        }
        if(tile==2){
            tilex = (x+seg_width/2)/seg_width-1;
            tiley = (y+seg_height/2)/seg_height;
        }
        if(tile==3){
            tilex = (x+seg_width/2)/seg_width;
            tiley = (y+seg_height/2)/seg_height;
        }

        int idx = tiley*256*w_seg + tilex*256;

        for(int i=0;i<workerload;i++){
            buffer[workeridx * workerload + tile + i] = cumhist[(workeridx*workerload)%256+idx+i];
        }
    }




    barrier(CLK_LOCAL_MEM_FENCE);

    // Buffer that has square shape

    if((y<h) && (x<w)){

        if(within){
            // Bilinear interpolation coefficients
            int left = (x+seg_width/2)/seg_width;
            int right = left+1;
            int up = (y+seg_height/2)/seg_height;
            int down = up+1;

            // tile coordinates
            int tile1_x = left-1;
            int tile1_y = up-1;
            // Bilinear interpolation coefficients
            float a = (y-((float)tile1_y+0.5)*seg_height)/seg_height;
            float b = (x-((float)tile1_x+0.5)*seg_width)/seg_width;

            if (a<0)    a=0;   // When going out of the upper bound
            if (a>1)    a=1;   // When going out of the lower bound
            if (b<0)    b=0;   // When going out of the left bound
            if (b>1)    b=1;   // When going out of the right bound

            //Bilinear interpolation
            float map1 = (float)buffer[0*257+(int)currentpixel];
            float map2 = (float)buffer[1*257+(int)currentpixel];
            float map3 = (float)buffer[2*257+(int)currentpixel];
            float map4 = (float)buffer[3*257+(int)currentpixel];
            out_values[y*w+x] = (1-a)*((1-b)*map1
                                        +b*map2)
                                +a*((1-b)*map3
                                        +b*map4);

        }
        else{
            const int w_seg = w/seg_width;
            const int h_seg = h/seg_height;

            int left = (x+seg_width/2)/seg_width;
            int right = left+1;
            int up = (y+seg_height/2)/seg_height;
            int down = up+1;

            float currentpixel = in_values[y*w+x];


            // tile coordinates
            int tile1_x = left-1;
            int tile1_y = up-1;
            int tile2_x = left;
            int tile2_y = up-1;
            int tile3_x = left-1;
            int tile3_y = up;
            int tile4_x = left;
            int tile4_y = up;

            // Mapped values at the four neighboring nodes
            int map1 = cumhist[tile1_y*256*w_seg + tile1_x*256 + (int)currentpixel];
            int map2 = cumhist[tile2_y*256*w_seg + tile2_x*256 + (int)currentpixel];
            int map3 = cumhist[tile3_y*256*w_seg + tile3_x*256 + (int)currentpixel];
            int map4 = cumhist[tile4_y*256*w_seg + tile4_x*256 + (int)currentpixel];

            // Bilinear interpolation coefficients
            float a = (y-((float)tile1_y+0.5)*seg_height)/seg_height;
            float b = (x-((float)tile1_x+0.5)*seg_width)/seg_width;

            if (a<0)    a=0;   // When going out of the upper bound
            if (a>1)    a=1;   // When going out of the lower bound              
            if (b<0)    b=0;   // When going out of the left bound                
            if (b>1)    b=1;   // When going out of the right bound

            // Bilinear interpolation
            out_values[y*w+x] = (1-a)*((1-b)*map1+b*map2)+a*((1-b)*map3+b*map4);
        }
    }
}