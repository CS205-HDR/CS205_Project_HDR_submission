
// This is the per pixel cl for mask



__kernel void
mask_nobuffer(__global __read_only float *gpu_in,
                 __global __read_only float *gpu_mask,
                 __global __write_only float *gpu_out,
                 int w, int h,
                 int mask_w, int mask_h)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if((y < h-1)  && (x < w-1) && (x>0) && (y>0)){
        float out_val = 0;
        for(int mh = 0; mh < mask_h; mh++)
            for(int mw = 0; mw < mask_w; mw++)
                out_val += gpu_mask[mh * mask_w + mw] * gpu_in[(y - mask_h / 2 + mh) * w + x - mask_w / 2 + mw];


        gpu_out[x + w * y] = out_val;

    } else {
        gpu_out[y * w + x] = gpu_in[y * w + x];
    }
}

