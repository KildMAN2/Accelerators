#include "ex1.h"

/* Compute an in-place inclusive prefix sum (scan) of arr[0..arr_size-1].
 * Uses thread 0 for the sequential scan and synchronises all threads after.
 * arr must reside in shared memory. */
__device__ void prefix_sum(int arr[], int arr_size) {
    if (threadIdx.x == 0) {
        for (int i = 1; i < arr_size; i++) {
            arr[i] += arr[i - 1];
        }
    }
    __syncthreads();
}

/**
 * Perform interpolation on a single image
 *
 * @param maps 3D array ([TILES_COUNT][TILES_COUNT][256]) of    
 *             the tiles’ maps, in global memory.
 * @param in_img single input image, in global memory.
 * @param out_img single output buffer, in global memory.
 */
__device__ 
void interpolate_device(uchar* maps ,uchar *in_img, uchar* out_img);

/* Number of threads per block – must be a multiple of TILE_WIDTH.
 * 256 = 4 * TILE_WIDTH gives one thread per histogram entry and good
 * occupancy for the pixel-processing loop. */
#define THREADS_PER_BLOCK (TILE_WIDTH * 4)

/* Process one image per thread block.
 *   blockIdx.x  – selects which image (and corresponding maps) to process
 *   threadIdx.x – co-operates with other threads in the block             */
__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) {
    const int image_idx = blockIdx.x;
    uchar *img_in   = all_in  + (size_t)image_idx * IMG_WIDTH * IMG_HEIGHT;
    uchar *img_out  = all_out + (size_t)image_idx * IMG_WIDTH * IMG_HEIGHT;
    uchar *img_maps = maps    + (size_t)image_idx * TILE_COUNT * TILE_COUNT * 256;

    __shared__ int hist[256];

    for (int tile_row = 0; tile_row < TILE_COUNT; tile_row++) {
        for (int tile_col = 0; tile_col < TILE_COUNT; tile_col++) {
            /* --- Step 1: clear the histogram in shared memory --- */
            for (int i = threadIdx.x; i < 256; i += blockDim.x) {
                hist[i] = 0;
            }
            __syncthreads();

            /* --- Step 2: build histogram with atomicAdd --- */
            const int tile_start_row = tile_row * TILE_WIDTH;
            const int tile_start_col = tile_col * TILE_WIDTH;
            const int tile_pixels    = TILE_WIDTH * TILE_WIDTH;

            for (int i = threadIdx.x; i < tile_pixels; i += blockDim.x) {
                int row = tile_start_row + i / TILE_WIDTH;
                int col = tile_start_col + i % TILE_WIDTH;
                uchar pixel = img_in[row * IMG_WIDTH + col];
                atomicAdd(&hist[pixel], 1);
            }
            __syncthreads();

            /* --- Step 3: convert histogram to CDF (prefix sum) --- */
            prefix_sum(hist, 256);
            /* hist[v] is now CDF[v] */

            /* --- Step 4: compute the map m[v] = floor(CDF[v]/T^2 * 255) --- */
            uchar *map = img_maps + (tile_row * TILE_COUNT + tile_col) * 256;
            for (int i = threadIdx.x; i < 256; i += blockDim.x) {
                map[i] = (uchar)((float)hist[i] / tile_pixels * 255.0f);
            }
            __syncthreads();
        }
    }

    /* --- Step 5: bilinear interpolation (provided implementation) --- */
    interpolate_device(img_maps, img_in, img_out);
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    uchar *d_in;   /* device buffer for one input  image (IMG_WIDTH * IMG_HEIGHT) */
    uchar *d_out;  /* device buffer for one output image (IMG_WIDTH * IMG_HEIGHT) */
    uchar *d_maps; /* device buffer for maps: TILE_COUNT * TILE_COUNT * 256        */
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    CUDA_CHECK(cudaMalloc(&context->d_in,   IMG_WIDTH * IMG_HEIGHT));
    CUDA_CHECK(cudaMalloc(&context->d_out,  IMG_WIDTH * IMG_HEIGHT));
    CUDA_CHECK(cudaMalloc(&context->d_maps, TILE_COUNT * TILE_COUNT * 256));

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img_in  = images_in  + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        uchar *img_out = images_out + (size_t)i * IMG_WIDTH * IMG_HEIGHT;

        /* 1. Copy input image from host to device */
        CUDA_CHECK(cudaMemcpy(context->d_in, img_in, IMG_WIDTH * IMG_HEIGHT,
                              cudaMemcpyHostToDevice));

        /* 2. Process the image on the GPU (single block) */
        process_image_kernel<<<1, THREADS_PER_BLOCK>>>(
            context->d_in, context->d_out, context->d_maps);
        CUDA_CHECK(cudaGetLastError());

        /* 3. Copy result back to host (also synchronises the kernel) */
        CUDA_CHECK(cudaMemcpy(img_out, context->d_out, IMG_WIDTH * IMG_HEIGHT,
                              cudaMemcpyDeviceToHost));
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    CUDA_CHECK(cudaFree(context->d_in));
    CUDA_CHECK(cudaFree(context->d_out));
    CUDA_CHECK(cudaFree(context->d_maps));

    free(context);
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    uchar *d_in;   /* device buffer for all input  images: N_IMAGES * IMG_WIDTH * IMG_HEIGHT */
    uchar *d_out;  /* device buffer for all output images: N_IMAGES * IMG_WIDTH * IMG_HEIGHT */
    uchar *d_maps; /* device buffer for all maps:  N_IMAGES * TILE_COUNT * TILE_COUNT * 256  */
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    const size_t img_total  = (size_t)N_IMAGES * IMG_WIDTH * IMG_HEIGHT;
    const size_t maps_total = (size_t)N_IMAGES * TILE_COUNT * TILE_COUNT * 256;

    CUDA_CHECK(cudaMalloc(&context->d_in,   img_total));
    CUDA_CHECK(cudaMalloc(&context->d_out,  img_total));
    CUDA_CHECK(cudaMalloc(&context->d_maps, maps_total));

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    const size_t img_total = (size_t)N_IMAGES * IMG_WIDTH * IMG_HEIGHT;

    /* Copy all input images to the device in one transfer */
    CUDA_CHECK(cudaMemcpy(context->d_in, images_in, img_total, cudaMemcpyHostToDevice));

    /* One kernel launch: each block handles one image */
    process_image_kernel<<<N_IMAGES, THREADS_PER_BLOCK>>>(
        context->d_in, context->d_out, context->d_maps);
    CUDA_CHECK(cudaGetLastError());

    /* Copy all output images back to the host in one transfer */
    CUDA_CHECK(cudaMemcpy(images_out, context->d_out, img_total, cudaMemcpyDeviceToHost));
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    CUDA_CHECK(cudaFree(context->d_in));
    CUDA_CHECK(cudaFree(context->d_out));
    CUDA_CHECK(cudaFree(context->d_maps));

    free(context);
}
