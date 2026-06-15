#include "ex1.h"

// Number of threads per block 
#define THREADS_PER_BLOCK (TILE_WIDTH * 8)


/* Parallel inclusive prefix-sum (Kogge-Stone scan).
 * Requires blockDim.x >= arr_size. Each thread handles one element.
 * Runs in O(log arr_size) steps instead of O(arr_size) on a single thread. */
_device_ void prefix_sum(int arr[], int arr_size) {
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


/* Process one image per thread block.
 *   blockIdx.x  – selects which image (and corresponding maps) to process
 *   threadIdx.x – co-operates with other threads in the block             */
/* Process one image per thread block.
 *   blockIdx.x  – selects which image (and corresponding maps) to process
 *   threadIdx.x – co-operates with other threads in the block             */
_global_ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) {
    const int image_idx = blockIdx.x;
    uchar *img_in   = all_in  + (size_t)image_idx * IMG_WIDTH * IMG_HEIGHT;
    uchar *img_out  = all_out + (size_t)image_idx * IMG_WIDTH * IMG_HEIGHT;
    uchar *img_maps = maps    + (size_t)image_idx * TILE_COUNT * TILE_COUNT * 256;

    _shared_ int hist[256];

    for (int tile_row = 0; tile_row < TILE_COUNT; tile_row++) {
        for (int tile_col = 0; tile_col < TILE_COUNT; tile_col++) {
            // Clear histogram
            for (int i = threadIdx.x; i < 256; i += blockDim.x) {
                hist[i] = 0;
            }
            __syncthreads();

            // Build histogram
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

            // Convert histogram to CDF (prefix sum)
            prefix_sum(hist, 256);

            // Compute the map m[v] = floor(CDF[v]/T^2 * 255)
            uchar *map = img_maps + (tile_row * TILE_COUNT + tile_col) * 256;
            for (int i = threadIdx.x; i < 256; i += blockDim.x) {
                map[i] = (uchar)((hist[i] * 255) / tile_pixels);
            }
            __syncthreads();
        }
    }

   // Interpolate image
    interpolate_device(img_maps, img_in, img_out);
}

/* Task serial context struct with necessary CPU / GPU pointers to process a single image */
struct task_serial_context {
    uchar *d_in;   //input image (IMG_WIDTH * IMG_HEIGHT)
    uchar *d_out;  //output image (IMG_WIDTH * IMG_HEIGHT)
    uchar *d_maps; //tile maps: TILE_COUNT * TILE_COUNT * 256
};

/* Allocate GPU memory for a single input image and a single output image.
 * 
 * Returns: allocated and initialized task_serial_context. */
struct task_serial_context *task_serial_init()
{
    auto context = new task_serial_context;

    CUDA_CHECK(cudaMalloc(&context->d_in,   IMG_WIDTH * IMG_HEIGHT * sizeof(uchar)));
    CUDA_CHECK(cudaMalloc(&context->d_out,  IMG_WIDTH * IMG_HEIGHT * sizeof(uchar)));
    CUDA_CHECK(cudaMalloc(&context->d_maps, TILE_COUNT * TILE_COUNT * 256 * sizeof(uchar)));

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void task_serial_process(struct task_serial_context *context, uchar *images_in, uchar *images_out)
{
    size_t image_size = IMG_WIDTH * IMG_HEIGHT * sizeof(uchar);

    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img_in  = images_in  + (size_t)i * image_size;
        uchar *img_out = images_out + (size_t)i * image_size;

        //   1. copy the relevant image from images_in to the GPU memory you allocated
        CUDA_CHECK(cudaMemcpy(context->d_in, img_in, image_size,
                              cudaMemcpyHostToDevice));

        //   2. invoke GPU kernel on this image
        process_image_kernel<<<1, THREADS_PER_BLOCK>>>(
            context->d_in, context->d_out, context->d_maps);
        CUDA_CHECK(cudaGetLastError());

        //   3. copy output from GPU memory to relevant location in images_out_gpu_serial
        CUDA_CHECK(cudaMemcpy(img_out, context->d_out, image_size,
                              cudaMemcpyDeviceToHost));
    }
}

/* Release allocated resources for the task-serial implementation. */
void task_serial_free(struct task_serial_context *context)
{
    CUDA_CHECK(cudaFree(context->d_in));
    CUDA_CHECK(cudaFree(context->d_out));
    CUDA_CHECK(cudaFree(context->d_maps));

    delete context;
}

/* Bulk GPU context struct with necessary CPU / GPU pointers to process all the images */
struct gpu_bulk_context {
    uchar *d_in;   //all input  images: N_IMAGES * IMG_WIDTH * IMG_HEIGHT 
    uchar *d_out;  //all output images: N_IMAGES * IMG_WIDTH * IMG_HEIGHT
    uchar *d_maps; //all tile maps:  N_IMAGES * TILE_COUNT * TILE_COUNT * 256
};

/* Allocate GPU memory for all the input images, output images, and maps.
 * 
 * Returns: allocated and initialized gpu_bulk_context. */
struct gpu_bulk_context *gpu_bulk_init()
{
    auto context = new gpu_bulk_context;

    const size_t img_total  = (size_t)N_IMAGES * IMG_WIDTH * IMG_HEIGHT* sizeof(uchar);
    const size_t maps_total = (size_t)N_IMAGES * TILE_COUNT * TILE_COUNT * 256 * sizeof(uchar);

    CUDA_CHECK(cudaMalloc(&context->d_in,   img_total));
    CUDA_CHECK(cudaMalloc(&context->d_out,  img_total));
    CUDA_CHECK(cudaMalloc(&context->d_maps, maps_total));

    return context;
}

/* Process all the images in the given host array and return the output in the
 * provided output host array */
void gpu_bulk_process(struct gpu_bulk_context *context, uchar *images_in, uchar *images_out)
{
    const size_t img_total = (size_t)N_IMAGES * IMG_WIDTH * IMG_HEIGHT* sizeof(uchar);

    // 1. copy all input images from images_in to the GPU memory you allocated
    CUDA_CHECK(cudaMemcpy(context->d_in, images_in, img_total, cudaMemcpyHostToDevice));

    // 2. invoke a kernel with N_IMAGES threadblocks, each working on a different image
    process_image_kernel<<<N_IMAGES, THREADS_PER_BLOCK>>>(
        context->d_in, context->d_out, context->d_maps);
    CUDA_CHECK(cudaGetLastError());

    // 3. copy output images from GPU memory to images_out
    CUDA_CHECK(cudaMemcpy(images_out, context->d_out, img_total, cudaMemcpyDeviceToHost));
}

/* Release allocated resources for the bulk GPU implementation. */
void gpu_bulk_free(struct gpu_bulk_context *context)
{
    CUDA_CHECK(cudaFree(context->d_in));
    CUDA_CHECK(cudaFree(context->d_out));
    CUDA_CHECK(cudaFree(context->d_maps));

    delete context;
}
