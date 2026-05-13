# Implementation Explanation — Homework 1: Adaptive Histogram Equalization

---

## Thread basics — the foundation of everything

Before any function, understand this: when you write `kernel<<<blocks, 256>>>`, CUDA runs the same function on 256 threads simultaneously. Each thread knows only one thing about itself: its number.

```cuda
threadIdx.x   // this thread's number within its block (0 to 255)
blockIdx.x    // which block this thread belongs to (0 to N_IMAGES-1)
blockDim.x    // total threads per block (256 in our case)
```

---

## Function 1: `prefix_sum`

```cuda
__device__ void prefix_sum(int arr[], int arr_size) {
    if (threadIdx.x == 0) {
        for (int i = 1; i < arr_size; i++) {
            arr[i] += arr[i - 1];
        }
    }
    __syncthreads();
}
```

**`__device__`** means this function runs on the GPU and can only be called from other GPU functions.

**`if (threadIdx.x == 0)`** — all 256 threads enter this function, but only thread 0 does the actual work. Why? The loop `arr[i] += arr[i-1]` is sequential — step 2 needs the result of step 1, step 3 needs step 2, etc. If multiple threads tried it in parallel they'd read stale values.

Example with `arr = [3, 1, 0, 2]`:
```
i=1: arr[1] = 1 + 3 = 4        arr is now [3, 4, 0, 2]
i=2: arr[2] = 0 + 4 = 4        arr is now [3, 4, 4, 2]
i=3: arr[3] = 2 + 4 = 6        arr is now [3, 4, 4, 6]
```
Meaning: 3 pixels ≤ 0, 4 pixels ≤ 1, 4 pixels ≤ 2, 6 pixels ≤ 3.

**`__syncthreads()`** — after thread 0 finishes, ALL 256 threads must arrive at this line before any of them can move past it. Without this, thread 5 might read `arr[5]` before thread 0 has updated it.

```
Thread  0: runs the loop... done → hits __syncthreads() → waits
Thread  1: skips the if      → hits __syncthreads() → waits
Thread  2: skips the if      → hits __syncthreads() → waits
...
Thread 255: skips the if     → hits __syncthreads() → waits
All 256 arrived → barrier releases → all continue
```

---

## Function 2: `process_image_kernel`

```cuda
__global__ void process_image_kernel(uchar *all_in, uchar *all_out, uchar *maps) {
```

**`__global__`** means this is a kernel — launched from the CPU, runs on the GPU.

---

### Part A — Which image does this block work on?

```cuda
    const int image_idx = blockIdx.x;
    uchar *img_in   = all_in  + (size_t)image_idx * IMG_WIDTH * IMG_HEIGHT;
    uchar *img_out  = all_out + (size_t)image_idx * IMG_WIDTH * IMG_HEIGHT;
    uchar *img_maps = maps    + (size_t)image_idx * TILE_COUNT * TILE_COUNT * 256;
```

`blockIdx.x` is the block's number. All 256 threads in the block share the same `blockIdx.x`.

- For the serial case (`kernel<<<1, 256>>>`): only block 0 exists → `image_idx = 0` → processes image 0.
- For the bulk case (`kernel<<<1000, 256>>>`): block 7 → `image_idx = 7` → its pointers jump to image 7's region in memory.

```
all_in memory layout:
[  image 0 (262144 bytes)  |  image 1 (262144 bytes)  |  image 2 ...  ]
 ↑                          ↑
 all_in + 0*262144          all_in + 1*262144
```

The `(size_t)` cast prevents integer overflow when multiplying large numbers.

---

### Part B — Shared histogram

```cuda
    __shared__ int hist[256];
```

**`__shared__`** means this array lives in fast shared memory — physically on the SM chip, not in the slow global GPU memory. All 256 threads in the block share the same `hist` array. Blocks on different SMs each have their own private copy.

Speed comparison:
```
Global memory (VRAM):  ~500 GB/s,  ~200 clock cycles latency
Shared memory:        ~10 TB/s,   ~4 clock cycles latency
```

Using shared memory for the histogram means all the `atomicAdd` operations happen at high speed.

---

### Part C — The tile loop

```cuda
    for (int tile_row = 0; tile_row < TILE_COUNT; tile_row++) {
        for (int tile_col = 0; tile_col < TILE_COUNT; tile_col++) {
```

`TILE_COUNT = 8` (512 / 64). So this loops 8×8 = 64 times. All 256 threads execute this loop together — they're always on the same iteration at the same time (enforced by `__syncthreads()` calls inside).

---

### Part D — Step 1: Clear histogram

```cuda
            for (int i = threadIdx.x; i < 256; i += blockDim.x) {
                hist[i] = 0;
            }
            __syncthreads();
```

`i` starts at `threadIdx.x` and jumps by `blockDim.x` (256). Since there are 256 threads and 256 slots, each thread handles exactly **one slot**:

```
Thread   0: i=0   → hist[0] = 0
Thread   1: i=1   → hist[1] = 0
Thread   2: i=2   → hist[2] = 0
...
Thread 255: i=255 → hist[255] = 0
```

All 256 slots cleared in a **single parallel step**. The `__syncthreads()` ensures every slot is zero before any thread starts adding to the histogram.

---

### Part E — Step 2: Build histogram

```cuda
            const int tile_start_row = tile_row * TILE_WIDTH;
            const int tile_start_col = tile_col * TILE_WIDTH;
            const int tile_pixels    = TILE_WIDTH * TILE_WIDTH;  // 64*64 = 4096

            for (int i = threadIdx.x; i < tile_pixels; i += blockDim.x) {
                int row = tile_start_row + i / TILE_WIDTH;
                int col = tile_start_col + i % TILE_WIDTH;
                uchar pixel = img_in[row * IMG_WIDTH + col];
                atomicAdd(&hist[pixel], 1);
            }
            __syncthreads();
```

`tile_pixels = 4096`, `blockDim.x = 256` → each thread handles `4096/256 = 16` pixels.

`i / TILE_WIDTH` and `i % TILE_WIDTH` convert a flat index into a (row, col) inside the tile.

For tile (0,0), thread 3 works on pixels at indices 3, 259, 515, ..., 3843:
```
i=3:    row=0+3/64=0,   col=0+3%64=3   → pixel at (row 0, col 3)
i=259:  row=0+259/64=4, col=0+259%64=3 → pixel at (row 4, col 3)
i=515:  row=0+515/64=8, col=0+515%64=3 → pixel at (row 8, col 3)
...
```
Every thread always reads **column 3** of the tile. Consecutive threads (0,1,2,3) read consecutive columns (0,1,2,3) of the same row → these are consecutive memory addresses → one memory transaction for the whole group. This is **coalescing**.

**`atomicAdd(&hist[pixel], 1)`** — two threads may have pixels with the same value. Without atomic:
```
Thread 0 and Thread 5 both have pixel value 128:
  Both read: hist[128] = 0
  Thread 0 writes: hist[128] = 1
  Thread 5 writes: hist[128] = 1   ← WRONG, should be 2
```
With `atomicAdd`, the hardware serializes conflicting accesses:
```
  Thread 0: reads 0, adds 1, writes 1
  Thread 5: reads 1, adds 1, writes 2   ← correct
```

---

### Part F — Step 3: CDF

```cuda
            prefix_sum(hist, 256);
            /* hist[v] is now CDF[v] */
```

This calls the function from above. After it returns, `hist[v]` = number of pixels with value ≤ v. For a tile of 4096 pixels where 2000 are dark (≤127) and 2096 are bright (≥128):

```
hist[127] = 2000   "2000 pixels have value ≤ 127"
hist[255] = 4096   "all 4096 pixels have value ≤ 255" (always true)
```

---

### Part G — Step 4: Compute the map

```cuda
            uchar *map = img_maps + (tile_row * TILE_COUNT + tile_col) * 256;
            for (int i = threadIdx.x; i < 256; i += blockDim.x) {
                map[i] = (uchar)((float)hist[i] / tile_pixels * 255.0f);
            }
            __syncthreads();
```

`img_maps + (tile_row * TILE_COUNT + tile_col) * 256` computes where this tile's 256-entry map starts in global memory. For tile (2,3): `(2*8 + 3)*256 = 19*256 = 4864` bytes offset.

Each thread writes its own map entries (same stride pattern as the clear step):
```
Thread   0: map[0]   = hist[0]/4096 * 255
Thread   1: map[1]   = hist[1]/4096 * 255
...
Thread 127: map[127] = 2000/4096 * 255 = 124
Thread 255: map[255] = 4096/4096 * 255 = 255
```

This is the equalization formula: `m[v] = floor(CDF[v] / T^2 * 255)`. A value that was at the 50th percentile (2000/4096 ≈ 48.8%) gets remapped to 48.8% of 255 ≈ 124. Values that were clustered together spread out across the full 0–255 range.

---

### Part H — Step 5: Interpolation

```cuda
    interpolate_device(img_maps, img_in, img_out);
```

Called once, after all 64 tiles are processed. For each output pixel it finds the 4 nearest tile centers, reads their maps, and blends:

```
v' = (1-α)(1-β) * map_TL[v]
   +    α (1-β) * map_TR[v]
   + (1-α)   β  * map_BL[v]
   +    α    β  * map_BR[v]
```

A pixel right at the center of tile (3,3) → α=0, β=0 → uses 100% of tile (3,3)'s map.
A pixel halfway between tile (3,3) and (3,4) → α=0.5 → 50/50 blend of both maps.
This avoids visible tile boundaries in the output image.

---

## Function 3: Task Serial (one image at a time)

### `task_serial_context` struct

```cuda
struct task_serial_context {
    uchar *d_in;   // device buffer for one input  image (IMG_WIDTH * IMG_HEIGHT)
    uchar *d_out;  // device buffer for one output image (IMG_WIDTH * IMG_HEIGHT)
    uchar *d_maps; // device buffer for maps: TILE_COUNT * TILE_COUNT * 256
};
```

Just a container for 3 GPU memory pointers. The `d_` prefix is a convention meaning "device" (GPU) pointer, to distinguish from CPU pointers.

### `task_serial_init`

```cuda
CUDA_CHECK(cudaMalloc(&context->d_in,   IMG_WIDTH * IMG_HEIGHT));      // 262 KB
CUDA_CHECK(cudaMalloc(&context->d_out,  IMG_WIDTH * IMG_HEIGHT));      // 262 KB
CUDA_CHECK(cudaMalloc(&context->d_maps, TILE_COUNT * TILE_COUNT * 256)); // 16 KB
```

`cudaMalloc` is like `malloc` but allocates on the GPU. Allocated once and reused for all 1000 images.

### `task_serial_process`

```cuda
for (int i = 0; i < N_IMAGES; i++) {
    uchar *img_in  = images_in  + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
    uchar *img_out = images_out + (size_t)i * IMG_WIDTH * IMG_HEIGHT;

    // 1. Copy input image from host to device
    CUDA_CHECK(cudaMemcpy(context->d_in, img_in, IMG_WIDTH * IMG_HEIGHT,
                          cudaMemcpyHostToDevice));

    // 2. Process the image on the GPU (single block)
    process_image_kernel<<<1, THREADS_PER_BLOCK>>>(
        context->d_in, context->d_out, context->d_maps);
    CUDA_CHECK(cudaGetLastError());

    // 3. Copy result back to host
    CUDA_CHECK(cudaMemcpy(img_out, context->d_out, IMG_WIDTH * IMG_HEIGHT,
                          cudaMemcpyDeviceToHost));
}
```

Each `cudaMemcpy` stalls the CPU until the transfer finishes. The `cudaMemcpy DeviceToHost` also implicitly waits for the kernel to finish. So the timeline is strictly:

```
Time ──────────────────────────────────────────────────────────────────────►
     [copy→GPU][kernel][copy←GPU][copy→GPU][kernel][copy←GPU]...  (×1000)
                  ↑ GPU only busy here, idle during all copies
```

### `task_serial_free`

```cuda
CUDA_CHECK(cudaFree(context->d_in));
CUDA_CHECK(cudaFree(context->d_out));
CUDA_CHECK(cudaFree(context->d_maps));
free(context);
```

Must mirror every `cudaMalloc` with `cudaFree`, otherwise GPU memory leaks.

---

## Function 4: GPU Bulk (all images at once)

### `gpu_bulk_context` struct

```cuda
struct gpu_bulk_context {
    uchar *d_in;   // N_IMAGES * IMG_WIDTH * IMG_HEIGHT  (~262 MB)
    uchar *d_out;  // N_IMAGES * IMG_WIDTH * IMG_HEIGHT  (~262 MB)
    uchar *d_maps; // N_IMAGES * TILE_COUNT * TILE_COUNT * 256  (~16 MB)
};
```

Same idea as serial but buffers hold ALL 1000 images at once.

### `gpu_bulk_init`

```cuda
const size_t img_total  = (size_t)N_IMAGES * IMG_WIDTH * IMG_HEIGHT;
const size_t maps_total = (size_t)N_IMAGES * TILE_COUNT * TILE_COUNT * 256;

CUDA_CHECK(cudaMalloc(&context->d_in,   img_total));
CUDA_CHECK(cudaMalloc(&context->d_out,  img_total));
CUDA_CHECK(cudaMalloc(&context->d_maps, maps_total));
```

### `gpu_bulk_process`

```cuda
const size_t img_total = (size_t)N_IMAGES * IMG_WIDTH * IMG_HEIGHT;

// 1. Copy ALL input images to the device in one transfer
CUDA_CHECK(cudaMemcpy(context->d_in, images_in, img_total, cudaMemcpyHostToDevice));

// 2. One kernel launch: each block handles one image
process_image_kernel<<<N_IMAGES, THREADS_PER_BLOCK>>>(
    context->d_in, context->d_out, context->d_maps);
CUDA_CHECK(cudaGetLastError());

// 3. Copy ALL output images back in one transfer
CUDA_CHECK(cudaMemcpy(images_out, context->d_out, img_total, cudaMemcpyDeviceToHost));
```

`kernel<<<N_IMAGES, THREADS_PER_BLOCK>>>` = `kernel<<<1000, 256>>>`.

The GPU scheduler sees 1000 independent blocks and fills every SM simultaneously. If the GPU has 40 SMs, 40 images are processed in parallel, then the next 40, etc.

```
Time ──────────────────────────────────────────────────►
     [──── copy to GPU (262 MB) ────][── kernel ──][── copy back ──]
                                           ↑ all 1000 images in parallel
```

Transfer and launch overhead happens **once** instead of 1000 times.

---

## Serial vs Bulk — side-by-side

| | Task Serial | GPU Bulk |
|---|---|---|
| GPU buffers | 1 image (~540 KB) | 1000 images (~540 MB) |
| Kernel launches | 1000 | 1 |
| `cudaMemcpy` calls | 2000 | 2 |
| GPU utilization | ~1 block / all SMs | all SMs fully loaded |
| Expected speedup | baseline | ~10–50× faster |

---

## `THREADS_PER_BLOCK` choice

```cuda
#define THREADS_PER_BLOCK (TILE_WIDTH * 4)   // 64 * 4 = 256
```

- Must be a **multiple of `TILE_WIDTH`** (assignment requirement).
- 256 matches the histogram size exactly → one thread per histogram slot in the clear and map-write steps (no wasted iterations).
- 256 / 4096 pixels per tile = 16 pixels per thread in the histogram step (good balance).
- 256 is a multiple of the GPU warp size (32) → no wasted lanes.

---

## `CUDA_CHECK` macro

```cuda
#define CUDA_CHECK(f) do {                                                         \
    cudaError_t e = f;                                                             \
    if (e != cudaSuccess) {                                                        \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,                  \
               cudaGetErrorString(e));                                             \
        exit(1);                                                                   \
    }                                                                              \
} while (0)
```

Wraps every CUDA API call. If anything fails (out of memory, invalid launch parameters, etc.) it prints the file name, line number, and error string, then exits. Without this, CUDA errors are silently ignored and the program produces wrong results or crashes later with no useful message.
