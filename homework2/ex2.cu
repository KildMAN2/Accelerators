#include "ex2.h"
#include <cuda/atomic>
#include <algorithm>
#include <new>

#define HIST_SIZE       256
#define MAPS_PER_IMAGE  (TILE_COUNT * TILE_COUNT * HIST_SIZE)
#define IMG_SIZE        (IMG_WIDTH * IMG_HEIGHT)

/* Parallel inclusive prefix-sum (Kogge-Stone scan).
 * Works for any blockDim.x >= arr_size: extra threads are masked out. */
__device__ void prefix_sum(int arr[], int arr_size) {
    int tid = threadIdx.x;
    for (int stride = 1; stride < arr_size; stride <<= 1) {
        int val = 0;
        if (tid < arr_size && tid >= stride) val = arr[tid - stride];
        __syncthreads();
        if (tid < arr_size) arr[tid] += val;
        __syncthreads();
    }
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

/* Process one image. All threads of the block cooperate.
 * Works for blockDim.x in {256, 512, 1024}. */
__device__
void process_image(uchar *in, uchar *out, uchar* maps) {
    __shared__ int hist[HIST_SIZE];

    for (int tr = 0; tr < TILE_COUNT; ++tr) {
        for (int tc = 0; tc < TILE_COUNT; ++tc) {
            for (int i = threadIdx.x; i < HIST_SIZE; i += blockDim.x)
                hist[i] = 0;
            __syncthreads();

            const int tsr = tr * TILE_WIDTH;
            const int tsc = tc * TILE_WIDTH;
            const int tile_pixels = TILE_WIDTH * TILE_WIDTH;
            for (int i = threadIdx.x; i < tile_pixels; i += blockDim.x) {
                int row = tsr + i / TILE_WIDTH;
                int col = tsc + i % TILE_WIDTH;
                atomicAdd(&hist[in[row * IMG_WIDTH + col]], 1);
            }
            __syncthreads();

            prefix_sum(hist, HIST_SIZE);

            uchar *map = maps + (tr * TILE_COUNT + tc) * HIST_SIZE;
            for (int i = threadIdx.x; i < HIST_SIZE; i += blockDim.x)
                map[i] = (uchar)((hist[i] * 255) / tile_pixels);
            __syncthreads();
        }
    }

    interpolate_device(maps, in, out);
}

__global__
void process_image_kernel(uchar *in, uchar *out, uchar* maps){
    process_image(in, out, maps);
}

/* =====================================================================
 * Part 1 — Streams server
 * ===================================================================== */

class streams_server : public image_processing_server
{
private:
    cudaStream_t streams[STREAM_COUNT];
    int          stream_img_id[STREAM_COUNT];   // -1 if free
    uchar       *d_in [STREAM_COUNT];
    uchar       *d_out[STREAM_COUNT];
    uchar       *d_maps[STREAM_COUNT];
    int          next_check;                    // round-robin hint for dequeue

public:
    streams_server() : next_check(0)
    {
        for (int i = 0; i < STREAM_COUNT; ++i) {
            CUDA_CHECK(cudaStreamCreate(&streams[i]));
            CUDA_CHECK(cudaMalloc(&d_in[i],   IMG_SIZE));
            CUDA_CHECK(cudaMalloc(&d_out[i],  IMG_SIZE));
            CUDA_CHECK(cudaMalloc(&d_maps[i], MAPS_PER_IMAGE));
            stream_img_id[i] = -1;
        }
    }

    ~streams_server() override
    {
        for (int i = 0; i < STREAM_COUNT; ++i) {
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
            CUDA_CHECK(cudaFree(d_in[i]));
            CUDA_CHECK(cudaFree(d_out[i]));
            CUDA_CHECK(cudaFree(d_maps[i]));
        }
    }

    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        for (int i = 0; i < STREAM_COUNT; ++i) {
            if (stream_img_id[i] == -1) {
                stream_img_id[i] = img_id;
                CUDA_CHECK(cudaMemcpyAsync(d_in[i], img_in, IMG_SIZE,
                                           cudaMemcpyHostToDevice, streams[i]));
                process_image_kernel<<<1, 1024, 0, streams[i]>>>(
                    d_in[i], d_out[i], d_maps[i]);
                CUDA_CHECK(cudaMemcpyAsync(img_out, d_out[i], IMG_SIZE,
                                           cudaMemcpyDeviceToHost, streams[i]));
                return true;
            }
        }
        return false;
    }

    bool dequeue(int *img_id) override
    {
        for (int n = 0; n < STREAM_COUNT; ++n) {
            int i = (next_check + n) % STREAM_COUNT;
            if (stream_img_id[i] == -1) continue;
            cudaError_t status = cudaStreamQuery(streams[i]);
            switch (status) {
            case cudaSuccess:
                *img_id = stream_img_id[i];
                stream_img_id[i] = -1;
                next_check = (i + 1) % STREAM_COUNT;
                return true;
            case cudaErrorNotReady:
                break;
            default:
                CUDA_CHECK(status);
            }
        }
        return false;
    }
};

std::unique_ptr<image_processing_server> create_streams_server()
{
    return std::make_unique<streams_server>();
}

/* =====================================================================
 * Part 2 — Producer/consumer queues
 * ===================================================================== */

/* TTAS lock kept in GPU memory. Only GPU threads (one per block) take it. */
struct gpu_lock {
    cuda::atomic<int> flag;

    __device__ void lock() {
        for (;;) {
            while (flag.load(cuda::memory_order_relaxed) != 0) { /* spin */ }
            if (flag.exchange(1, cuda::memory_order_acquire) == 0) return;
        }
    }
    __device__ bool try_lock() {
        if (flag.load(cuda::memory_order_relaxed) != 0) return false;
        return flag.exchange(1, cuda::memory_order_acquire) == 0;
    }
    __device__ void unlock() {
        flag.store(0, cuda::memory_order_release);
    }
};

/* MPMC ring queue stored in pinned host memory.
 * head/tail use system-scope atomics so the CPU and GPU synchronize via
 * release/acquire across PCIe. The slot buffer follows the header in memory. */
template <typename T>
class ring_queue {
public:
    cuda::atomic<int> head;       // next slot to read  (consumer side)
    cuda::atomic<int> tail;       // next slot to write (producer side)
    int               capacity;   // power of 2
    int               mask;       // capacity - 1

    __device__ __host__ T *slots() {
        return reinterpret_cast<T *>(reinterpret_cast<char *>(this)
                                     + sizeof(ring_queue));
    }

    __host__ void init(int cap) {
        new (&head) cuda::atomic<int>(0);
        new (&tail) cuda::atomic<int>(0);
        capacity = cap;
        mask     = cap - 1;
    }

    __device__ __host__ bool is_empty() {
        int h = head.load(cuda::memory_order_acquire);
        int t = tail.load(cuda::memory_order_acquire);
        return h == t;
    }

    __device__ __host__ bool is_full() {
        int t = tail.load(cuda::memory_order_relaxed);
        int h = head.load(cuda::memory_order_acquire);
        return (t - h) >= capacity;
    }

    /* Producer: write slot first, then publish via release store on tail. */
    __device__ __host__ void push(const T &item) {
        int t = tail.load(cuda::memory_order_relaxed);
        slots()[t & mask] = item;
        tail.store(t + 1, cuda::memory_order_release);
    }

    /* Consumer: read slot under acquire-ordered tail, then release head. */
    __device__ __host__ T pop() {
        int h = head.load(cuda::memory_order_relaxed);
        T item = slots()[h & mask];
        head.store(h + 1, cuda::memory_order_release);
        return item;
    }

    static size_t bytes(int cap) {
        return sizeof(ring_queue) + (size_t)cap * sizeof(T);
    }
};

struct request_entry {
    int    img_id;
    uchar *img_in;
    uchar *img_out;
};

struct response_entry {
    int img_id;
};

struct queue_ctx {
    ring_queue<request_entry>  *req_q;
    ring_queue<response_entry> *resp_q;
    gpu_lock                   *req_lock;     // protects GPU-side consumers
    gpu_lock                   *resp_lock;    // protects GPU-side producers
    cuda::atomic<int>          *stop_flag;    // pinned host; system scope
    uchar                      *maps_pool;    // num_blocks * MAPS_PER_IMAGE
};

/* Persistent kernel: each threadblock loops dequeue -> process -> enqueue. */
__global__ void persistent_kernel(queue_ctx ctx)
{
    __shared__ request_entry my_req;
    __shared__ int           sig;        // 1 = stop, 2 = got request

    uchar *my_maps = ctx.maps_pool + (size_t)blockIdx.x * MAPS_PER_IMAGE;

    while (true) {
        if (threadIdx.x == 0) {
            sig = 0;
            for (;;) {
                if (ctx.req_lock->try_lock()) {
                    if (!ctx.req_q->is_empty()) {
                        my_req = ctx.req_q->pop();
                        ctx.req_lock->unlock();
                        sig = 2;
                        break;
                    }
                    ctx.req_lock->unlock();
                }
                if (ctx.stop_flag->load(cuda::memory_order_acquire) != 0
                    && ctx.req_q->is_empty()) {
                    sig = 1;
                    break;
                }
            }
        }
        __syncthreads();

        if (sig == 1) break;

        process_image(my_req.img_in, my_req.img_out, my_maps);
        __syncthreads();

        if (threadIdx.x == 0) {
            for (;;) {
                if (ctx.resp_lock->try_lock()) {
                    if (!ctx.resp_q->is_full()) {
                        response_entry r{my_req.img_id};
                        ctx.resp_q->push(r);
                        ctx.resp_lock->unlock();
                        break;
                    }
                    ctx.resp_lock->unlock();
                }
            }
        }
        __syncthreads();
    }
}

/* Compute how many threadblocks can run concurrently, taking the most
 * restrictive of: threads-per-SM, registers-per-SM, shared-mem-per-SM,
 * and the absolute max blocks-per-SM allowed by the architecture. */
static int compute_threadblocks_count(int threads_per_block)
{
    cudaDeviceProp prop;
    int dev;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    const int regs_per_thread  = 32;                          // -maxrregcount=32
    const int shmem_per_block  = sizeof(int) * HIST_SIZE      // hist[256]
                               + 64                           // my_req + sig
                               + 1024;                        // interpolate_device

    int by_threads = prop.maxThreadsPerMultiProcessor / threads_per_block;
    int by_regs    = prop.regsPerMultiprocessor / (threads_per_block * regs_per_thread);
    int by_shmem   = (int)(prop.sharedMemPerMultiprocessor / shmem_per_block);
    int by_blocks  = prop.maxBlocksPerMultiProcessor;

    int blocks_per_sm = std::min({by_threads, by_regs, by_shmem, by_blocks});
    if (blocks_per_sm < 1) blocks_per_sm = 1;
    return blocks_per_sm * prop.multiProcessorCount;
}

static int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

class queue_server : public image_processing_server
{
private:
    int   num_blocks;
    int   queue_capacity;

    char *req_q_buf;                    // pinned host
    char *resp_q_buf;                   // pinned host
    ring_queue<request_entry>  *req_q;
    ring_queue<response_entry> *resp_q;

    gpu_lock          *d_req_lock;      // GPU memory
    gpu_lock          *d_resp_lock;     // GPU memory
    cuda::atomic<int> *h_stop_flag;     // pinned host (system-scope atomic)

    uchar *d_maps_pool;                 // per-block maps

public:
    queue_server(int threads)
    {
        num_blocks     = compute_threadblocks_count(threads);
        queue_capacity = next_pow2(16 * num_blocks);

        /* Queues live in pinned host memory, shared with the GPU. */
        CUDA_CHECK(cudaMallocHost(&req_q_buf,
            ring_queue<request_entry>::bytes(queue_capacity)));
        CUDA_CHECK(cudaMallocHost(&resp_q_buf,
            ring_queue<response_entry>::bytes(queue_capacity)));
        req_q  = reinterpret_cast<ring_queue<request_entry>  *>(req_q_buf);
        resp_q = reinterpret_cast<ring_queue<response_entry> *>(resp_q_buf);
        req_q->init(queue_capacity);
        resp_q->init(queue_capacity);

        /* Locks live in GPU memory: RMW across PCIe is not atomic. */
        CUDA_CHECK(cudaMalloc(&d_req_lock,  sizeof(gpu_lock)));
        CUDA_CHECK(cudaMalloc(&d_resp_lock, sizeof(gpu_lock)));
        CUDA_CHECK(cudaMemset(d_req_lock,  0, sizeof(gpu_lock)));
        CUDA_CHECK(cudaMemset(d_resp_lock, 0, sizeof(gpu_lock)));

        /* Stop flag in pinned host memory so the CPU can flip it cheaply. */
        CUDA_CHECK(cudaMallocHost(&h_stop_flag, sizeof(cuda::atomic<int>)));
        new (h_stop_flag) cuda::atomic<int>(0);

        /* Per-block scratch maps. */
        CUDA_CHECK(cudaMalloc(&d_maps_pool,
                              (size_t)num_blocks * MAPS_PER_IMAGE));

        queue_ctx ctx;
        ctx.req_q     = req_q;
        ctx.resp_q    = resp_q;
        ctx.req_lock  = d_req_lock;
        ctx.resp_lock = d_resp_lock;
        ctx.stop_flag = h_stop_flag;
        ctx.maps_pool = d_maps_pool;

        persistent_kernel<<<num_blocks, threads>>>(ctx);
        CUDA_CHECK(cudaGetLastError());
    }

    ~queue_server() override
    {
        h_stop_flag->store(1, cuda::memory_order_release);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(d_maps_pool));
        CUDA_CHECK(cudaFree(d_req_lock));
        CUDA_CHECK(cudaFree(d_resp_lock));

        h_stop_flag->~atomic();
        CUDA_CHECK(cudaFreeHost(h_stop_flag));

        req_q->head.~atomic();
        req_q->tail.~atomic();
        resp_q->head.~atomic();
        resp_q->tail.~atomic();
        CUDA_CHECK(cudaFreeHost(req_q_buf));
        CUDA_CHECK(cudaFreeHost(resp_q_buf));
    }

    /* Single CPU producer for req_q -- no CPU-side lock needed. */
    bool enqueue(int img_id, uchar *img_in, uchar *img_out) override
    {
        if (req_q->is_full()) return false;
        request_entry r{img_id, img_in, img_out};
        req_q->push(r);
        return true;
    }

    /* Single CPU consumer for resp_q -- no CPU-side lock needed. */
    bool dequeue(int *img_id) override
    {
        if (resp_q->is_empty()) return false;
        response_entry r = resp_q->pop();
        *img_id = r.img_id;
        return true;
    }
};

std::unique_ptr<image_processing_server> create_queues_server(int threads)
{
    return std::make_unique<queue_server>(threads);
}
