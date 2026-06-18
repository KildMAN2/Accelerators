# Homework 2 — Implementation Notes

This document explains everything that was filled into `ex2.cu`: what each
piece does, the synchronization design, and why the code is structured the
way it is.

The file implements two alternative GPU image-processing **servers**, both
exposing the abstract interface defined in `ex2.h`:

```cpp
class image_processing_server {
    virtual bool enqueue(int img_id, uchar *img_in, uchar *img_out) = 0;
    virtual bool dequeue(int *img_id) = 0;
};
```

* `streams_server` — uses 64 CUDA streams, one in-flight request per stream.
* `queue_server`   — runs a *persistent* CUDA kernel that pulls requests
  off a CPU↔GPU producer/consumer queue and pushes results back.

---

## 0. Requirements → what we implemented (with examples)

This table maps each requirement from the assignment to the exact place in
`ex2.cu` that satisfies it, plus a short example of the resulting behavior.

| # | Requirement (assignment) | What we did | Example |
|---|--------------------------|-------------|---------|
| 1a | Streams server, 1024 threads, **one block per image** | `process_image_kernel<<<1, 1024, 0, stream>>>` launched from `streams_server::enqueue` | `./ex2 streams 0` → `distance from baseline 0` |
| 1 | Use **64 streams**, pick a free one, else return `false` | `cudaStream_t streams[STREAM_COUNT]` (`STREAM_COUNT==64`); `enqueue` scans for `stream_img_id[i]==-1` | when all 64 are busy, `enqueue` returns `false` and the main loop retries |
| 1 | `dequeue` must **not block**, recheck next time | `cudaStreamQuery` + round-robin `next_check`; `cudaErrorNotReady` → skip | the main loop keeps spinning, latency stays ~0.06 ms |
| 2a | Compute concurrent threadblocks from device props, **no hard-coding** | `compute_threadblocks_count()` uses `cudaGetDeviceProperties` | 1024 thr → 1 block/SM, 512 → 2, 256 → 4 on Turing |
| 2 | Queue size = `2^ceil(log2(16·TB))` | `queue_capacity = next_pow2(16 * num_blocks)` | 40 blocks → `next_pow2(640)` = 1024 slots |
| 2b | MPMC queue in pinned host memory | `ring_queue<T>` placed on a `cudaMallocHost` buffer | both CPU and GPU read/write the same slots |
| 2b | **Two TAS/TTAS locks**, GPU-resident | `gpu_lock req_lock, resp_lock` via `cudaMalloc` | RMW kept off PCIe (not atomic across the bus) |
| 2b | Release/Acquire ordering | `cuda::atomic` head/tail with `memory_order_release`/`acquire` | producer's slot write is visible before tail publish |
| 2d-i | Kernel launched in constructor, terminated in destructor | `persistent_kernel<<<num_blocks, threads>>>`; `stop_flag` + `cudaDeviceSynchronize` | destructor returns only after the kernel exits |
| 1d/h | Correctness across **256/512/1024 threads** | every loop uses `for (i=threadIdx.x; i<N; i+=blockDim.x)` | all three thread counts print `distance ... 0` |
| 1c/e | Check CUDA errors, free all resources | `CUDA_CHECK(...)` on every API call; destructors free everything | no leaks, clean exit |

Quick verification:

```text
$ ./ex2 streams 0
throughput = 46934.7 (req/sec)   distance from baseline 0

$ ./ex2 queue 1024 0
throughput =  7769.2 (req/sec)   distance from baseline 0

$ ./ex2 queue 512 0
throughput =  3047.5 (req/sec)   distance from baseline 0

$ ./ex2 queue 256 0
throughput =  1430.6 (req/sec)   distance from baseline 0
```

---

## 0.5 Background concepts (deep-dive for learning)

This section explains the underlying ideas the assignment is built on. If
you already know CUDA you can skip it — it is here so the *why* behind the
code is clear.

### What is a CUDA stream?

A **stream** is an ordered queue of GPU work (kernel launches and memory
copies). The rules are simple but powerful:

* Operations **in the same stream** execute **in order** — operation *N+1*
  starts only after operation *N* has finished.
* Operations **in different streams** have **no ordering** between them, so
  the GPU is free to overlap them (run a kernel in stream A while copying
  memory in stream B).

Think of streams as checkout lanes in a supermarket: each lane serves its
customers one by one, but several lanes serve customers simultaneously.

```text
stream 0:  H2D ──▶ kernel ──▶ D2H        ┐
stream 1:  H2D ──▶ kernel ──▶ D2H        │  these three lanes
stream 2:  H2D ──▶ kernel ──▶ D2H        ┘  overlap in time
```

Without streams (the "default stream"), everything is serialized — that is
basically homework 1's task-serial version. By spreading 64 independent
images over 64 streams, the copy of one image overlaps with the compute of
another, which is why the streams server reaches ~47 K req/s.

`cudaMemcpyAsync` and `kernel<<<grid, block, shmem, stream>>>` are the
*asynchronous* API calls: they **return immediately** to the CPU after
*enqueuing* the work, they do not wait for the GPU. That is what lets
`enqueue` return instantly and the main loop keep feeding requests.

### How do we know when a stream is finished — without blocking?

Two options:

* `cudaStreamSynchronize(s)` — **blocks** the CPU until stream `s` is empty.
  We must **not** use this in `dequeue` (it would stall the whole server).
* `cudaStreamQuery(s)` — **non-blocking** poll. Returns `cudaSuccess` if the
  stream is idle, or `cudaErrorNotReady` if work is still pending.

We poll with `cudaStreamQuery` so the server can do other useful work
(enqueue more images, check other streams) instead of waiting.

### Pinned (page-locked) host memory — `cudaMallocHost`

Normal `malloc` memory is *pageable*: the OS may move it around or swap it
to disk. The GPU's DMA engine cannot safely read pageable memory, so a
normal copy first stages data through a hidden pinned buffer (slower).

`cudaMallocHost` allocates **pinned** memory: its physical address is fixed,
so:

* The GPU's DMA engine can read/write it **directly** over PCIe.
* It can be **mapped into the GPU's address space**, meaning a GPU kernel
  can dereference a host pointer and the access turns into a PCIe
  transaction — exactly how the persistent kernel reads the request queue
  and the image pixels.

This is the foundation of the whole Part-2 design: the queues live in
pinned memory so **both** the CPU and the GPU can touch the same bytes.

### Atomics and memory ordering (`cuda::atomic`)

An **atomic** variable can be read/written by multiple threads (even on
different devices) without tearing. But atomicity alone is not enough — we
also need **ordering**: a guarantee about *which other writes are visible*
when an atomic is observed. The orderings we use:

* `memory_order_relaxed` — atomic, but **no** ordering guarantee. Cheap.
  Used in the TTAS spin where we only need to see *eventually* that the
  lock changed.
* `memory_order_release` (on a store) + `memory_order_acquire` (on the
  matching load) form a **release/acquire pair**. Everything the producer
  wrote *before* the release store becomes visible to the consumer *after*
  it observes the value with the acquire load. This is the "publish" model:

  ```text
  producer:  data = X;                       // ordinary write
             flag.store(1, release);   ─┐    // publish
  consumer:  while(flag.load(acquire)==0);  ◀┘   // observe
             use(data);                      // guaranteed to see X
  ```

`cuda::atomic<T>` defaults to **`thread_scope_system`**, the widest scope,
which means the ordering holds even **between the CPU and the GPU across
PCIe**. That is exactly what `hello-shmem.cu` demonstrates and what we
reuse for the queues.

### Why locks must live in GPU memory

A lock is taken with a **read-modify-write** (RMW) operation
(`exchange`: read the old value and write 1 in one indivisible step). PCIe
**cannot perform an RMW atomically** — it only carries plain reads and
writes. So an atomic RMW only works on memory *local* to the processor doing
it. Since the GPU threadblocks take the locks, the locks are allocated with
`cudaMalloc` (GPU memory). The *data* (queues) can be in host memory because
we only do plain loads/stores on them; the *locks* cannot.

### TAS vs. TTAS lock

* **TAS** (test-and-set): keep calling `exchange(1)` until it returns 0.
  Every attempt is an RMW that invalidates the cache line on all cores —
  heavy bus traffic under contention.
* **TTAS** (test-and-test-and-set): first spin on a cheap `load` until the
  lock *looks* free, and only then try the expensive `exchange`. While the
  lock is held, every waiter just reads its own cached copy — no bus
  traffic until the owner releases. We use TTAS for that reason.

### Persistent kernel ("megakernel")

Normally you launch a kernel, it runs, it exits. A **persistent kernel** is
launched **once** and loops forever (`while(true)`) until told to stop. The
threadblocks stay resident on the SMs and repeatedly pull work from a queue:

```text
while (running) {
    req  = dequeue_request();   // from CPU→GPU queue
    res  = process_image(req);
    enqueue_response(res);      // to GPU→CPU queue
}
```

The advantage is that we pay the kernel-launch cost only once instead of
per image, and the GPU can start a new image the instant a block is free,
with no CPU round-trip. The challenge is the synchronization (locks +
release/acquire) we built above, and a clean shutdown (the `stop_flag`).

### Grid / block / thread refresher

* A **thread** runs the kernel code; `threadIdx.x` is its index in the block.
* A **block** (threadblock) is a group of up to 1024 threads that share
  `__shared__` memory and can `__syncthreads()`. One block processes one
  image here.
* A **grid** is all the blocks of one launch; `blockIdx.x` selects which
  block (in Part 2, which of the persistent workers) you are.
* `__syncthreads()` is a **barrier**: every thread in the block waits until
  all reach it. We use it so the whole block agrees on the shared histogram
  and on the dequeued request before continuing.

### PCIe: posted vs. non-posted (used in questions j/k)

* A **posted** transaction (memory *write*) is fire-and-forget: the sender
  does not wait for a reply. Fast.
* A **non-posted** transaction (memory *read*) requires a *completion* to
  travel back. The sender stalls for the full round-trip latency (µs scale).

This distinction is why moving the CPU→GPU queue into GPU memory (so the CPU
*writes* instead of the GPU *reading*) can be faster — see §4(j).

---

## 1. Per-image GPU algorithm (shared by both servers)

### `prefix_sum(int arr[], int arr_size)`
Kogge–Stone inclusive scan. Works for any `blockDim.x >= arr_size`; threads
with `tid >= arr_size` are masked out on both the read and the write so that
running with 1024 threads over a 256-element histogram is still safe:

```cpp
for (int stride = 1; stride < arr_size; stride <<= 1) {
    int val = 0;
    if (tid < arr_size && tid >= stride) val = arr[tid - stride];
    __syncthreads();
    if (tid < arr_size) arr[tid] += val;
    __syncthreads();
}
```

The two `__syncthreads()` enforce the read-before-write/write-before-read
order across all participating warps.

### `process_image(uchar *in, uchar *out, uchar *maps)`
Same algorithm as in homework 1, but adapted to:

* 128×128 images and `TILE_COUNT = 2` (i.e. 2×2 = 4 tiles).
* **Any** `blockDim.x ∈ {256, 512, 1024}` — every loop uses the
  `for (i = threadIdx.x; i < N; i += blockDim.x)` idiom.

For each of the 4 tiles:
1. Zero the shared histogram `hist[256]`.
2. Atomically increment `hist[pixel]` for each pixel of the tile.
3. Convert the histogram to an inclusive CDF via `prefix_sum`.
4. Write the tile map `m[v] = (CDF[v] * 255) / tile_pixels` into the
   global maps buffer.

Finally call `interpolate_device(maps, in, out)` (provided in
`libutils.a`) to produce the bilinear-interpolated output image.

### `process_image_kernel`
Trivial `<<<...>>>` entry point that just calls `process_image`. Used by the
streams server with `<<<1, 1024, 0, stream>>>`.

---

## 2. Streams server (Part 1)

### State
* `cudaStream_t streams[STREAM_COUNT]` — 64 streams.
* `int stream_img_id[STREAM_COUNT]` — the `img_id` currently in flight in
  each stream, or `-1` if the slot is free.
* `uchar *d_in[i], *d_out[i], *d_maps[i]` — private device buffers per
  stream (so that operations on different streams cannot alias).
* `int next_check` — round-robin hint to keep `dequeue` fair.

### `enqueue`
Find the first slot `i` with `stream_img_id[i] == -1`. If none exists,
return `false` (the main loop will retry). Otherwise enqueue, in that
stream and in order:

1. `cudaMemcpyAsync(d_in[i], img_in, IMG_SIZE, H→D, stream)`
2. `process_image_kernel<<<1, 1024, 0, stream>>>(d_in[i], d_out[i], d_maps[i])`
3. `cudaMemcpyAsync(img_out, d_out[i], IMG_SIZE, D→H, stream)`

Operations on the same stream execute in program order; operations on
different streams may overlap. We never call `cudaStreamSynchronize` here.

#### Why these 3 lines matter (line-by-line)

```cpp
CUDA_CHECK(cudaMemcpyAsync(d_in[i], img_in, IMG_SIZE,
                  cudaMemcpyHostToDevice, streams[i]));
process_image_kernel<<<1, 1024, 0, streams[i]>>>(
   d_in[i], d_out[i], d_maps[i]);
CUDA_CHECK(cudaMemcpyAsync(img_out, d_out[i], IMG_SIZE,
                  cudaMemcpyDeviceToHost, streams[i]));
```

1. `cudaMemcpyAsync(...HostToDevice...)` enqueues an async input transfer
  of one image from CPU pinned memory (`img_in`) into this stream's device
  input buffer (`d_in[i]`).
2. `process_image_kernel<<<..., streams[i]>>>` enqueues the GPU computation
  for that same image in the same stream.
3. `cudaMemcpyAsync(...DeviceToHost...)` enqueues an async output transfer
  from this stream's device output buffer (`d_out[i]`) back to the caller's
  host output (`img_out`).

Because all three operations are enqueued to the same stream, CUDA preserves
their order exactly:

$$
  ext{H2D copy} \rightarrow \text{kernel} \rightarrow \text{D2H copy}
$$

At the same time, different streams can run these pipelines concurrently, so
copy and compute from different requests can overlap.

### `dequeue`
Round-robin over occupied slots starting from `next_check`, calling
`cudaStreamQuery(streams[i])`:

| Result               | Action                                              |
|----------------------|-----------------------------------------------------|
| `cudaSuccess`        | Return that slot's `img_id`, mark slot free.        |
| `cudaErrorNotReady`  | Move on to the next slot.                           |
| anything else        | `CUDA_CHECK` → abort.                               |

`cudaStreamQuery` is non-blocking — we never stall the main loop.

### Destructor
`cudaStreamSynchronize` each stream (defensive — main has already drained
everything), then destroy streams and free buffers.

#### Worked example — a request through the streams server

Suppose `enqueue(42, in, out)` is called and stream 7 is the first free slot:

```text
slot scan: 0..6 busy, slot 7 free  → claim it
stream_img_id[7] = 42
stream 7 ◀ cudaMemcpyAsync(d_in[7] ← in)        (H→D, async)
stream 7 ◀ process_image_kernel<<<1,1024>>>      (compute maps + interpolate)
stream 7 ◀ cudaMemcpyAsync(out ← d_out[7])       (D→H, async)
enqueue returns true
```

Later iterations call `dequeue`:

```text
dequeue: cudaStreamQuery(stream 7) = cudaErrorNotReady  → still running, skip
...
dequeue: cudaStreamQuery(stream 7) = cudaSuccess        → done!
         *img_id = 42; stream_img_id[7] = -1 (slot freed); return true
```

Because all three operations were placed on the *same* stream, they run in
order; the D→H copy cannot start before the kernel finishes. Streams 0..63
run independently, so up to 64 images are processed concurrently.

### Streams measurements (tasks b/c/d)

Measured with:

```bash
./ex2 streams 0
./ex2 streams <load>
```

For cleaner reporting, I used:

* 5 runs of `./ex2 streams 0`, and took the **median throughput** as `maxLoad`.
* 3 repeats per load point in task (c), and reported **median throughput** and
  **median latency**.

`maxLoad` (median of 5 runs at `load=0`) was:

```text
maxLoad = 4239.1 req/sec
```

For task (c), load was varied from `maxLoad/10` to `2*maxLoad` in 10 equal
steps. Collected samples (median across 3 repeats per point):

| k | Load (req/sec) | Throughput (req/sec) | Median latency (ms) |
|---|---------------:|---------------------:|--------------------:|
| 0 |      423.91    |             420.9    |            7.8141   |
| 1 |     1318.83    |            1306.4    |            8.3023   |
| 2 |     2213.75    |            2179.2    |            9.2616   |
| 3 |     3108.67    |            3053.1    |           11.4254   |
| 4 |     4003.59    |            3782.7    |           21.7124   |
| 5 |     4898.52    |            4201.7    |          226.2297   |
| 6 |     5793.44    |            4240.0    |          378.9552   |
| 7 |     6688.36    |            4252.1    |          441.8282   |
| 8 |     7583.28    |            4259.3    |          525.5255   |
| 9 |     8478.20    |            4244.3    |          618.9658   |

Latency-throughput graph (task d), with linear X-axis and marked points:

```mermaid
xychart-beta
    title "Streams: Median Latency vs Throughput"
    x-axis "Throughput (req/sec)" 0 --> 4500
    y-axis "Median latency (ms)" 0 --> 700
    line "samples" [420.9, 1306.4, 2179.2, 3053.1, 3782.7, 4201.7, 4240.0, 4252.1, 4259.3, 4244.3] [7.8141, 8.3023, 9.2616, 11.4254, 21.7124, 226.2297, 378.9552, 441.8282, 525.5255, 618.9658]
    scatter "points" [420.9, 1306.4, 2179.2, 3053.1, 3782.7, 4201.7, 4240.0, 4252.1, 4259.3, 4244.3] [7.8141, 8.3023, 9.2616, 11.4254, 21.7124, 226.2297, 378.9552, 441.8282, 525.5255, 618.9658]
```

What we learn from the graph:

* At low-to-moderate load, throughput grows almost linearly while latency
  stays low (~8-22 ms).
* Around ~3.8k-4.2k req/sec, the curve reaches its knee and throughput
  begins to saturate.
* Beyond the knee, offered load keeps increasing but throughput remains near
  ~4.2k req/sec, while latency rises steeply (hundreds of ms), indicating
  queue buildup in a capacity-limited regime.

---

## 3. Producer–Consumer queue server (Part 2)

### 3.1 How many threadblocks can run concurrently?

Implemented in `compute_threadblocks_count`:

```cpp
int by_threads = prop.maxThreadsPerMultiProcessor / threads_per_block;
int by_regs    = prop.regsPerMultiprocessor / (threads_per_block * 32);
int by_shmem   = prop.sharedMemPerMultiprocessor / shmem_per_block;
int by_blocks  = prop.maxBlocksPerMultiProcessor;
int blocks_per_sm = min({by_threads, by_regs, by_shmem, by_blocks});
return blocks_per_sm * prop.multiProcessorCount;
```

Formula used in words:

$$
blocks_{per\_SM} = \min\left(
\left\lfloor\frac{T_{SM}}{T_{block}}\right\rfloor,
\left\lfloor\frac{R_{SM}}{R_{thread}\cdot T_{block}}\right\rfloor,
\left\lfloor\frac{S_{SM}}{S_{block}}\right\rfloor,
B_{SM}^{max}
\right)
$$

$$
total_{blocks} = blocks_{per\_SM} \cdot SM_{count}
$$

Where the code gets each term:

* $T_{SM}$, $R_{SM}$, $S_{SM}$, $B_{SM}^{max}$, and `SM_count` come from
  `cudaGetDeviceProperties()`.
* $T_{block}$ is the runtime argument (`threads` in queue mode).
* $R_{thread}=32$ from `-maxrregcount=32` in the Makefile.
* $S_{block}$ is computed as:

  $$
  S_{block} = \text{sizeof(hist[256])} + \text{sizeof(request\_entry)} + \text{sizeof(int)} + 1024
  $$

  The `1024` bytes are the assignment-given shared memory used by
  `interpolate_device()`.

The four ceilings correspond to the four hardware resources an SM has to
share:

| Resource                       | Per-block cost                                  | Source of value          |
|--------------------------------|-------------------------------------------------|--------------------------|
| Threads                        | `threads_per_block`                             | command-line parameter   |
| Registers                      | `threads_per_block × 32`                        | `-maxrregcount=32`       |
| Shared memory                  | `hist[256]·4 + sizeof(request_entry) + sizeof(int) + 1024 (interpolate_device)` | manual + spec            |
| Absolute block cap per SM      | `prop.maxBlocksPerMultiProcessor`              | `cudaGetDeviceProperties` |

Nothing is hard-coded for a specific GPU model — all device limits are read
at runtime via `cudaGetDeviceProperties`. The only fixed values are those
given by the assignment/build setup: 32 registers per thread and 1024 bytes
of shared memory inside `interpolate_device()`.

On Turing (sm_75, 64 KB regs, 64 KB shared, 1024 threads, max 16 blocks per SM):
* 1024 threads → 1 block/SM (capped by threads)
* 512  threads → 2 blocks/SM (capped by threads)
* 256  threads → 4 blocks/SM (capped by threads)

On a T4 (40 SMs) that yields 40 / 80 / 160 concurrent blocks respectively.

### 3.1.1 Background: why this occupancy model is correct

An SM can only host a block if **all** required resources are available at
the same time.

For one candidate block, the scheduler checks four constraints:

* Thread slots: does the SM still have `threads_per_block` thread capacity?
* Register file: can it reserve `threads_per_block * 32` registers?
* Shared memory: can it reserve `shmem_per_block` bytes?
* Architectural block limit: is the per-SM block counter below
  `maxBlocksPerMultiProcessor`?

If any one answer is "no", that extra block cannot be placed. This is why
the valid block count is the minimum of all four ceilings. In other words,
the tightest resource is the bottleneck resource.

This is the same logic used by occupancy calculators: occupancy is not set by
one "main" limit, but by whichever resource runs out first for the current
kernel configuration.

### 3.1.2 Why we compute this at runtime (and why it matters)

The assignment asks for no hard-coded device-specific block count. Runtime
calculation matters for two reasons:

* Portability: different GPUs have different SM counts, register files,
  shared-memory sizes, and max block limits.
* Correct provisioning: `num_blocks` drives both persistent-kernel grid size
  and queue capacity (`next_pow2(16 * num_blocks)`).

If `num_blocks` is too low, GPU workers are underutilized and throughput
drops. If it is too high, extra blocks cannot run concurrently, and queue/
memory sizing may be inflated for no benefit.

So this function is not just a "formula requirement". It determines how many
persistent workers can actually execute together and therefore directly
affects end-to-end server behavior.

### 3.1.3 Beginner walkthrough (with numbers)

You can read `compute_threadblocks_count()` as: "how many workers can one SM
fit, then multiply by number of SMs."

For each SM, we compute 4 candidates:

* `by_threads = maxThreadsPerMultiProcessor / threads_per_block`
* `by_regs    = regsPerMultiprocessor / (threads_per_block * 32)`
* `by_shmem   = sharedMemPerMultiprocessor / shmem_per_block`
* `by_blocks  = maxBlocksPerMultiProcessor`

Then we take:

* `blocks_per_sm = min(by_threads, by_regs, by_shmem, by_blocks)`

because a block must satisfy all constraints at once.

Worked example (typical T4-like values):

* `maxThreadsPerMultiProcessor = 1024`
* `regsPerMultiprocessor = 65536`
* `sharedMemPerMultiprocessor = 65536`
* `maxBlocksPerMultiProcessor = 16`
* `shmem_per_block = hist(256*4) + request_entry + int + 1024 ≈ 2112 bytes`

Case A: `threads_per_block = 1024`

* `by_threads = 1024 / 1024 = 1`
* `by_regs    = 65536 / (1024*32) = 2`
* `by_shmem   = 65536 / 2112 = 31`
* `by_blocks  = 16`
* `blocks_per_sm = min(1,2,31,16) = 1`

Case B: `threads_per_block = 512`

* `by_threads = 1024 / 512 = 2`
* `by_regs    = 65536 / (512*32) = 4`
* `by_shmem   = 31`
* `by_blocks  = 16`
* `blocks_per_sm = min(2,4,31,16) = 2`

Case C: `threads_per_block = 256`

* `by_threads = 1024 / 256 = 4`
* `by_regs    = 65536 / (256*32) = 8`
* `by_shmem   = 31`
* `by_blocks  = 16`
* `blocks_per_sm = min(4,8,31,16) = 4`

If the GPU has 40 SMs, total concurrent workers are:

* `1*40 = 40` (1024 threads)
* `2*40 = 80` (512 threads)
* `4*40 = 160` (256 threads)

This is exactly why the code computes the value dynamically instead of
hard-coding: different GPUs will change these limits and therefore change
the correct number of concurrent blocks.

### 3.2 The MPMC ring queue (`ring_queue<T>`)

Single header struct followed by a flexible array of `T slots[capacity]`:

```cpp
template <typename T>
class ring_queue {
public:
    cuda::atomic<int> head;       // next slot to read  (consumer side)
    cuda::atomic<int> tail;       // next slot to write (producer side)
    int               capacity;   // power of two
    int               mask;       // capacity - 1
    T *slots() { return (T *)(this + 1); }
    ...
};
```

* `head`/`tail` are *monotonically increasing* indices; the actual slot is
  `[index & mask]`. `capacity` is rounded up to the next power of two so
  that masking is a single AND.
* The queue lives in **pinned host memory** (`cudaMallocHost`) so that the
  CPU and the GPU map the same physical pages and both can read/write the
  slot array directly.

`cuda::atomic<int>` defaults to `thread_scope_system`, which is the strict
ordering needed for cross-PCIe synchronization between the host and the
device (same as in `hello-shmem.cu`).

### 3.3 The TAS / TTAS lock

```cpp
struct gpu_lock {
    cuda::atomic<int> flag;
    __device__ void lock() {
        for (;;) {
            while (flag.load(cuda::memory_order_relaxed) != 0) {}
            if (flag.exchange(1, cuda::memory_order_acquire) == 0) return;
        }
    }
    __device__ void unlock() { flag.store(0, cuda::memory_order_release); }
};
```

This is a textbook **TTAS** lock:
1. Spin with a cheap relaxed load until the lock *looks* free (this reads
   from a single shared cache line — no bus traffic until it changes).
2. Only when it looks free do we issue the expensive `exchange` (a
   read-modify-write).

The lock is **allocated with `cudaMalloc`** because read-modify-write
operations are not atomic across PCIe — a lock the GPU has to acquire
must live in GPU memory. Both `req_lock` and `resp_lock` are GPU-resident.

We use two locks (one per queue) as required by the assignment.

### 3.4 Memory ordering (answer to Question 2c)

There are two independent producer/consumer pairs:

```
CPU ──enqueue──▶ req_q  ──dequeue──▶  GPU threadblocks
GPU threadblocks ──enqueue──▶ resp_q ──dequeue──▶ CPU
```

For each pair, the **producer side** (`push`) does:

```
int t = tail.load(memory_order_relaxed);
int h = head.load(memory_order_acquire);             // ⓪ observe free slots
if ((t - h) >= capacity) return false;               // full -> reject
slots[t & mask] = item;                              // ① payload store
tail.store(t + 1, memory_order_release);             // ② publish
```

The **consumer side** (`pop`) does:

```
int h = head.load(memory_order_relaxed);
int t = tail.load(memory_order_acquire);             // ③ observe publish
if (h == t) return false;                            // empty -> reject
item = slots[h & mask];                              // ④ payload load
head.store(h + 1, memory_order_release);             // ⑤ free slot
```

The happens-before edges we need are:

* **Producer → Consumer:** ① *happens-before* ④. This is established by
  the release/acquire pair (② ⇒ ③). Without the release on `tail` the
  consumer could read garbage from `slots[]`.
* **Consumer → Producer:** ④ *happens-before* the next ① that overwrites
  the same slot. This is established by the release/acquire pair on `head`
  (⑤ released by the consumer, acquired by the producer's full check
  inside `push`). Without this the producer could overwrite a slot the
  consumer has not yet finished reading.

In addition, on the *GPU side* the queue is multi-consumer (multi-producer
for `resp_q`). The **critical section** guarded by `req_lock` /
`resp_lock` is:

```
{   pop()/push() : empty/full check + slot read/write + head/tail update   }
```

i.e. everything from observing the queue state to publishing the new
state happens atomically inside a single `pop`/`push` call under the lock. The lock's `exchange(..., memory_order_acquire)` and
`store(0, memory_order_release)` provide the inter-threadblock
synchronization (one block's writes inside the critical section are
visible to the next block that takes the lock).

We deliberately did **not** put a lock on the CPU side: there is only one
CPU producer for `req_q` and one CPU consumer for `resp_q` (the test
harness in `main.cu`), and release/acquire on `head`/`tail` is sufficient.

### 3.5 The persistent kernel

```cpp
__global__ void persistent_kernel(queue_ctx ctx) {
    __shared__ request_entry my_req;
    __shared__ int           sig;

    while (true) {
        // --- dequeue one request (only thread 0 of the block) ---
        if (threadIdx.x == 0) {
            sig = 0;
            for (;;) {
                if (ctx.req_lock->try_lock()) {
                    if (ctx.req_q->pop(my_req)) {     // pop checks empty
                        ctx.req_lock->unlock();
                        sig = 2;  break;
                    }
                    ctx.req_lock->unlock();
                }
                if (ctx.stop_flag->load(memory_order_acquire) != 0
                    && ctx.req_q->is_empty()) { sig = 1; break; }
            }
        }
        __syncthreads();
        if (sig == 1) break;

        // --- process image with the whole block ---
        process_image(my_req.img_in, my_req.img_out, my_maps);
        __syncthreads();

        // --- enqueue response (only thread 0) ---
        if (threadIdx.x == 0) {
            response_entry r{my_req.img_id};
            for (;;) {
                if (ctx.resp_lock->try_lock()) {
                    if (ctx.resp_q->push(r)) {        // push checks full
                        ctx.resp_lock->unlock(); break;
                    }
                    ctx.resp_lock->unlock();
                }
            }
        }
        __syncthreads();
    }
}
```

Important details:

* **Only thread 0 of a block** takes/releases the locks. The whole block
  then cooperates in `process_image`. This avoids unnecessary contention
  on the lock and is the natural fit because each request is one
  threadblock's worth of work.
* The dequeued request is copied into `__shared__` (`my_req`) so all
  threads in the block see it after the `__syncthreads`.
* The `sig` flag (also `__shared__`) is how thread 0 communicates the
  decision (got a request / stop) to its peers.
* The stop check is **outside** the request lock so a parked threadblock
  cannot starve while another block is in the critical section.

### 3.6 `queue_server` lifecycle

Constructor:

1. Compute `num_blocks` and `queue_capacity = next_pow2(16 * num_blocks)`.
2. `cudaMallocHost` both queues, then call `init(capacity)` which
   placement-news `head` and `tail` to 0.
3. `cudaMalloc` and `cudaMemset(0)` the two GPU locks.
4. `cudaMallocHost` the stop flag (system-scope atomic).
5. `cudaMalloc` the per-block maps pool: `num_blocks * MAPS_PER_IMAGE` bytes.
6. Launch `persistent_kernel<<<num_blocks, threads>>>`.

Destructor:

1. `h_stop_flag->store(1, release)` — every threadblock will see this
   the next time it loops in the dequeue spin and is observed *only*
   after the local request queue is empty, so we never lose a request.
2. `cudaDeviceSynchronize()` — wait for the kernel to actually exit.
3. Free GPU buffers, destruct atomics, free pinned buffers.

`enqueue` returns `false` if the request queue is full (so the test
harness retries); `dequeue` returns `false` if the response queue is
empty.

### 3.7 Why these choices (short rationale)

This is the "why" behind the Part-2 design decisions:

* **Persistent kernel instead of per-request launches:** removes repeated
  kernel launch overhead and keeps GPU workers hot, which is better for a
  server-like request stream.
* **Single shared request/response queues:** gives natural load balancing;
  whichever block becomes free can take the next request.
* **Pinned host memory for queues (`cudaMallocHost`):** CPU and GPU can
  access the same queue storage directly, enabling CPU↔GPU message passing.
* **GPU-resident locks (`cudaMalloc`):** lock acquisition uses RMW atomics
  (`exchange`), which must be local to the GPU for correctness.
* **Release/acquire memory ordering:** guarantees queue payload visibility
  before publishing queue indices; prevents stale or reordered reads.
* **One lock per queue (`req_lock`, `resp_lock`):** minimizes critical
  section scope while still serializing multi-block queue updates safely.
* **Only thread 0 handles queue ops in each block:** reduces lock contention;
  full block still participates in compute-heavy image processing.
* **Stop flag + device synchronize in destructor:** ensures clean shutdown
  (no stuck persistent kernel, no freeing memory while still in use).

#### Worked example — a request through the queue server

Assume `threads=1024`, Turing T4 (40 SMs) → `num_blocks=40`,
`queue_capacity = next_pow2(16·40) = next_pow2(640) = 1024` slots.

CPU side (single producer / single consumer, no CPU lock):

```text
enqueue(42, in, out):
    req_q.is_full()? no
    slots[tail & 1023] = {42, in, out}      ① payload store
    tail.store(tail+1, release)             ② publish  → returns true
```

GPU side — one of the 40 persistent blocks (only thread 0 touches locks):

```text
thread 0:
    req_lock.try_lock()      → got it (TTAS: relaxed load, then acquire-exchange)
    req_q.is_empty()? no
    my_req = req_q.pop()      ③ acquire tail, ④ read slot, ⑤ release head
    req_lock.unlock()         (release-store 0)
    sig = 2
__syncthreads()              → all 1024 threads now see my_req

process_image(my_req.img_in, my_req.img_out, my_maps)   // whole block cooperates
__syncthreads()

thread 0:
    resp_lock.try_lock()     → got it
    resp_q.is_full()? no
    resp_q.push({42})        publish response
    resp_lock.unlock()
```

CPU side again:

```text
dequeue(&id):
    resp_q.is_empty()? no
    id = resp_q.pop() = 42   → returns true   (req 42 completed)
```

Shutdown (destructor):

```text
h_stop_flag.store(1, release)
  ↳ each block, after draining req_q, sees stop_flag != 0 → sig = 1 → break
cudaDeviceSynchronize()      → kernel exits, then we free everything
```

This illustrates the two release/acquire handshakes (one per queue) and
why the locks only need to wrap the GPU-side critical section.

---

## 4. Answers to the discussion questions

### (h) What do the throughput vs. #threads curves teach us?

Going from 1024 → 512 → 256 threads per block:

* The amount of **parallelism inside one block** drops, so the *latency
  per image* grows.
* The number of **concurrently resident blocks per SM** grows
  (1 → 2 → 4 on Turing), so the SM can hide more memory latency.
* Net throughput is the product of "blocks per SM × images per second
  per block". With more, smaller blocks we get better SM utilization
  *if* the kernel was previously occupancy-limited. If it was
  compute-limited the extra blocks just trade latency for queue depth.

Observed (random workload, T4 device 2):

| Configuration              | Throughput (req/sec) | Median latency (ms) |
|----------------------------|----------------------:|--------------------:|
| streams (1 block, 64 streams) |              ~47 000 |               ~0.06 |
| queue, 1024 threads        |               ~7 800 |                ~800 |
| queue,  512 threads        |               ~3 050 |               ~1700 |
| queue,  256 threads        |               ~1 430 |               ~3750 |

The queue server is much slower than the streams server because the
persistent kernel reads each pixel directly from pinned host memory
across PCIe (no H→D copy), while the streams server copies the image
into GPU DRAM once and processes it from there. Lowering the thread
count further degrades the queue server because each block is slower
and the extra concurrent blocks cannot compensate.

### (i) Per-block queues vs. one shared queue

A pair of queues *per threadblock* would eliminate inter-block lock
contention and remove the TAS/TTAS locks entirely (each queue has
exactly one producer and one consumer). The downside is **load
imbalance**: if one block happens to finish later, its private input
queue could be full while another block's queue is empty, so the CPU
producer would either skip "blocked" queues (complicating enqueue and
biasing fairness) or stall.

A single shared queue acts as **automatic load balancing**: any free
block grabs the next pending request. The cost is the global lock and
the cache-line contention on `head`/`tail`. In our measurements the
shared queue wins because the lock is taken only once per *image* (one
thread per block), not once per CUDA thread.

### (j) Move CPU-to-GPU queue into GPU memory

Today the CPU *writes* into the CPU-to-GPU queue. PCIe writes from the
CPU are **posted** transactions: the CPU fires and forgets, the write
goes through write-combining buffers and a single DMA burst. PCIe
*reads* (which is what the GPU does when polling the queue in pinned
host memory) are **non-posted**: the GPU sends a Memory-Read TLP, must
wait for a Completion TLP, and the round-trip latency is a few µs.

If the CPU-to-GPU queue lived in **GPU memory**:

* CPU enqueue becomes a **PCIe write** to the GPU (posted, very fast).
* GPU dequeue becomes a **local DRAM read** (tens of ns, fully cached).

So we replace the slow polling reads with cheap posted writes, removing
PCIe round-trips from the hot path. The GPU-to-CPU queue should stay in
host memory for the symmetric reason: the GPU posts writes, the CPU
reads locally from DRAM.

### (k) How to expose GPU memory to the CPU

A region of the GPU's framebuffer has to be exposed as a PCIe **BAR**
window. The GPU driver maps that BAR into the kernel's physical address
space and then into the requesting process via an
`mmap`-able device node (in CUDA this is what `cudaHostRegister` /
`cudaHostGetDevicePointer` / GPUDirect peer mapping do under the hood).
Once mapped, CPU stores to the virtual address translate (via the
chipset and the IOMMU) into PCIe MMIO writes that the GPU's PCIe block
forwards into the BAR-backed region of HBM/GDDR. Care must be taken with
write-combining and `wmb()` style barriers to ensure ordering, because
MMIO writes from the CPU are reordered through the chipset's posted
write buffers.

---

## 5. Build & run

```bash
make ex2
./ex2 streams <load>            # part 1
./ex2 queue   <threads> <load>  # part 2; threads ∈ {256, 512, 1024}
```

`make` produces no warnings; every run prints
`distance from baseline 0 (should be zero)` confirming bit-for-bit
identity with the CPU reference.
