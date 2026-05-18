/* test.cu – Correctness tests for ex1 GPU implementations.
 *
 * Compile & run:  make test && ./test
 *
 * Each test fills all N_IMAGES images with a specific pattern, runs the CPU
 * reference, the GPU task-serial implementation, and the GPU bulk
 * implementation, then verifies that both GPU outputs match the CPU output
 * exactly (squared-distance == 0).
 */

#include "ex1.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>

// ── image generators ────────────────────────────────────────────────────────

static void fill_constant(uchar *imgs, uchar val)
{
    memset(imgs, val, (size_t)N_IMAGES * IMG_WIDTH * IMG_HEIGHT);
}

static void fill_gradient_h(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img = imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        for (int r = 0; r < IMG_HEIGHT; r++)
            for (int c = 0; c < IMG_WIDTH; c++)
                img[r * IMG_WIDTH + c] = (uchar)((unsigned)c * 255 / (IMG_WIDTH - 1));
    }
}

static void fill_gradient_v(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img = imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        for (int r = 0; r < IMG_HEIGHT; r++)
            for (int c = 0; c < IMG_WIDTH; c++)
                img[r * IMG_WIDTH + c] = (uchar)((unsigned)r * 255 / (IMG_HEIGHT - 1));
    }
}

static void fill_checkerboard(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img = imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        for (int r = 0; r < IMG_HEIGHT; r++)
            for (int c = 0; c < IMG_WIDTH; c++) {
                int tr = r / TILE_WIDTH, tc = c / TILE_WIDTH;
                img[r * IMG_WIDTH + c] = ((tr + tc) & 1) ? 255 : 0;
            }
    }
}

static void fill_random(uchar *imgs, unsigned seed)
{
    srand(seed);
    size_t n = (size_t)N_IMAGES * IMG_WIDTH * IMG_HEIGHT;
    for (size_t i = 0; i < n; i++)
        imgs[i] = (uchar)(rand() & 0xFF);
}

static void fill_two_values(uchar *imgs, uchar a, uchar b)
{
    size_t n = (size_t)N_IMAGES * IMG_WIDTH * IMG_HEIGHT;
    for (size_t i = 0; i < n; i++)
        imgs[i] = (i & 1) ? b : a;
}

/* Each image gets a unique constant value (image i → value i % 256).
 * This is the most important extra test: verifies blocks don't interfere. */
static void fill_mixed_constants(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar val = (uchar)(i % 256);
        memset(imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT, val,
               IMG_WIDTH * IMG_HEIGHT);
    }
}

/* Each image gets a unique random seed – different data in every block. */
static void fill_mixed_random(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        srand((unsigned)i * 2654435761u); /* Knuth multiplicative hash */
        uchar *img = imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        for (int p = 0; p < IMG_WIDTH * IMG_HEIGHT; p++)
            img[p] = (uchar)(rand() & 0xFF);
    }
}

/* 1-pixel-wide alternating stripes (thin = very different histogram shape). */
static void fill_stripes_1px(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img = imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        for (int r = 0; r < IMG_HEIGHT; r++)
            for (int c = 0; c < IMG_WIDTH; c++)
                img[r * IMG_WIDTH + c] = (uchar)(((r + c) & 1) ? 255 : 0);
    }
}

/* Value flips exactly at every tile boundary (stresses tile-edge interpolation). */
static void fill_tile_boundary_flip(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img = imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        for (int r = 0; r < IMG_HEIGHT; r++)
            for (int c = 0; c < IMG_WIDTH; c++) {
                int tr = r / TILE_WIDTH, tc = c / TILE_WIDTH;
                img[r * IMG_WIDTH + c] = (uchar)(((tr ^ tc) & 1) ? 200 : 50);
            }
    }
}

/* Only one pixel per tile is non-zero (sparse histogram). */
static void fill_sparse(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img = imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        memset(img, 0, IMG_WIDTH * IMG_HEIGHT);
        /* set the centre pixel of each tile to 200 */
        int half = TILE_WIDTH / 2;
        for (int tr = 0; tr < TILE_COUNT; tr++)
            for (int tc = 0; tc < TILE_COUNT; tc++) {
                int r = tr * TILE_WIDTH + half;
                int c = tc * TILE_WIDTH + half;
                img[r * IMG_WIDTH + c] = 200;
            }
    }
}

/* Diagonal gradient (combination of row + col index). */
static void fill_diagonal(uchar *imgs)
{
    for (int i = 0; i < N_IMAGES; i++) {
        uchar *img = imgs + (size_t)i * IMG_WIDTH * IMG_HEIGHT;
        for (int r = 0; r < IMG_HEIGHT; r++)
            for (int c = 0; c < IMG_WIDTH; c++)
                img[r * IMG_WIDTH + c] =
                    (uchar)(((unsigned)(r + c)) * 255 / ((IMG_HEIGHT - 1) + (IMG_WIDTH - 1)));
    }
}

// ── small wrappers so every fill has the same void(*)(uchar*) signature ─────

static void fill_zeros(uchar *i)      { fill_constant(i, 0);          }
static void fill_white(uchar *i)      { fill_constant(i, 255);        }
static void fill_gray(uchar *i)       { fill_constant(i, 128);        }
static void fill_grad_h(uchar *i)     { fill_gradient_h(i);           }
static void fill_grad_v(uchar *i)     { fill_gradient_v(i);           }
static void fill_chess(uchar *i)      { fill_checkerboard(i);         }
static void fill_alt_0_255(uchar *i)  { fill_two_values(i, 0, 255);   }
static void fill_alt_64_192(uchar *i) { fill_two_values(i, 64, 192);  }
static void fill_rand42(uchar *i)     { fill_random(i, 42);           }
static void fill_rand123(uchar *i)    { fill_random(i, 123);          }
static void fill_rand999(uchar *i)    { fill_random(i, 999);          }
static void fill_mixed_c(uchar *i)    { fill_mixed_constants(i);      }
static void fill_mixed_r(uchar *i)    { fill_mixed_random(i);         }
static void fill_stripes(uchar *i)    { fill_stripes_1px(i);          }
static void fill_tbflip(uchar *i)     { fill_tile_boundary_flip(i);   }
static void fill_spar(uchar *i)       { fill_sparse(i);               }
static void fill_diag(uchar *i)       { fill_diagonal(i);             }

// ── test table ───────────────────────────────────────────────────────────────

struct TestCase { const char *name; void (*fill)(uchar *); };

static const TestCase TESTS[] = {
    { "All zeros (black image)",           fill_zeros      },
    { "All 255  (white image)",            fill_white      },
    { "All 128  (mid-gray image)",         fill_gray       },
    { "Horizontal gradient  0→255",        fill_grad_h     },
    { "Vertical gradient    0→255",        fill_grad_v     },
    { "Diagonal gradient",                 fill_diag       },
    { "Checkerboard (tile-sized squares)", fill_chess      },
    { "Alternating pixels   0/255",        fill_alt_0_255  },
    { "Alternating pixels  64/192",        fill_alt_64_192 },
    { "1-pixel stripes (thin pattern)",    fill_stripes    },
    { "Tile-boundary value flip",          fill_tbflip     },
    { "Sparse (1 hot pixel per tile)",     fill_spar       },
    { "Mixed batch: unique constant/img",  fill_mixed_c    },
    { "Mixed batch: unique random/img",    fill_mixed_r    },
    { "Random (seed  42)",                 fill_rand42     },
    { "Random (seed 123)",                 fill_rand123    },
    { "Random (seed 999)",                 fill_rand999    },
};
static const int N_TESTS = (int)(sizeof(TESTS) / sizeof(TESTS[0]));

// ── helpers ──────────────────────────────────────────────────────────────────

static long long sq_distance(const uchar *a, const uchar *b, size_t n)
{
    long long d = 0;
    for (size_t i = 0; i < n; i++) {
        long long diff = (long long)a[i] - b[i];
        d += diff * diff;
    }
    return d;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main()
{
    /* ---- device selection (match main.cu) ---- */
    int device_id = 3;
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (device_id >= ndev) device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));
    printf("Using device %d\n\n", device_id);

    const size_t total = (size_t)N_IMAGES * IMG_WIDTH * IMG_HEIGHT;

    /* ---- allocate pinned host buffers ---- */
    uchar *imgs_in, *imgs_cpu, *imgs_serial, *imgs_bulk, *imgs_bulk2;
    CUDA_CHECK(cudaHostAlloc(&imgs_in,     total, 0));
    CUDA_CHECK(cudaHostAlloc(&imgs_cpu,    total, 0));
    CUDA_CHECK(cudaHostAlloc(&imgs_serial, total, 0));
    CUDA_CHECK(cudaHostAlloc(&imgs_bulk,   total, 0));
    CUDA_CHECK(cudaHostAlloc(&imgs_bulk2,  total, 0));

    /* ---- init GPU contexts once ---- */
    struct task_serial_context *ts = task_serial_init();
    struct gpu_bulk_context    *gb = gpu_bulk_init();

    int passed = 0, failed = 0;

    // ── Section 1: correctness ────────────────────────────────────────────
    printf("=== Correctness tests ===\n");
    printf("%-42s  %8s  %8s  %8s  %s\n",
           "Test", "ser↔cpu", "bulk↔cpu", "ser↔bulk", "Result");
    printf("%s\n", "------------------------------------------------------------------------------");

    for (int t = 0; t < N_TESTS; t++) {
        TESTS[t].fill(imgs_in);

        for (int i = 0; i < N_IMAGES; i++)
            cpu_process(imgs_in  + (size_t)i * IMG_WIDTH * IMG_HEIGHT,
                        imgs_cpu + (size_t)i * IMG_WIDTH * IMG_HEIGHT,
                        IMG_WIDTH, IMG_HEIGHT);

        task_serial_process(ts, imgs_in, imgs_serial);
        long long d_serial = sq_distance(imgs_cpu, imgs_serial, total);

        gpu_bulk_process(gb, imgs_in, imgs_bulk);
        long long d_bulk = sq_distance(imgs_cpu, imgs_bulk, total);

        long long d_sb = sq_distance(imgs_serial, imgs_bulk, total);

        bool ok = (d_serial == 0 && d_bulk == 0 && d_sb == 0);
        printf("%-42s  %8lld  %8lld  %8lld  [%s]\n",
               TESTS[t].name, d_serial, d_bulk, d_sb, ok ? "PASS" : "FAIL");
        if (ok) passed++; else failed++;
    }
    printf("%s\n\n", "------------------------------------------------------------------------------");

    // ── Section 2: determinism – same input must give same output ─────────
    printf("=== Determinism test (random seed 42, run twice) ===\n");
    fill_random(imgs_in, 42);
    gpu_bulk_process(gb, imgs_in, imgs_bulk);   /* run 1 */
    gpu_bulk_process(gb, imgs_in, imgs_bulk2);  /* run 2 */
    {
        long long d = sq_distance(imgs_bulk, imgs_bulk2, total);
        bool ok = (d == 0);
        printf("  bulk run1 vs run2 distance: %lld  [%s]\n\n", d, ok ? "PASS" : "FAIL");
        if (ok) passed++; else failed++;
    }

    // ── Section 3: performance – bulk must be >> faster than serial ───────
    printf("=== Performance test ===\n");
    fill_random(imgs_in, 77);

    double t0 = get_time_msec();
    task_serial_process(ts, imgs_in, imgs_serial);
    double t_serial = get_time_msec() - t0;

    t0 = get_time_msec();
    gpu_bulk_process(gb, imgs_in, imgs_bulk);
    double t_bulk = get_time_msec() - t0;

    double speedup = t_serial / t_bulk;
    /* Bulk parallelises all N_IMAGES blocks; expect at least 10× speedup. */
    bool perf_ok = (speedup >= 10.0);
    printf("  task serial: %.1f ms  |  bulk: %.1f ms  |  speedup: %.1fx  [%s]\n\n",
           t_serial, t_bulk, speedup, perf_ok ? "PASS" : "FAIL");
    if (perf_ok) passed++; else failed++;

    // ── Summary ───────────────────────────────────────────────────────────
    printf("%d / %d tests passed.\n", passed, N_TESTS + 2 /* determinism + perf */);

    /* ---- cleanup ---- */
    task_serial_free(ts);
    gpu_bulk_free(gb);
    CUDA_CHECK(cudaFreeHost(imgs_in));
    CUDA_CHECK(cudaFreeHost(imgs_cpu));
    CUDA_CHECK(cudaFreeHost(imgs_serial));
    CUDA_CHECK(cudaFreeHost(imgs_bulk));
    CUDA_CHECK(cudaFreeHost(imgs_bulk2));

    return (failed > 0) ? 1 : 0;
}
