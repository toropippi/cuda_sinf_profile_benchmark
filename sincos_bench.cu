// sincos_bench.cu
// ----------------------------------------------------------------------------
// CUDA Compute Benchmark for sin/cos and Mandelbrot overlap
// ----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// エラーチェックをシンプルな関数に
inline void checkCuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "CUDA error %s (%d) at %s:%d\n",
                     cudaGetErrorString(err), int(err), file, line);
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(err) (checkCuda((err), __FILE__, __LINE__))

// モードを列挙型で定義
enum class Mode : int {
    SinMandel = 1,  // sin + Mandelbrot
    SinOnly    = 2,  // sin only
    MandelOnly = 3,  // Mandelbrot only
    CosOnly    = 4   // cos only
};

// マンデルブロー１ステップ
__device__ __forceinline__ float2 mandelbrot_step(float2 z) {
    const float2 c = { -0.7f, 0.27015f };
    // z := (x^2 - y^2, 2xy) + c
    return make_float2(
        z.x * z.x - z.y * z.y + c.x,
        2.0f * z.x * z.y + c.y
    );
}

// ベンチマークカーネル（テンプレートで分岐を解消）
template<Mode M>
__global__ void benchKernel(uint32_t iterations, float* result) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    float x  = 0.000123f * (gid + 1);
    float2 z = make_float2(0.0001f * (gid / 64), x);

    for (uint32_t i = 0; i < iterations; ++i) {
        if constexpr (M == Mode::SinMandel) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                x = __sinf(x);
                z = mandelbrot_step(z);
            }
        }
        else if constexpr (M == Mode::SinOnly) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                x = __sinf(x);
            }
        }
        else if constexpr (M == Mode::MandelOnly) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                z = mandelbrot_step(z);
            }
        }
        else { // Mode::CosOnly
            x = __cosf(x);
        }
    }

    // ブロック内スレッド 0 が書き込み
    if (threadIdx.x == 0) {
        result[0] = x + z.x;
    }
}

// カーネル起動支援関数
void launch(Mode mode, uint32_t iterations, int device = 0) {
    CUDA_CHECK(cudaSetDevice(device));

    // Managed メモリで簡潔に
    float* d_result = nullptr;
    CUDA_CHECK(cudaMallocManaged(&d_result, sizeof(float)));
    d_result[0] = 0.0f;

    constexpr int THREADS = 1024;
    constexpr int BLOCKS  = 64;

    switch (mode) {
        case Mode::SinMandel:
            benchKernel<Mode::SinMandel><<<BLOCKS, THREADS>>>(iterations, d_result);
            break;
        case Mode::SinOnly:
            benchKernel<Mode::SinOnly><<<BLOCKS, THREADS>>>(iterations, d_result);
            break;
        case Mode::MandelOnly:
            benchKernel<Mode::MandelOnly><<<BLOCKS, THREADS>>>(iterations, d_result);
            break;
        case Mode::CosOnly:
            benchKernel<Mode::CosOnly><<<1, 1>>>(iterations, d_result); // Cos はシングル
            break;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("Result = %f\n", d_result[0]);
    CUDA_CHECK(cudaFree(d_result));
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::printf("Usage: %s <mode 1-4> <iterations> [device]\n", argv[0]);
        std::printf(" 1: Sin+Mandel  2: Sin only  3: Mandel only  4: Cos only\n");
        return 0;
    }

    int m = std::atoi(argv[1]);
    uint32_t iters = std::atoi(argv[2]);
    int dev = (argc >= 4) ? std::atoi(argv[3]) : 0;

    if (m < 1 || m > 4) {
        std::fprintf(stderr, "Invalid mode: %d\n", m);
        return EXIT_FAILURE;
    }

    launch(static_cast<Mode>(m), iters, dev);
    return 0;
}
