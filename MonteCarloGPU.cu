#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#define N 1024
#define BLOCK_SIZE 256

__global__ void initRandomStates(curandState *state, unsigned long seed, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        curand_init(seed, id, 0, &state[id]);
    }
}

__global__ void generateUniform(curandState *state, float *randomNumbers, int n) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < n) {
        randomNumbers[id] = curand_uniform(&state[id]);
    }
}

__global__ void generateNormal(curandState *state, float *randomNumbers, int n) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
      float u1 = curand_uniform(&state[id]);
      float u2 = curand_uniform(&state[id]);
      randomNumbers[id] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2); // Z ~ N(0, 1)
    }
}

__global__ void simulateStockPaths(curandState *state, float *stockPrices, float S0, float r, float sigma, float T, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        float u1 = curand_uniform(&state[id]);
        float u2 = curand_uniform(&state[id]);
        float Z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);

         // Calculate stock price using Black-Scholes formula
        stockPrices[id] = S0 * expf((r - 0.5f * sigma * sigma) * T + sigma * sqrtf(T) * Z);
    }
}

__global__ void calculatePayoff(float *stockPrices, float *payoffs, float k, int n) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    if (id < n) {
        payoffs[id] = fmax(stockPrices[id] - k, 0.0f);
    }
}

float calculateOptionPrice(float *payoffs, int n, float r, float T) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += payoffs[i];
    }
    return expf(-r * T) * (sum / n);
}

__global__ void calculatePayoffShared(float *stockPrices, float *payoffs, float K, int n) {
    __shared__ float sharedPayoffs[256]; // Shared memory for thread block

    int tid = threadIdx.x;
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Calculate payoff
    sharedPayoffs[tid] = (id < n) ? fmaxf(stockPrices[id] - K, 0.0f) : 0.0f;

    __syncthreads();

    // Reduce payoffs within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedPayoffs[tid] += sharedPayoffs[tid + stride];
        }
        __syncthreads();
    }

    // Write block's result to global memory
    if (tid == 0) {
        atomicAdd(payoffs, sharedPayoffs[0]);
    }
}

void saveToCSV(const float *data, int size, const std::string &filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (int i = 0; i < size; i++) {
            file << data[i] << "\n";
        }
        file.close();
        std::cout << "Data saved to " << filename << std::endl;
    }
    else {
        std::cerr << "Error Opening File: " << filename << std::endl;
    }
}


int main() {
    auto start_cpu = std::chrono::high_resolution_clock::now();

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);

    const float S0 = 100.0f;  // Initial stock price
    const float r = 0.05f;    // Risk-free interest rate
    const float sigma = 0.2f; // Volatility
    const float T = 1.0f;     // Time to maturity (in years)

    // Allocate memory for stock prices and RNG states
    float *d_stockPrices;
    curandState *d_states;
    cudaMalloc(&d_stockPrices, N * sizeof(float));
    cudaMalloc(&d_states, N * sizeof(curandState));

    // Initialize random states
    initRandomStates<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_states, 1234, N);

    cudaDeviceSynchronize();

    // Simulate stock price paths
    simulateStockPaths<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_states, d_stockPrices, S0, r, sigma, T, N
    );
    cudaDeviceSynchronize();

    // Copy results back to host
    float *h_stockPrices = new float[N];
    cudaMemcpy(h_stockPrices, d_stockPrices, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the first few stock prices
    for (int i = 0; i < 10; ++i) {
        std::cout << "Stock Price " << i + 1 << ": " << h_stockPrices[i] << std::endl;
    }

        // Allocate memory for payoffs
    float *d_payoffs;
    cudaMalloc(&d_payoffs, N * sizeof(float));

    // Calculate payoffs
    const float K = 100.0f;  // Strike price
    calculatePayoff<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_stockPrices, d_payoffs, K, N
    );

    // Copy payoffs back to host
    float *h_payoffs = new float[N];
    cudaMemcpy(h_payoffs, d_payoffs, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Calculate option price
    float optionPrice = calculateOptionPrice(h_payoffs, N, r, T);
    std::cout << "\nOption Price: " << optionPrice << std::endl;

    float *d_totalPayoff;
    cudaMalloc(&d_totalPayoff, sizeof(float));
    cudaMemset(d_totalPayoff, 0, sizeof(float));

    calculatePayoffShared<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_stockPrices, d_totalPayoff, K, N
    );

    cudaDeviceSynchronize();

    float totalPayoff;
    cudaMemcpy(&totalPayoff, d_totalPayoff, sizeof(float), cudaMemcpyDeviceToHost);
    optionPrice = expf(-r * T) * (totalPayoff / N);

    saveToCSV(h_stockPrices, N, "stock_prices.csv");

    // Free memory
    delete[] h_payoffs;
    cudaFree(d_payoffs);
    delete[] h_stockPrices;
    cudaFree(d_stockPrices);
    cudaFree(d_states);

    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end_cpu - start_cpu;
    printf("CPU execution time: %f ms\n", duration.count());

    cudaEventRecord(stop_gpu);

    cudaEventSynchronize(stop_gpu);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);

    printf("Kernel execution time: %f ms\n", milliseconds);

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
