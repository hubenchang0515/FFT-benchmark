#include "config.h"
#include <cstdio>
#include <cstdlib>
#include <cuda/std/cmath>
#include <cuda/std/complex>

using Complex = cuda::std::complex<double>;

__device__ 
Complex omega(int k, int n)
{
    double angle = 2.0*pi*k/n;
    return Complex{cuda::std::cos(angle), -cuda::std::sin(angle)};
}

__global__
void polynomialItems(const Complex* K, size_t n, Complex* items, int forward=1)
{
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= n || tid >= n)
        return;

    items[i] = K[tid] * cuda::std::pow(omega(forward * bid, n), tid);
}

__global__
void sumItems(const Complex* items, size_t n, Complex* values)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    for (size_t j = 0; j < n; j++)
    {
        values[i] += items[i * n + j];
    }
}

__device__
Complex polynomialValue(const Complex* K, size_t n, const Complex& w)
{
    Complex v = 0;
    for (size_t i = 0; i < n; i++)
    {
        v += K[i] * cuda::std::pow(w, i);
    }

    return v;
}

__global__
void transformForDft(const Complex* in, size_t n, Complex* out, int forward=1)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    out[i] = polynomialValue(in, n, omega(forward * i, n));
}

__global__
void div(Complex* data, size_t n, double v)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    data[i] /= v;
}

// 显存开销和并法规模较大的方案
void dft1(const Complex* in, size_t n, Complex* out, int forward=1)
{
    // 分配显存
    Complex* devIn;
    Complex* devOut;
    Complex* devItems;
    cudaMalloc(&devIn, n * sizeof(Complex));
    cudaMalloc(&devOut, n * sizeof(Complex));
    cudaMalloc(&devItems, n * n * sizeof(Complex));

    // 将内存数据拷贝到显存
    cudaMemcpy(devIn, in, n * sizeof(Complex), cudaMemcpyHostToDevice);

    // 计算
    polynomialItems<<<n, n>>>(devIn, n, devItems, forward);
    sumItems<<<1, n>>>(devItems, n, devOut);

    if (forward < 0)
    {
        div<<<1, n>>>(devOut, n, n);
    }

    // 将显存数据拷贝到内存
    cudaMemcpy(out, devOut, n * sizeof(Complex), cudaMemcpyDeviceToHost);

    // 释放显存
    cudaFree(devItems);
    cudaFree(devOut);
    cudaFree(devIn);
}

// 显存开销和并法规模较小的方案
void dft2(const Complex* in, size_t n, Complex* out, int forward=1)
{
    // 分配显存
    Complex* devIn;
    Complex* devOut;
    cudaMalloc(&devIn, n * sizeof(Complex));
    cudaMalloc(&devOut, n * sizeof(Complex));

    // 将内存数据拷贝到显存
    cudaMemcpy(devIn, in, n * sizeof(Complex), cudaMemcpyHostToDevice);

    transformForDft<<<(n+1023)/1024, 1024>>>(devIn, n, devOut, forward);

    if (forward < 0)
    {
        div<<<(n+1023)/1024, 1024>>>(devOut, n, n);
    }

    // 将显存数据拷贝到内存
    cudaMemcpy(out, devOut, n * sizeof(Complex), cudaMemcpyDeviceToHost);

    // 释放显存
    cudaFree(devOut);
    cudaFree(devIn);
}

void dft(const Complex* in, size_t n, Complex* out, int forward=1)
{
    if (n <= 1024)
        dft1(in, n, out, forward);
    else
        dft2(in, n, out, forward);
}

void idft(const Complex* values, size_t n, Complex* coefficients)
{
    dft(values, n, coefficients, -1);
}

// 拆分系数向量
__global__
void split(const Complex* in, size_t n, Complex* out)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    out[i] = in[i*2];
    out[i+n/2] = in[i*2+1];
}

// 循环拆分系数向量直到最底层
void splitAll(Complex** devData, Complex** devTemp, size_t n)
{
    // log(n)轮计算
    for (size_t groupSize = n; groupSize > 2; groupSize = groupSize/2)
    {
        size_t blocks = groupSize/2 <= 1024 ? 1 : ((groupSize/2 + 1023) / 1024);
        size_t threads = groupSize/2 <= 1024 ? (groupSize/2) : 1024;
        for (size_t i = 0; i < n; i+=groupSize)
        {
            split<<<blocks, threads>>>(*devData + i, groupSize, *devTemp + i);
        }
        cuda::std::swap(*devData, *devTemp);
    }
}

// 从底层分组向上合并
__global__ 
void combine(Complex* data, size_t n, size_t groupSize, Complex* temp, int forward=1)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + groupSize/2 >= n)
        return;

    
    size_t groupIndex = i % groupSize;

    // 右半边和左半边一起计算出，因此右半直接跳过
    if (groupIndex >= groupSize/2)
        return;

    auto w = omega(forward*groupIndex, groupSize);
    temp[i] = data[i] + w * data[i + groupSize/2]; // 左半
    temp[i+ groupSize/2] = data[i] - w * data[i + groupSize/2]; // 右半
}


void fft(Complex* data, size_t n, int forward=1)
{
    Complex* temp = new Complex[n];

    // 分配显存
    Complex* devData;
    Complex* devTemp;
    cudaMalloc(&devData, n * sizeof(Complex));
    cudaMalloc(&devTemp, n * sizeof(Complex));

    cudaMemcpy(devData, data, n * sizeof(Complex), cudaMemcpyHostToDevice);

    // 将内存数据拷贝到显存
    splitAll(&devData, &devTemp, n);

    size_t blocks = n <= 1024 ? 1 : ((n + 1023) / 1024);
    size_t threads = n <= 1024 ? n : 1024;
    for (size_t groupSize = 2; groupSize <= n; groupSize *= 2)
    {
        combine<<<blocks, threads>>>(devData, n, groupSize, devTemp, forward);
        cuda::std::swap(devData, devTemp);
    }

    if (forward < 0)
    {
        div<<<blocks, threads>>>(devData, n, n);
    }

    // 将显存数据拷贝回内存
    cudaMemcpy(data, devData, n * sizeof(Complex), cudaMemcpyDeviceToHost);

    cudaFree(devData);
    cudaFree(devTemp);
}

void ifft(Complex* data, size_t n)
{
    fft(data, n, -1);
}

int main()
{
    // 分配内存并初始化数据
    Complex* data = new Complex[N];

    for (size_t i = 0; i < N; i++)
    {
        data[i].real(i);
        data[i].imag(i);
    }

    // 变换
    fft(data, N);
    for (size_t i = 0; i < N; i++)
    {
        printf("%f + %fi\n", data[i].real(), data[i].imag());
    }
    printf("\n");

    // 逆变换
    ifft(data, N);
    for (size_t i = 0; i < N; i++)
    {
        printf("%f + %fi\n", data[i].real(), data[i].imag());
    }

    delete[] data;

    return 0;
}