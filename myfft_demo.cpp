#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <complex>
#include "config.h"

using Complex = std::complex<double>;

// 复数中的单位根
Complex omega(int k, int n)
{
    double angle = 2.0*pi*k/n;
    return Complex{std::cos(angle), -std::sin(angle)};
}

// 多项式求值 k0 + k1*x + k2*x^2 + k3 * x^3 + ... + k<n-1> * x^<n-1>
Complex polynomialValue(const std::vector<Complex>& K, const Complex& w)
{
    Complex v = 0;
    size_t n = K.size();
    for (size_t i = 0; i < n; i++)
    {
        v += K[i] * std::pow(w, i);
    }

    return v;
}

std::vector<Complex> transformForDft(const std::vector<Complex>& in, int forward=1)
{
    size_t n = in.size();
    std::vector<Complex> out(n);
    for(size_t i = 0; i < n; i++)
    {
        out[i] = polynomialValue(in, omega(forward * i, n));
    }
    return out;
}

std::vector<Complex> dft(const std::vector<Complex>& coefficients)
{
    return transformForDft(coefficients);
}

std::vector<Complex> idft(const std::vector<Complex>& values)
{
    size_t n = values.size();
    auto out = transformForDft(values, -1);
    for (auto& v : out)
    {
        v /= n;
    }

    return out;
}

// 拆分系数向量，返回两个ector
void split(const std::vector<Complex>& A, std::vector<Complex>& even, std::vector<Complex>& odd)
{
    size_t n = A.size();
    even.clear();
    odd.clear();

    for (size_t i = 0; i < n; i+=2)
    {
        even.emplace_back(A[i]);
        odd.emplace_back(i+1 < n ? A[i+1] : 0);
    }
}

// 在原 vector 上拆分系数向量
void split(std::vector<Complex>::iterator first, std::vector<Complex>::iterator last)
{
    size_t n = last - first;
    std::vector<Complex> temp{first, last}; // TODO:能否不使用临时变量
    for (size_t i = 0; i < n; i+=2)
    {
        first[i/2] = temp[i];
        first[i/2+n/2] = temp[i+1];
    }
}

// 循环拆分系数向量直到最底层
void splitAll(std::vector<Complex>& data)
{
    size_t n = data.size();

    for (size_t groupSize = n; groupSize > 2; groupSize = groupSize/2)
    {
        for (size_t i = 0; i < n; i+=groupSize)
            split(data.begin() + i, data.begin() + i + groupSize);
    }
}

// 递归方案
std::vector<Complex> transformForFft1(const std::vector<Complex>& in, int forward=1)
{
    size_t n = in.size();
    if (n <= 1)
        return in;

    std::vector<Complex> evenIn;
    std::vector<Complex> oddIn;
    split(in, evenIn, oddIn);

    auto evenOut = transformForFft1(evenIn, forward);
    auto oddOut = transformForFft1(oddIn, forward);

    std::vector<Complex> out(n);
    for (size_t k =0; k < n/2; k++)
    {
        auto w = omega(forward*k, n);
        out[k] = evenOut[k] + w * oddOut[k];
        out[k + n/2] = evenOut[k] - w * oddOut[k];
    }

    return out;
}

// 非递归方案，蝶形算法
std::vector<Complex> transformForFft2(const std::vector<Complex>& in, int forward=1)
{
    size_t n = in.size();
    auto out = in;
    splitAll(out);
    for (size_t groupSize = 2; groupSize <= n; groupSize *= 2)
    {
        // 分组计算
        //    0 1 2 3       0=0+1 1=2+3 2=0-1 3=2-3
        //  0 2  |  1 3     0=0+2 2=0-2 1=1+3 3=1-3
        // 0 | 2 | 1 | 3    0=0 2=2 1=1 3=3
        auto temp = out; // TODO:能否不使用临时变量
        for (size_t k = 0; k + groupSize/2 < n; k+=1)
        {
            size_t groupIndex = k % groupSize;

            // 右半边和左半边一起计算出，因此右半直接跳过
            if (groupIndex >= groupSize/2)
            {
                k += groupSize/2 - 1;
                continue;
            }

            auto w = omega(forward*groupIndex, groupSize);
            out[k] = temp[k] + w * temp[k + groupSize/2]; // 左半
            out[k + groupSize/2] = temp[k] - w * temp[k + groupSize/2]; // 右半
        }
    }

    return out;
}


std::vector<Complex> fft(const std::vector<Complex>& coefficients)
{
    size_t n = coefficients.size();

    // 长度不是2的整数次幂
    if ((n & (n-1)) != 0)
        return dft(coefficients);
    
    return transformForFft2(coefficients);
        
}

std::vector<Complex> ifft(const std::vector<Complex>& values)
{
    size_t n = values.size();
    // 长度不是2的整数次幂
    if ((n & (n-1)) != 0)
        return idft(values);

    auto out = transformForFft2(values, -1);
    for (auto& v : out)
    {
        v /= n;
    }

    return out;
}

// 长度扩展为 2^n，不足的部分补0
std::vector<Complex> expand(const std::vector<Complex>& in)
{
    size_t len = in.size();
    size_t n = static_cast<size_t>(std::ceil(std::log(len) / std::log(2)));
    auto out = in;
    out.resize(std::pow(2,n), 0);
    return out;
}

int main()
{
    std::vector<Complex> coefficients(N); // 系数向量

    for (size_t i = 0; i < N; i++)
    {
        coefficients[i].real(i);
        coefficients[i].imag(i);
    }

    auto values = fft(coefficients);
    for (auto& v : values)
    {
        printf("%f + %fi\n", v.real(), v.imag());
    }
    printf("\n");

    coefficients = ifft(values);

    for (auto& v : coefficients)
    {
        printf("%f + %fi\n", v.real(), v.imag());
    }

    return 0;
}