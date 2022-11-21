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
    return std::complex{std::cos(angle), -std::sin(angle)};
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

// 拆分系数向量
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

std::vector<Complex> transformForFft(const std::vector<Complex>& in, int forward=1)
{
    size_t n = in.size();
    if (n <= 1)
        return in;

    std::vector<Complex> evenIn;
    std::vector<Complex> oddIn;
    split(in, evenIn, oddIn);

    auto evenOut = transformForFft(evenIn, forward);
    auto oddOut = transformForFft(oddIn, forward);

    std::vector<Complex> out(n);
    for (size_t k =0; k < n/2; k++)
    {
        out[k] = evenOut[k] + omega(forward*k, n) * oddOut[k];
        out[k + n/2] = evenOut[k] - omega(forward*k, n) * oddOut[k];
    }

    return out;
}

std::vector<Complex> fft(const std::vector<Complex>& coefficients)
{
    return transformForFft(coefficients);
}

std::vector<Complex> ifft(const std::vector<Complex>& values)
{
    size_t n = values.size();
    auto out = transformForFft(values, -1);
    for (auto& v : out)
    {
        v /= n;
    }

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
    for (size_t i = 0; i < N; i++)
    {
        printf("%f + %fi\n", values[i].real(), values[i].imag());
    }

    coefficients = ifft(values);

    for (size_t i = 0; i < N; i++)
    {
        printf("%f + %fi\n", coefficients[i].real(), coefficients[i].imag());
    }

    std::vector<Complex> odd;
    std::vector<Complex> even;
    split(coefficients, odd, even);

    // printf("%zu %zu\n", odd.size(), even.size());
}