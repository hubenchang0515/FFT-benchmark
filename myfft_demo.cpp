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

// 多项式求值
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

// k0 + k1*x + k2*x^2 + k3 * x^3 + ... + k<n-1> * x^<n-1>
std::vector<Complex> dft(const std::vector<Complex>& coefficients)
{
    size_t n = coefficients.size();
    std::vector<Complex> values(n);
    for(size_t i = 0; i < n; i++)
    {
        values[i] = polynomialValue(coefficients, omega(i, n));
    }
    return values;
}


std::vector<Complex> idft(const std::vector<Complex>& values)
{
    size_t n = values.size();
    std::vector<Complex> coefficients(n);
    for(size_t i = 0; i < n; i++)
    {
        coefficients[i] = polynomialValue(values, omega(-i, n)) / static_cast<double>(n);
    }
    return coefficients;
}

// std::vector<Complex> fft(const std::vector<Complex>& coefficients)
// {
//     size_t n = coefficients.size();
//     if (n == 1)
//         return {coefficients[0]};

//     std::vector<Complex> values(n);
//     for(size_t i = 0; i < n; i++)
//     {
//         std::vector<Complex> even;
//         std::vector<Complex> odd;
//         split(coefficients, even, odd);
//     }
//     return values;
// }


int main()
{
    std::vector<Complex> coefficients(N); // 系数向量

    for (size_t i = 0; i < N; i++)
    {
        coefficients[i].real(i);
        coefficients[i].imag(i);
    }


    auto values = dft(coefficients);
    for (size_t i = 0; i < N; i++)
    {
        printf("%f + %fi\n", values[i].real(), values[i].imag());
    }

    coefficients = idft(values);

    for (size_t i = 0; i < N; i++)
    {
        printf("%f + %fi\n", coefficients[i].real(), coefficients[i].imag());
    }

    std::vector<Complex> odd;
    std::vector<Complex> even;
    split(coefficients, odd, even);

    // printf("%zu %zu\n", odd.size(), even.size());
}