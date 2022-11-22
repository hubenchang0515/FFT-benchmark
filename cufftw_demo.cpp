#include <cufftw.h>
#include "config.h"

int main()
{
    fftw_complex* in = (fftw_complex*)fftwf_malloc(N * sizeof(fftw_complex));
    fftw_complex* out = (fftw_complex*)fftwf_malloc(N * sizeof(fftw_complex));

    for (size_t i = 0; i < N; i++)
    {
        in[i][0] = i;
        in[i][1] = i;
    }

    // 正变换
    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    for (size_t i = 0; i < N; i++)
    {
        printf("%f+%fi\n", out[i][0], out[i][1]);
    }

    // 逆变换
    plan = fftw_plan_dft_1d(N, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    for (size_t i = 0; i < N; i++)
    {
        printf("%f+%fi\n", in[i][0]/N, in[i][1]/N);
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    
    return 0;
}