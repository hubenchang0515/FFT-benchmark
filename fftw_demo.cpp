#include <fftw3.h>
#include "config.h"

int main()
{
    fftw_complex* in = fftw_alloc_complex(N);
    fftw_complex* out = fftw_alloc_complex(N);

    fftw_plan plan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_MEASURE);

    for (size_t i = 0; i < N; i++)
    {
        in[i][0] = i;
        in[i][1] = i;
    }

    fftw_execute(plan);

    for (size_t i = 0; i < N; i++)
    {
        printf("%f+%fi\n", out[i][0], out[i][1]);
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}