# Generated by [MakeMake](https://github.com/hubenchang0515/makemake)

.PHONY: all install uninstall clean

all: fftw_demo cufftw_demo myfft_demo

install: all

uninstall:

clean:
	 rm -f fftw_demo.o
	 rm -f cufftw_demo.o
	 rm -f myfft_demo.o

fftw_demo : fftw_demo.o  
	g++ -o $@ $^ -lfftw3 

fftw_demo.o: fftw_demo.cpp config.h
	g++ -c  fftw_demo.cpp -O2 -W -Wall -Wextra 

cufftw_demo : cufftw_demo.o  
	nvcc -o $@ $^ -lcufftw 

cufftw_demo.o : cufftw_demo.cpp \
    /usr/local/cuda-11.8/bin/../targets/x86_64-linux/include/cufftw.h \
    /usr/local/cuda-11.8/bin/../targets/x86_64-linux/include/cufft.h \
    /usr/local/cuda-11.8/bin/../targets/x86_64-linux/include/cuComplex.h \
    /usr/local/cuda-11.8/bin/../targets/x86_64-linux/include/vector_types.h \
    /usr/local/cuda-11.8/bin/../targets/x86_64-linux/include/crt/host_defines.h \
    /usr/local/cuda-11.8/bin/../targets/x86_64-linux/include/driver_types.h \
    /usr/local/cuda-11.8/bin/../targets/x86_64-linux/include/library_types.h \
    config.h
	nvcc -c  cufftw_demo.cpp -O2 

myfft_demo : myfft_demo.o  
	g++ -o $@ $^  

myfft_demo.o: myfft_demo.cpp config.h
	g++ -c  myfft_demo.cpp -O2 -W -Wall -Wextra 
