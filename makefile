CFLAGS=

all: fastblur cudablur2 cudablur3

cudablur3: cudablur3.o
	nvcc $(CFLAGS) cudablur3.o -o cudablur3 -lm

cudablur2: cudablur2.o
	nvcc $(CFLAGS) cudablur2.o -o cudablur2 -lm

fastblur: fastblur.o
	gcc $(CFLAGS) fastblur.o -o fastblur -lm

cudablur3.o: cudablur3.cu
	nvcc -c $(CFLAGS) cudablur3.cu -o cudablur3.o

cudablur2.o: cudablur2.cu
	nvcc -c $(CFLAGS) cudablur2.cu -o cudablur2.o

fastblur.o: fastblur.c
	gcc -c $(CFLAGS) fastblur.c -o fastblur.o

clean:
	rm -f cudablur2.o fastblur.o cudablur3.o fastblur cudablur2 cudablur3 output.png
