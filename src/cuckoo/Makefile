.POSIX:
.SUFFIXES:

OPT ?= -O2
#OPT ?= -g

GCC_ARCH_FLAGS ?= -march=native
GPP_ARCH_FLAGS ?= -march=native

# -Wno-deprecated-declarations shuts up Apple OSX clang
FLAGS ?= -Wall -Wno-format -Wno-deprecated-declarations -D_POSIX_C_SOURCE=200112L $(OPT) -DPREFETCH -I. $(CPPFLAGS) -pthread
GPP ?= g++ $(GPP_ARCH_FLAGS) -std=c++11 $(FLAGS)
CFLAGS ?= -Wall -Wno-format -fomit-frame-pointer $(OPT)
GCC ?= gcc $(GCC_ARCH_FLAGS) -std=gnu11 $(CFLAGS)
BLAKE_2B_SRC ?= ../crypto/blake2b-ref.c
SIPHASH_SRC = ../crypto/siphash.cpp

NVCC ?= nvcc -std=c++11 

#all : simpletest leantest meantest

simpletest:     simple19
	./simple19 -n 38

leantest:       lean19x1
	./lean19x1 -n 38

meantest:	mean29x4
	./mean29x4 -n 63 -t 4 -s

verifytest:     lean19x1 verify19
	./lean19x1 -n 38 | grep ^Sol | ./verify19 -n 38

verify19:	../crypto/siphash.hpp cuckoo.h verify.c Makefile
	$(GPP) -o $@ -DPROOFSIZE=42 -DEDGEBITS=19 verify.c $(BLAKE_2B_SRC)

# simple

simple19:	../crypto/siphash.hpp cuckoo.h cyclebase.cpp cyclebase.hpp simple.cpp Makefile
	$(GPP) -o $@ -DIDXSHIFT=0 -DPROOFSIZE=42 -DEDGEBITS=19 simple.cpp cyclebase.cpp cuckoo.c $(BLAKE_2B_SRC) $(SIPHASH_SRC)

simple29:	../crypto/siphash.hpp cuckoo.h cyclebase.hpp simple.cpp Makefile
	$(GPP) -o $@ -DIDXSHIFT=0 -DPROOFSIZE=42 -DEDGEBITS=29 simple.cpp cyclebase.cpp cuckoo.c $(BLAKE_2B_SRC) $(SIPHASH_SRC)

# mean

mean19x8:	cuckoo.h ../crypto/siphash.hpp mean.hpp mean.cpp Makefile
	$(GPP) -o $@ -mavx2 -DXBITS=2 -DNSIPHASH=8 -DEDGEBITS=19 mean.cpp cuckoo.c $(BLAKE_2B_SRC) $(SIPHASH_SRC)

mean29x4:	cuckoo.h ../crypto/siphash.hpp mean.hpp mean.cpp Makefile
	$(GPP) -o $@ -mno-avx2 -DNSIPHASH=4 -DEDGEBITS=29 mean.cpp cuckoo.c $(BLAKE_2B_SRC) $(SIPHASH_SRC)

mean29x8:	cuckoo.h ../crypto/siphash.hpp mean.hpp mean.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DEDGEBITS=29 mean.cpp cuckoo.c $(BLAKE_2B_SRC) $(SIPHASH_SRC)

mean29x1:	cuckoo.h ../crypto/siphash.hpp mean.hpp mean.cpp Makefile
	$(GPP) -o $@ -DNSIPHASH=1 -DEDGEBITS=29 mean.cpp cuckoo.c $(BLAKE_2B_SRC) $(SIPHASH_SRC)

# lean

lean19x1:	../crypto/siphash.hpp cuckoo.h  lean.hpp lean.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DNSIPHASH=1 -DEDGEBITS=19 lean.cpp cuckoo.c $(BLAKE_2B_SRC) $(SIPHASH_SRC)

lean19x8:	../crypto/siphash.hpp cuckoo.h  lean.hpp lean.cpp Makefile
	$(GPP) -o $@ -DATOMIC -DNSIPHASH=8 -DEDGEBITS=19 lean.cpp cuckoo.c $(BLAKE_2B_SRC) $(SIPHASH_SRC)

lean29x8:	../crypto/siphash.hpp cuckoo.h  lean.hpp lean.cpp Makefile
	$(GPP) -o $@ -mavx2 -DNSIPHASH=8 -DATOMIC -DEDGEBITS=29 lean.cpp cuckoo.c $(BLAKE_2B_SRC) $(SIPHASH_SRC)


# cuda

cuda29:		../crypto/siphash.cuh mean.cu Makefile
	$(NVCC) -o $@ -DEDGEBITS=29 -arch sm_35 mean.cu cuckoo.c $(BLAKE_2B_SRC)

cuda29_1:		../crypto/siphash.cuh mean.cu Makefile
	$(NVCC) -o $@ -DNB=1 -DEDGEBITS=29 -arch sm_35 mean.cu $(BLAKE_2B_SRC)

lcuda29:	../crypto/siphash.cuh lean.cu Makefile
	$(NVCC) -o $@ -DEDGEBITS=29 -arch sm_35 lean.cu $(BLAKE_2B_SRC)

