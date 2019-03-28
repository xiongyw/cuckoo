// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2016 John Tromp

#include "lean.hpp"
#include <unistd.h>

#define MAXSOLS 8
// arbitrary length of header hashed into siphash key
#ifndef HEADERLEN
#define HEADERLEN 80
#endif


int main(int argc, char **argv) {
  int nthreads = 1;
  int ntrims   = 2 + (PART_BITS+3)*(PART_BITS+4);
  int nonce = 0;
  int range = 1;
  char header[HEADERLEN];
  unsigned len;
  u64 time0, time1;
  u32 timems;
  int c;

  memset(header, 0, sizeof(header));
  while ((c = getopt (argc, argv, "h:m:n:r:t:x:")) != -1) {
    switch (c) {
      case 'h':
        len = strlen(optarg);
        assert(len <= sizeof(header));
        memcpy(header, optarg, len);
        break;
      case 'x':
        len = strlen(optarg)/2;
        assert(len <= sizeof(header));
        for (u32 i=0; i<len; i++)
          sscanf(optarg+2*i, "%2hhx", header+i);
        break;
      case 'n':
        nonce = atoi(optarg);
        break;
      case 'r':
        range = atoi(optarg);
        break;
      case 'm':
        ntrims = atoi(optarg);
        break;
      case 't':
        nthreads = atoi(optarg);
        break;
    }
  }
  printf("Looking for %d-cycle on cuckoo%d(header=\"%s\",nonce=%d", PROOFSIZE, EDGEBITS+1, header, nonce);
  if (range > 1)
    printf("-%d", nonce+range-1);
  printf(") with 50%% edges, %d trims, %d threads\n", ntrims, nthreads);

  /*
     const u32 PART_MASK = (1 << PART_BITS) - 1;
     const u64 ONCE_BITS = NEDGES >> PART_BITS;
     const u64 TWICE_BYTES = (2 * ONCE_BITS) / 8;
     const u64 TWICE_ATOMS = TWICE_BYTES / sizeof(atwice);
     const u32 TWICE_PER_ATOM = sizeof(atwice) * 4;
   */
  printf("PART_MASK     =%08x\n", PART_MASK);
  printf("ONCE_BITS     =%016x\n", ONCE_BITS);
  printf("TWICE_BYTES   =%016x\n", TWICE_BYTES);
  printf("sizeof(atwice)=%d\n", sizeof(atwice));
  printf("TWICE_ATOMS   =%016x\n", TWICE_BYTES);
  printf("TWICE_PER_ATOM=%08x\n", TWICE_PER_ATOM);

  u64 edgeBytes = NEDGES/8; // edge bitmap: one bit per edge
  u64 nodeBytes = TWICE_ATOMS*sizeof(atwice);  // counter storage for one partition
  int edgeUnit, nodeUnit;
  for (edgeUnit=0; edgeBytes >= 1024; edgeBytes>>=10,edgeUnit++) ;
  for (nodeUnit=0; nodeBytes >= 1024; nodeBytes>>=10,nodeUnit++) ;
  printf("Using %d%cB edge and %d%cB node memory, %d-way siphash, and %d-byte counters (2-bit per node)\n",
     (int)edgeBytes, " KMGT"[edgeUnit], (int)nodeBytes, " KMGT"[nodeUnit], NSIPHASH, SIZEOF_TWICE_ATOM);

  thread_ctx *threads = (thread_ctx *)calloc(nthreads, sizeof(thread_ctx));
  assert(threads);
  cuckoo_ctx ctx(nthreads, ntrims, MAXSOLS); // all threads share the same cuckoo_ctx

  u32 sumnsols = 0;
  for (int r = 0; r < range; r++) {
    time0 = timestamp();
    ctx.setheadernonce(header, sizeof(header), nonce + r);
    ctx.barry.clear();

    // spawn all threads, each starts working immediately
    for (int t = 0; t < nthreads; t++) {
      threads[t].id = t;
      threads[t].ctx = &ctx;
      int err = pthread_create(&threads[t].thread, NULL, worker, (void *)&threads[t]);
      assert(err == 0);
    }

    // sleep(33); ctx.abort();

    for (int t = 0; t < nthreads; t++) {
      int err = pthread_join(threads[t].thread, NULL);
      assert(err == 0);
    }

    time1 = timestamp(); timems = (time1 - time0) / 1000000;
    printf("Time: %d ms\n", timems);
    for (unsigned s = 0; s < ctx.nsols; s++) {
      printf("Solution");
      for (int i = 0; i < PROOFSIZE; i++)
        printf(" %jx", (uintmax_t)ctx.sols[s][i]);
      printf("\n");
    }
    sumnsols += ctx.nsols;
  }
  free(threads);
  printf("%d total solutions\n", sumnsols);
  return 0;
}
