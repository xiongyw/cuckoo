// Cuckaroo Cycle, a memory-hard proof-of-work by John Tromp
// Copyright (c) 2018-2019 Jiri Vadura (photon) and John Tromp
// This software is covered by the FAIR MINING license

#include <stdio.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include "cuckaroo.hpp"
#include "graph.hpp"
#include "../crypto/siphash.cuh"
#include "../crypto/blake2.h"

#define ROO_VERBOSE            1

#define SEEDA_CUSTOM_BLOCK     0   // run a custom number of blocks for SeedA()
#define SEEDA_BLOCKS          64

#define ROUND0_CUSTOM_BLOCK    0   // run a custom number of blocks for Round(0)
#define ROUND0_BLOCKS         64

#define ROUNDN_CUSTOM_BLOCK    0   // run a custom number of blocks for Round(1~175)
#define ROUNDN_BLOCKS         64

#define NULL_SIPKEYS           0   // force set sipkeys to 0
#define TEST_FLUSHA            0   // show warning when tmp[][] buffer is not big enough causing edges lost
#define TEST_FLUSHB            0   // show warning when tmp[][] buffer is not big enough causing edges lost
#define DOUBLE_FLUSHA          1   // FLUSHA=32
#define DOUBLE_FLUSHB          1   // FLUSHB=16

#define DUMP_SEEDA             0   // dump bucket/index content after SeedA
#define DUMP_SEEDB             0   // dump bucket/index content after SeedB
#define DUMP_ROUND0            0   // dump bucket/index content after Round(0)
#define DUMP_ROUND1            0   // dump bucket/index content after Round(1)
#define DUMP_ROUND175          0   // dump bucket/index content after Round()
#define DUMP_TAIL              0   // dump bufferB content after Tail()

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint64_t u64; // save some typing

#ifndef MAXSOLS
#define MAXSOLS 4
#endif

#ifndef IDXSHIFT
// number of bits of compression of surviving edge endpoints
// reduces space used in cycle finding, but too high a value
// results in NODE OVERFLOW warnings and fake cycles
#define IDXSHIFT 12
#endif

const u32 MAXEDGES = NEDGES >> IDXSHIFT;

#ifndef XBITS
#define XBITS 6
#endif

#define NODEBITS (EDGEBITS + 1)

const u32 NX        = 1 << XBITS;
const u32 NX2       = NX * NX;
const u32 XMASK     = NX - 1;
const u32 YBITS     = XBITS;
const u32 NY        = 1 << YBITS;
const u32 YZBITS    = EDGEBITS - XBITS;
const u32 ZBITS     = YZBITS - YBITS;
const u32 NZ        = 1 << ZBITS;
const u32 ZMASK     = NZ - 1;

#ifndef NEPS_A
#define NEPS_A 133
#endif
#ifndef NEPS_B
#define NEPS_B 88
#endif
#define NEPS 128

const u32 EDGES_A = NZ * NEPS_A / NEPS;
const u32 EDGES_B = NZ * NEPS_B / NEPS;

const u32 ROW_EDGES_A = EDGES_A * NY;
const u32 ROW_EDGES_B = EDGES_B * NY;

// Number of Parts of BufferB, all but one of which will overlap BufferA
#ifndef NB
#define NB 2
#endif

#ifndef NA
#define NA  ((NB * NEPS_A + NEPS_B-1) / NEPS_B)
#endif

__constant__ uint2 recoveredges[PROOFSIZE];
__constant__ uint2 e0 = {0,0};

/*
 * return the hashes for the last edge of the edge-block, and all the hashes in buf[]
 * are not XORed with the last one yet, so it will be done out of the scope of this
 * function. but why?
 */
__device__ u64 dipblock(const siphash_keys &keys, const word_t edge, u64 *buf) {
    diphash_state shs(keys);
    word_t edge0 = edge & ~EDGE_BLOCK_MASK;
    u32 i;
    for (i=0; i < EDGE_BLOCK_MASK; i++) {
        shs.hash24(edge0 + i);
        buf[i] = shs.xor_lanes();
    }
    shs.hash24(edge0 + i);
    buf[i] = 0;  // to be XORed with the last value to get the last value
    return shs.xor_lanes();
}

__device__ u32 endpoint(uint2 nodes, int uorv) {
    return uorv ? nodes.y : nodes.x;
}

#ifndef FLUSHA // should perhaps be in trimparams and passed as template parameter
#if DOUBLE_FLUSHA
#define FLUSHA 32
#else
#define FLUSHA 16  /* for SeedA, number of edges to be batch-moved from shared mem to global buf */
#endif
#endif

/*
 * Seeding is divided into two steps:
 * 1. SeedA(): put all edge(u,v) into buckets[row=uX][col=gpu-block-idx/NX].
 * 2. SeedB(): sort each row of buckets that edge(u,v} is in buckets[uX][uY]. or, if treat buckets matrix
 *    as an one-dimentional array, to put edge(u,v) into buckets[uXY].
 *
 * - each cell is a bucket of size maxOut
 * - in global buf, cells are ordered row by row (top to bottom); and within each row, from left to right
 *
 *                        \ | 0 | 1 | 2 |...| 63|
 *                       ---+---+---+---+---+---|
 *                        0 |   |   |   |   |   |
 *                       ---+---+---+---+   +---|
 *                        1 |   |   |   |   |   |
 *                       ---+---+---+---+   +---|
 *                       ...|          .....    |
 *                       ---+---+---+---+   +---|
 *                       63 |   |   |   |   |   |
 *                       -----------------------+
 *
 * 1. for edge-generation, all edges are divided by all threads according to their global idx,
 *    i.e., edges-per-thread(ept)=NEDGES/nthreads. For instance, if each thread handles 2
 *    edge-blocks (each edge-block contains 64 edges):
 *      edge-block idx:    0 1 2 3 4 5 ...
 *      thread global idx: 0 0 1 1 2 2 ...
 *    THREADS_HAVE_EDGES is to ensure that each thread at least have one edge-block to process.
 * 2. a thread puts edge(u,v) into buckets[uX][group%NX], where group is the gpu-block idx of the thread. it's assumed
 *    that genA.blocks a multiple of NX, such that edges is evenly distributed into a row of buckets. Ex, for
 *    cuckaroo29, genA.blocks=4096. it's possible that multiple blocks write to same bucket column, in this case, global
 *    indexesE[] with atomicAdd() is used for synchronization among blocks.
 */
template<int maxOut>  /* when invoked, maxOut is EDGEDS_A, which is the bucket size in genA */
__global__ void SeedA(const siphash_keys &sipkeys, ulonglong4 * __restrict__ buffer, u32 * __restrict__ indexes) {
    const int group = blockIdx.x;         // a block is a "group" of threads
    const int dim = blockDim.x;           // tpb (thread-per-block)
    const int lid = threadIdx.x;          // current thread's local idx in its block
    const int gid = group * dim + lid;    // current thread's global idx
    const int nthreads = gridDim.x * dim; // total number of threads
    const int FLUSHA2 = 2*FLUSHA;         // why dobule? search ROWS_LIMIT_LOSSES for reasoning on this.

#if ROO_VERBOSE
    if (group== 0 && lid == 0)
        printf("SeedA(): maxOut=%d, gridDim.x=%d, blockDim.x=%d\n", maxOut, gridDim.x, blockDim.x);
#endif

#if 0
    printf("SeedA(): group=%4d, dim=%3d, lid=%3d, gid=%8d, nthreads=%d, FLUSHA=%d\n",
            group, dim, lid, gid, nthreads, FLUSHA);
#endif

    /*
     * __shared__ is for intra-block threads communication
     */
    __shared__ uint2 tmp[NX][FLUSHA2]; // needs to be ulonglong4 aligned. 16KB
    __shared__ int counters[NX];  /* one counter per row; 256B */
    const int TMPPERLL4 = sizeof(ulonglong4) / sizeof(uint2);  /* how many tmp elements to fit one ulonglong4 */
    /* all automatic variables other than arrays are stored in registers; arrays are in local memory, which is actually global mem */
    u64 buf[EDGE_BLOCK_SIZE]; /* to hold one edge-block hash result */

    /*
     * reset some elements in the shared counters[] corresponding to the block, in parallel.
     * - in case tpb < NX, one thread resets multiple counters
     * - in case tpb > NX, only the first NX threads of the block do the counters resetting
     */
    for (int row = lid; row < NX; row += dim) {
        counters[row] = 0;
    }

    __syncthreads(); /* barrier to sync threads within a block */

    /* the column for the current block to write to */
    const int col = group % NX;
    /* edges per thread */
#if SEEDA_CUSTOM_BLOCK
    const int loops = 512; // assuming THREADS_HAVE_EDGES checked
#else
    const int loops = NEDGES / nthreads; // assuming THREADS_HAVE_EDGES checked
#endif

    for (int blk = 0; blk < loops; blk += EDGE_BLOCK_SIZE) { // blk is count of hashed edges of the current thread

        /*
         * this implies that all threads divide the edge space sequentially according to their global thread id (i.e., cross gpu block)
         */
        u32 nonce0 = gid * loops + blk;

        const u64 last = dipblock(sipkeys, nonce0, buf);
        for (u32 e = 0; e < EDGE_BLOCK_SIZE; e++) {
            u64 edge = buf[e] ^ last;  // finialize the siphash() to get the two nodes of an edge
            u32 node0 = edge & EDGEMASK;          // u
            u32 node1 = (edge >> 32) & EDGEMASK;  // v
#if SEEDA_CUSTOM_BLOCK
            if (node1 == 0x6a32577) printf("gotit: nonce=%d: u=0x%08x, v=%08x\n", nonce0, node0, node1);
            // print some edges to verify correctness of dipblock()
            if (gid == 0 && blk == 0 && e < EDGE_BLOCK_SIZE) {
                //printf("micro-nonce=%d, last_lo=0x%08x, last_hi=0x%08x\n", nonce0, (uint32_t)last, (uint32_t)(last >> 32));
                //printf("micro-nonce=%d, e=%d, buf[%d]={lo=%08x, hi=0x%08x}\n", nonce0, e, e, (uint32_t)buf[e], (uint32_t)(buf[e]>>32));
                printf("micro-nonce=%d, e=%2d: u=0x%08x, v=0x%08x\n", nonce0, e, node0, node1);
            }
#endif
            int row = node0 >> YZBITS;            // uX
            /*
             * atomicAdd(): when a thread executes this operation, a memory address is read, has the value of ‘val’ added
             * to it, and the result is written back to memory. The original value of the memory at location ‘address’
             * is returned to the thread.
             */
#if TEST_FLUSHA
            int counter0 = (int)atomicAdd(counters + row, 1);
            if (counter0 >= FLUSHA2) {
                printf("################ SeedA(): possible edge lost. counter0=%d, FLUSHA2=%d\n", counter0, FLUSHA2);
            }
            int counter = min(counter0, (int)(FLUSHA2-1)); // assuming ROWS_LIMIT_LOSSES checked
#else
            int counter = min((int)atomicAdd(counters + row, 1), (int)(FLUSHA2-1)); // assuming ROWS_LIMIT_LOSSES checked
#endif
            tmp[row][counter] = make_uint2(node0, node1);
            __syncthreads();

            /*
             * for a particular row, only one thread in the block will execute the following, since the value of counter (for the row)
             * is different from other threads in the block
             */
            if (counter == FLUSHA-1) { /* -1: atomicAdd() returns the value before add */
                int localIdx = min(FLUSHA2, counters[row]); // at this point, `counters[row]` may be bigger than `counter+1`.
                int newCount = localIdx % FLUSHA;
                int nflush = localIdx - newCount;
                u32 grp = row * NX + col; // bucket idx. it implies the buckets in buffer is aranged row by row.
                int cnt = min((int)atomicAdd(indexes + grp, nflush), (int)(maxOut - nflush)); // cnt is the current index of the dst bucket
#if SEEDA_CUSTOM_BLOCK
                if (grp == 0) {
                  printf("gid=%3d: cnt=0x%08x, nflush=%d, counter=%d, newCount=%d\n", gid, cnt, nflush, counters[row], newCount);
                  for (int k = 0; k < counters[row]; k ++) {
                      printf("%2d: 0x%08x, 0x%08x\n", k, tmp[row][k].x, tmp[row][k].y);
                  }
                }
#endif
                /* move from share mem to global buf */
                for (int i = 0; i < nflush; i += TMPPERLL4) {
                    buffer[((u64)grp * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *)(&tmp[row][i]);
                }
                /* update share mem: tmp[row][] and counters[row] */
                for (int t = 0; t < newCount; t++) {
                    tmp[row][t] = tmp[row][t + nflush];
                }
                counters[row] = newCount;
            }
            __syncthreads();
        }
    }

    /*
     * flush the remaining edges in shared mem tmp[][], if there is any.
     * only part of the threads in the block execute the following
     */
    uint2 zero = make_uint2(0, 0);
    for (int row = lid; row < NX; row += dim) {
        int localIdx = min(FLUSHA2, counters[row]);
        u32 grp = row * NX + col;

        /*
         * zero out some irrelevant edges at the end of TMPPERLL4 edges, since the flush unit is TMPPERLL4 edges;
         * the consequence is that in the end of the bucket, some of the edges may be null-edges, that's why
         * in SeedB(), a null-edge check is employed.
         */
        for (int j = localIdx; j % TMPPERLL4; j++) {
            tmp[row][j] = zero;
        }

        // flush
        for (int i = 0; i < localIdx; i += TMPPERLL4) {
            int cnt = min((int)atomicAdd(indexes + grp, TMPPERLL4), (int)(maxOut - TMPPERLL4));
            buffer[((u64)grp * maxOut + cnt) / TMPPERLL4] = *(ulonglong4 *)(&tmp[row][i]);
        }
    }
}

template <typename Edge> __device__ bool null(Edge e);

__device__ bool null(u32 nonce) {
    return nonce == 0;
}

__device__ bool null(uint2 nodes) {
    return nodes.x == 0 && nodes.y == 0;
}

#ifndef FLUSHB
#if DOUBLE_FLUSHB
#define FLUSHB 16
#else
#define FLUSHB 8
#endif
#endif

/*
 * goal: put edge(u,v) into buckets[uX][uY], based on the fact that edge(u,v) is in buckets[uX][?].
 *
 * - num of cuda-blocks equal to num of buckets (4096), such that each cuda-block processes a bucket,
 *   i.e., cuda-block i process buckets[i]. buckets[i] is in row i/NX.
 */
template<int maxOut> /* EDGES_A: max bucket size for both src and dst */
__global__ void SeedB(const uint2 * __restrict__ source, ulonglong4 * __restrict__ destination, const u32 * __restrict__ srcIdx, u32 * __restrict__ dstIdx) {
    const int group = blockIdx.x;
    const int dim = blockDim.x;
    const int lid = threadIdx.x;
    const int FLUSHB2 = 2 * FLUSHB;

    __shared__ uint2 tmp[NX][FLUSHB2];
    const int TMPPERLL4 = sizeof(ulonglong4) / sizeof(uint2);
    __shared__ int counters[NX];

#if ROO_VERBOSE
    if (group== 0 && lid == 0)
        printf("SeedB(): maxOut=%d, gridDim.x=%d, blockDim.x=%d\n", maxOut, gridDim.x, blockDim.x);
#endif

    for (int col = lid; col < NX; col += dim)
        counters[col] = 0;
    __syncthreads();

    /* the row of the bucket processed by current block */
    const int row = group / NX;
    /* num of edges in the current bucket */
    const int bucketEdges = min((int)srcIdx[group], (int)maxOut);
    /* max num of edges processed by each thread in the block. threads's edges are interleaved for locality */
    const int loops = (bucketEdges + dim-1) / dim;
    for (int loop = 0; loop < loops; loop++) {
        int col;
        int counter = 0;
        const int edgeIndex = loop * dim + lid;
        if (edgeIndex < bucketEdges) {
            const int index = group * maxOut + edgeIndex;
            uint2 edge = __ldg(&source[index]);  // cuda intrinsic, read-only LoaD from Global memory. why it's faster?
            if (!null(edge)) {
                u32 node1 = edge.x; // u
                col = (node1 >> ZBITS) & XMASK; // uY

#if TEST_FLUSHB
                int counter0 = (int)atomicAdd(counters + col, 1); 
                if (counter0 >= (FLUSHB2)) {
                  printf("################ SeedB(): possible edge lost. counter0=%d, FLUSHB2=%d\n", counter0, FLUSHB2);
                }
                counter = min(counter0, (int)(FLUSHB2-1)); // assuming COLS_LIMIT_LOSSES checked
#else
                counter = min((int)atomicAdd(counters + col, 1), (int)(FLUSHB2-1)); // assuming COLS_LIMIT_LOSSES checked
#endif
                tmp[col][counter] = edge;
            }
        }
        __syncthreads();
        if (counter == FLUSHB-1) {
            int localIdx = min(FLUSHB2, counters[col]);
            int newCount = localIdx % FLUSHB;
            int nflush = localIdx - newCount;
            u32 grp = row * NX + col;
#ifdef SYNCBUG  // what's this?
            if (grp==0x2d6) printf("group %x size %d lid %d nflush %d\n", group, bucketEdges, lid, nflush);
#endif
            int cnt = min((int)atomicAdd(dstIdx + grp, nflush), (int)(maxOut - nflush));
            for (int i = 0; i < nflush; i += TMPPERLL4)
                destination[((u64)grp * maxOut + cnt + i) / TMPPERLL4] = *(ulonglong4 *)(&tmp[col][i]);
            for (int t = 0; t < newCount; t++) {
                tmp[col][t] = tmp[col][t + nflush];
            }
            counters[col] = newCount;
        }
        __syncthreads(); 
    }
    uint2 zero = make_uint2(0, 0);
    for (int col = lid; col < NX; col += dim) {
        int localIdx = min(FLUSHB2, counters[col]);
        u32 grp = row * NX + col;
#ifdef SYNCBUG
        if (group==0x2f2 && grp==0x2d6) printf("group %x size %d lid %d localIdx %d\n", group, bucketEdges, lid, localIdx);
#endif
        for (int j = localIdx; j % TMPPERLL4; j++)
            tmp[col][j] = zero;
        for (int i = 0; i < localIdx; i += TMPPERLL4) {
            int cnt = min((int)atomicAdd(dstIdx + grp, TMPPERLL4), (int)(maxOut - TMPPERLL4));
            destination[((u64)grp * maxOut + cnt) / TMPPERLL4] = *(ulonglong4 *)(&tmp[col][i]);
        }
    }
}

/*
 * edge-counters for a bucket: 2-bit * NZ. so the num of 32-bit words for a counter array is 2*NZ/32=NZ/16.
 * the 1st bit is stored in the 1st NZ/32 words, and the 2nd bits stored in the 2nd NZ/32 words.
 * so each 32-bit word stores 1st bit (or 2nd bit) of edges' counters, use `>>5` get the word offset.
 * - when write, both bits may be updated
 * - when read, only need to read the 2nd bit
 */
__device__ __forceinline__  void Increase2bCounter(u32 *ecounters, const int bucket) {
    int word = bucket >> 5;
    unsigned char bit = bucket & 0x1F;
    u32 mask = 1 << bit;

    u32 old = atomicOr(ecounters + word, mask) & mask;
    if (old)
        atomicOr(ecounters + word + NZ/32, mask);
}

__device__ __forceinline__  bool Read2bCounter(u32 *ecounters, const int bucket) {
    int word = bucket >> 5;
    unsigned char bit = bucket & 0x1F;

    return (ecounters[word + NZ/32] >> bit) & 1;
}

#if ROO_VERBOSE
__global__ void live_edges(int round, const u32* idx0, const u32* idx1) {
    const int group = blockIdx.x;
    const int dim = blockDim.x;
    const int lid = threadIdx.x;
    int gid = group * dim + lid;

    u32 nr_edges0 = 0;
    u32 nr_edges1 = 0;
    float pct0 = 0., pct1 = 0.;

    //if (round > 4) return;

    if (gid == 0) {
        for (u32 i = 0; i < NX2; i ++) {
            nr_edges0 += idx0[i];
            nr_edges1 += idx1[i];
        }
        pct0 = (float)nr_edges0 / NEDGES;
        pct1 = (float)nr_edges1 / NEDGES;
        if (round == 0) {
            printf("After round %3d, NEDGES=%08x, nr_edges=%08x (%.6f)\n", round, NEDGES, (nr_edges0 + nr_edges1), pct0 + pct1);
        } else {
            if (round & 1) {
                printf("After round %3d, NEDGES=%08x, nr_edges=%08x (%.6f)\n", round, NEDGES, nr_edges0, pct0);
            } else {
                printf("After round %3d, NEDGES=%08x, nr_edges=%08x (%.6f)\n", round, NEDGES, nr_edges1, pct1);
            }
        }
    }
}

__global__ void print_indexs(const u32* idx)
{
    const int group = blockIdx.x;
    const int dim = blockDim.x;
    const int lid = threadIdx.x;
    int gid = group * dim + lid;

    if (gid == 0) {
        for (int row = 0; row < NX; row ++) {
            printf("%2d: ", row);
            for (int col = 0; col < NX; col ++) {
                printf("%2d ", idx[row * NX + col]);
            }
            printf("\n");
        }
    }
}

__global__ void print_bucket(const u32* buffer, const u32 bucket_max_sz_word, const u32 bucket_row, const u32 bucket_col, const u32 nr_edges_to_print)
{
    const int group = blockIdx.x;
    const int dim = blockDim.x;
    const int lid = threadIdx.x;
    int gid = group * dim + lid;

    if (gid == 0) {
        u32 bucket_idx = bucket_row * NX + bucket_col;
        const u32* p = buffer + bucket_max_sz_word * bucket_idx;
        printf("bucket[row=%d][col=%d]:\n", bucket_row, bucket_col);
        for (int i = 0; i < nr_edges_to_print; i ++) {
            printf("%6d: 0x%08x, 0x%08x (uX=%d)\n", i, p[i*2], p[i*2+1], p[i*2]>>YZBITS);
        }
    }
}
#endif

/*
 * Round is for trimming, trim on src buffer and put the result in dst buffer.
 * NP: num of partitions of the source buffer.  each partition contains all buckets, the size of each bucket is partitioned. e.g.,
 *  when NP=2, there are two partitions of the source buffer:
 *                        \ | 0 | 1 | 2 |...| 63|
 *          partition 0: ---+---+---+---+---+---|
 *                        0 |   |   |   |   |   |
 *                       ---+---+---+---+   +---|
 *                        1 |   |   |   |   |   |
 *                       ---+---+---+---+   +---|
 *                       ...|          .....    |
 *                       ---+---+---+---+   +---|
 *                       63 |   |   |   |   |   |
 *         partition 1:  -----------------------+
 *                        0 |   |   |   |   |   |
 *                       ---+---+---+---+   +---|
 *                        1 |   |   |   |   |   |
 *                       ---+---+---+---+   +---|
 *                       ...|          .....    |
 *                       ---+---+---+---+   +---|
 *                       63 |   |   |   |   |   |
 *                       -----------------------+
 *
 * maxIn: max size of source bucket
 * maxOut: max size of dst bucket
 *
 * for each partition: one cuda-block processes one bucket (the src bucket size is maxIn/NP).
 */
template<int NP, int maxIn, int maxOut>
__global__ void Round(const int round, const uint2 * __restrict__ src, uint2 * __restrict__ dst, const u32 * __restrict__ srcIdx, u32 * __restrict__ dstIdx) {
    const int group = blockIdx.x;
    const int dim = blockDim.x;
    const int lid = threadIdx.x;
    const int COUNTERWORDS = NZ / 16; // 16 2-bit counters per 32-bit word

#if ROO_VERBOSE
    if (group== 0 && lid == 0)
        printf("Round(round=%d): NP=%d, maxIn=%d, maxOut=%d, gridDim.x=%d, blockDim.x=%d\n", round, NP, maxIn, maxOut, gridDim.x, blockDim.x);
#endif

    __shared__ u32 ecounters[COUNTERWORDS];

    for (int i = lid; i < COUNTERWORDS; i += dim)
        ecounters[i] = 0;
    __syncthreads();

    for (int i = 0; i < NP; i++, src += NX2 * maxIn, srcIdx += NX2) {
        const int edgesInBucket = min(srcIdx[group], maxIn);
        // if (!group && !lid) printf("round %d size  %d\n", round, edgesInBucket);
        const int loops = (edgesInBucket + dim-1) / dim;

        for (int loop = 0; loop < loops; loop++) {
            const int lindex = loop * dim + lid;
            if (lindex < edgesInBucket) {
                const int index = maxIn * group + lindex;
                uint2 edge = __ldg(&src[index]);
                if (null(edge)) continue;
                u32 node = endpoint(edge, round&1);
                Increase2bCounter(ecounters, node & ZMASK);
            }
        }
    }

    __syncthreads();

    src -= NP * NX2 * maxIn; srcIdx -= NP * NX2;
    for (int i = 0; i < NP; i++, src += NX2 * maxIn, srcIdx += NX2) {
        const int edgesInBucket = min(srcIdx[group], maxIn);
        const int loops = (edgesInBucket + dim-1) / dim;
        for (int loop = 0; loop < loops; loop++) {
            const int lindex = loop * dim + lid;
            if (lindex < edgesInBucket) {
                const int index = maxIn * group + lindex;
                uint2 edge = __ldg(&src[index]);
                if (null(edge)) continue;
                u32 node0 = endpoint(edge, round&1);
                if (Read2bCounter(ecounters, node0 & ZMASK)) {
                    u32 node1 = endpoint(edge, (round&1)^1);
                    const int bucket = node1 >> ZBITS; /* XY, one dimensional buckets[] index */
                    const int bktIdx = min(atomicAdd(dstIdx + bucket, 1), maxOut - 1);
                    dst[bucket * maxOut + bktIdx] = (round&1) ? make_uint2(node1, node0) : make_uint2(node0, node1);
                }
            }
        }
    }
}

/*
 * pack all surviving edges in destination buffer, and store nedges in dstIdx[0].
 * a gpu-block handles a bucket; within the gpu-block, the 1st thread update the shared
 * dstIdx, and all threads copy one edge at a time from the bucket to the destination,
 * in an interleaved way exploiting spatial locality.
 */
template<int maxIn>
__global__ void Tail(const uint2 *source, uint2 *destination, const u32 *srcIdx, u32 *dstIdx) {
    const int lid = threadIdx.x;
    const int group = blockIdx.x;
    const int dim = blockDim.x; // tail.tpb=1024
    int myEdges = srcIdx[group];
    __shared__ int destIdx;

#if ROO_VERBOSE
    if (group== 0 && lid == 0)
        printf("Tail(): maxIn=%d, gridDim.x=%d, blockDim.x=%d\n", maxIn, gridDim.x, blockDim.x);
#endif

    if (lid == 0)
        destIdx = atomicAdd(dstIdx, myEdges);
    __syncthreads();
    for (int i = lid; i < myEdges; i += dim)
        destination[destIdx + lid] = source[group * maxIn + lid];
}

#define checkCudaErrors_V(ans) ({if (gpuAssert((ans), __FILE__, __LINE__) != cudaSuccess) return;})
#define checkCudaErrors_N(ans) ({if (gpuAssert((ans), __FILE__, __LINE__) != cudaSuccess) return NULL;})
#define checkCudaErrors(ans) ({int retval = gpuAssert((ans), __FILE__, __LINE__); if (retval != cudaSuccess) return retval;})

inline int gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    int device_id;
    cudaGetDevice(&device_id);
    if (code != cudaSuccess) {
        snprintf(LAST_ERROR_REASON, MAX_NAME_LEN, "Device %d GPUassert: %s %s %d", device_id, cudaGetErrorString(code), file, line);
        cudaDeviceReset();
        if (abort) return code;
    }
    return code;
}

__global__ void Recovery(const siphash_keys &sipkeys, ulonglong4 *buffer, int *indexes) {
    const int gid = blockDim.x * blockIdx.x + threadIdx.x;
    const int lid = threadIdx.x;
    const int nthreads = blockDim.x * gridDim.x;
    const int loops = NEDGES / nthreads;
    __shared__ u32 nonces[PROOFSIZE];
    u64 buf[EDGE_BLOCK_SIZE];

#if ROO_VERBOSE
    if (blockIdx.x == 0 && lid == 0)
        printf("Recovery(): gridDim.x=%d, blockDim.x=%d\n", gridDim.x, blockDim.x);
#endif

    if (lid < PROOFSIZE) nonces[lid] = 0;
    __syncthreads();
    for (int blk = 0; blk < loops; blk += EDGE_BLOCK_SIZE) {
        u32 nonce0 = gid * loops + blk;
        const u64 last = dipblock(sipkeys, nonce0, buf);
        for (int i = 0; i < EDGE_BLOCK_SIZE; i++) {
            u64 edge = buf[i] ^ last;
            u32 u = edge & EDGEMASK;
            u32 v = (edge >> 32) & EDGEMASK;
            for (int p = 0; p < PROOFSIZE; p++) { //YO
                if (recoveredges[p].x == u && recoveredges[p].y == v) {
                    nonces[p] = nonce0 + i;
                }
            }
        }
    }
    __syncthreads();
    if (lid < PROOFSIZE) {
        if (nonces[lid] > 0)
            indexes[lid] = nonces[lid];
    }
}

struct blockstpb {
    u16 blocks;
    u16 tpb;
};

struct trimparams {
    u16 ntrims;
    blockstpb genA;
    blockstpb genB;
    blockstpb trim;
    blockstpb tail;
    blockstpb recover;

    trimparams() {
        // note: tpb are kind of fixed, and blocks are calculated by `nthreads/tpb`: see `fill_default_params()`
        ntrims              =  176;
        genA.blocks         = 4096;
        genA.tpb            =  256;
        genB.blocks         =  NX2;
        genB.tpb            =  128;
        trim.blocks         =  NX2;
        trim.tpb            =  512;
        tail.blocks         =  NX2;
        tail.tpb            = 1024;
        recover.blocks      = 1024;
        recover.tpb         = 1024;
    }
};

typedef u32 proof[PROOFSIZE];

// maintains set of trimmable edges
struct edgetrimmer {
    trimparams tp;
    edgetrimmer *dt;
    size_t sizeA, sizeB;
    const size_t indexesSize = NX * NY * sizeof(u32);  // one index per bucket
    u8 *bufferA;
    u8 *bufferB;
    u8 *bufferAB;
    u32 *indexesE[1+NB];
    u32 nedges;
    u32 *uvnodes;
    siphash_keys sipkeys, *dipkeys;
    bool abort;
    bool initsuccess = false;

    edgetrimmer(const trimparams _tp) : tp(_tp) {
        checkCudaErrors_V(cudaMalloc((void**)&dt, sizeof(edgetrimmer)));
        checkCudaErrors_V(cudaMalloc((void**)&uvnodes, PROOFSIZE * 2 * sizeof(u32)));
        checkCudaErrors_V(cudaMalloc((void**)&dipkeys, sizeof(siphash_keys)));
        for (int i = 0; i < 1+NB; i++) {
            checkCudaErrors_V(cudaMalloc((void**)&indexesE[i], indexesSize));
        }
        sizeA = ROW_EDGES_A * NX * sizeof(uint2);
        sizeB = ROW_EDGES_B * NX * sizeof(uint2);
        const size_t bufferSize = sizeA + sizeB / NB;
        assert(bufferSize >= sizeB + sizeB / NB / 2); // ensure enough space for Round 1
        checkCudaErrors_V(cudaMalloc((void**)&bufferA, bufferSize));
        bufferAB = bufferA + sizeB / NB;
        bufferB  = bufferA + bufferSize - sizeB;
        assert(bufferA + sizeA == bufferB + sizeB * (NB-1) / NB); // ensure alignment of overlap
        cudaMemcpy(dt, this, sizeof(edgetrimmer), cudaMemcpyHostToDevice);
        initsuccess = true;
    }
    u64 globalbytes() const {
        return (sizeA+sizeB/NB) + (1+NB) * indexesSize + sizeof(siphash_keys) + PROOFSIZE * 2 * sizeof(u32) + sizeof(edgetrimmer);
    }
    ~edgetrimmer() {
        checkCudaErrors_V(cudaFree(bufferA));
        for (int i = 0; i < 1+NB; i++) {
            checkCudaErrors_V(cudaFree(indexesE[i]));
        }
        checkCudaErrors_V(cudaFree(dipkeys));
        checkCudaErrors_V(cudaFree(uvnodes));
        checkCudaErrors_V(cudaFree(dt));
        cudaDeviceReset();
    }
    u32 trim() {
        cudaEvent_t start, stop;
        checkCudaErrors(cudaEventCreate(&start)); checkCudaErrors(cudaEventCreate(&stop));
        cudaMemcpy(dipkeys, &sipkeys, sizeof(sipkeys), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        float durationA, durationB;
        cudaEventRecord(start, NULL);


#if ROO_VERBOSE
        printf("%s(): tp.genA.blocks=%d, tp.genA.tpb=%d, EDGE_BLOCK_SIZE=%d\n", __FUNCTION__, tp.genA.blocks, tp.genA.tpb, EDGE_BLOCK_SIZE);
        printf("%s(): tp.genB.blocks=%d, tp.genB.tpb=%d\n", __FUNCTION__, tp.genB.blocks, tp.genB.tpb);
        printf("%s(): tp.trim.blocks=%d, tp.trim.tpb=%d\n", __FUNCTION__, tp.trim.blocks, tp.trim.tpb);
        printf("%s(): tp.tail.blocks=%d, tp.tail.tpb=%d\n", __FUNCTION__, tp.tail.blocks, tp.tail.tpb);
#endif

        cudaMemset(indexesE[1], 0, indexesSize);
#if SEEDA_CUSTOM_BLOCK
        SeedA<EDGES_A><<<SEEDA_BLOCKS, tp.genA.tpb>>>(*dipkeys, (ulonglong4*)bufferAB, indexesE[1]);
#else
        SeedA<EDGES_A><<<tp.genA.blocks, tp.genA.tpb>>>(*dipkeys, (ulonglong4*)bufferAB, indexesE[1]);
#endif
        checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop); cudaEventElapsedTime(&durationA, start, stop);
        if (abort) return false;

#if DUMP_SEEDA
#define BUFFAB_FILE    "SeedA-bufferAB.bin"
#define INDEXE1_FILE   "SeedA-indexesE1.bin"
        {
            uint8_t* tmp = (uint8_t*)malloc(sizeA);
            assert(tmp);

            // bufferA
            //for (int i = 0; i < NX; i ++) print_bucket<<<1, 1>>>((u32*)bufferA, EDGES_A * 2, i/*row*/, 0/*col*/, EDGES_A/*nr_edges*/);
            cudaMemcpy(tmp, bufferAB, sizeA, cudaMemcpyDeviceToHost);
            FILE* fp = fopen(BUFFAB_FILE, "wb");
            fwrite(tmp, 1, sizeA, fp);
            fclose(fp);

            // indexsE[1]
            printf("indexesE[1] after SeedA():\n"); print_indexs<<<1,1>>>((uint32_t*)indexesE[1]);

            cudaMemcpy(tmp, indexesE[1], indexesSize, cudaMemcpyDeviceToHost);
            fp = fopen(INDEXE1_FILE, "wb");
            fwrite(tmp, 1, indexesSize, fp);
            fclose(fp);

            free(tmp);
        }
#endif

        cudaEventRecord(start, NULL);
        cudaMemset(indexesE[0], 0, indexesSize);

        u32 qA = sizeA/NA;
        u32 qE = NX2 / NA;
        for (u32 i = 0; i < NA; i++) {
            SeedB<EDGES_A><<<tp.genB.blocks/NA, tp.genB.tpb>>>((uint2*)(bufferAB+i*qA), (ulonglong4*)(bufferA+i*qA), indexesE[1]+i*qE, indexesE[0]+i*qE);
            if (abort) return false;
        }


        checkCudaErrors(cudaDeviceSynchronize()); cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop); cudaEventElapsedTime(&durationB, start, stop);
        checkCudaErrors(cudaEventDestroy(start)); checkCudaErrors(cudaEventDestroy(stop));
        print_log("Seeding completed in %.0f + %.0f ms\n", durationA, durationB);
        if (abort) return false;

#if ROO_VERBOSE
        live_edges<<<1,1>>>(-1, indexesE[0], indexesE[1]);
#endif

#if DUMP_SEEDB
#define BUFFA_FILE     "SeedB-bufferA.bin"
#define INDEXE0_FILE   "SeedB-indexesE0.bin"

        {
            uint8_t* tmp = (uint8_t*)malloc(sizeA);
            assert(tmp);

            // bufferA
            //printf("EDGES_A=%d\n", EDGES_A); // 136192
            //for (int i = 0; i < NX; i ++) print_bucket<<<1, 1>>>((u32*)bufferA, EDGES_A * 2, i/*row*/, 0/*col*/, EDGES_A/*nr_edges*/);
            cudaMemcpy(tmp, bufferA, sizeA, cudaMemcpyDeviceToHost);
            FILE* fp = fopen(BUFFA_FILE, "wb");
            fwrite(tmp, 1, sizeA, fp);
            fclose(fp);

            // indexsE[0]
            printf("indexesE[0] after SeedB():\n"); print_indexs<<<1,1>>>((uint32_t*)indexesE[0]);

            cudaMemcpy(tmp, indexesE[0], indexesSize, cudaMemcpyDeviceToHost);
            fp = fopen(INDEXE0_FILE, "wb");
            fwrite(tmp, 1, indexesSize, fp);
            fclose(fp);

            free(tmp);
        }
#endif
        for (u32 i = 0; i < NB; i++) cudaMemset(indexesE[1+i], 0, indexesSize);

        qA = sizeA/NB;
        const size_t qB = sizeB/NB;
        qE = NX2 / NB;
        for (u32 i = NB; i--; ) { // i = {1, 0}
#if ROUND0_CUSTOM_BLOCK
            printf("Round(0): i=%d\n", i);
            Round<1, EDGES_A, EDGES_B/NB><<<ROUND0_BLOCKS, tp.trim.tpb>>>(0, (uint2*)(bufferA+i*qA), (uint2*)(bufferB+i*qB), indexesE[0]+i*qE, indexesE[1+i]);
#else
            Round<1, EDGES_A, EDGES_B/NB><<<tp.trim.blocks/NB, tp.trim.tpb>>>(0, (uint2*)(bufferA+i*qA), (uint2*)(bufferB+i*qB), indexesE[0]+i*qE, indexesE[1+i]); // to .632, then to .316
#endif
            if (abort) return false;
        }

#if ROO_VERBOSE
        live_edges<<<1,1>>>(0, indexesE[1], indexesE[2]);
#endif

#if DUMP_ROUND0
#define ROUND0_INDEXE1_FILE     "Round0-indexesE1.bin"
#define ROUND0_INDEXE2_FILE     "Round0-indexesE2.bin"

        {
            uint8_t* tmp = (uint8_t*)malloc(indexesSize);
            assert(tmp);

            printf("indexesE[2] after Round(0):\n"); print_indexs<<<1,1>>>((uint32_t*)indexesE[2]);
            cudaMemcpy(tmp, indexesE[2], indexesSize, cudaMemcpyDeviceToHost);
            FILE* fp = fopen(ROUND0_INDEXE2_FILE, "wb");
            fwrite(tmp, 1, indexesSize, fp);
            fclose(fp);

            printf("indexesE[1] after Round(0):\n"); print_indexs<<<1,1>>>((uint32_t*)indexesE[1]);
            cudaMemcpy(tmp, indexesE[1], indexesSize, cudaMemcpyDeviceToHost);
            fp = fopen(ROUND0_INDEXE1_FILE, "wb");
            fwrite(tmp, 1, indexesSize, fp);
            fclose(fp);


            free(tmp);
        }
#endif
        // return 0; // time 53%

        cudaMemset(indexesE[0], 0, indexesSize);

#if ROUNDN_CUSTOM_BLOCK
        Round<NB, EDGES_B/NB, EDGES_B/2><<<ROUNDN_BLOCKS, tp.trim.tpb>>>(1, (const uint2 *)bufferB, (uint2 *)bufferA, indexesE[1], indexesE[0]); // to .296
#else
        Round<NB, EDGES_B/NB, EDGES_B/2><<<tp.trim.blocks, tp.trim.tpb>>>(1, (const uint2 *)bufferB, (uint2 *)bufferA, indexesE[1], indexesE[0]); // to .296
#endif
        if (abort) return false;
#if ROO_VERBOSE
        live_edges<<<1,1>>>(1, indexesE[0], indexesE[1]);
#endif
#if DUMP_ROUND1
#define ROUND1_INDEXE_FILE     "Round1-indexesE0.bin"

        {
            uint8_t* tmp = (uint8_t*)malloc(indexesSize);
            assert(tmp);

            printf("indexesE[0] after Round(1):\n"); print_indexs<<<1,1>>>((uint32_t*)indexesE[0]);
            cudaMemcpy(tmp, indexesE[0], indexesSize, cudaMemcpyDeviceToHost);
            FILE* fp = fopen(ROUND1_INDEXE_FILE, "wb");
            fwrite(tmp, 1, indexesSize, fp);
            fclose(fp);

            free(tmp);
        }
#endif
        cudaMemset(indexesE[1], 0, indexesSize);

        // return 0;  // about 61% run time

        Round<1, EDGES_B/2, EDGES_A/4><<<tp.trim.blocks, tp.trim.tpb>>>(2, (const uint2 *)bufferA, (uint2 *)bufferB, indexesE[0], indexesE[1]); // to .176
        if (abort) return false;
#if ROO_VERBOSE
        live_edges<<<1,1>>>(2, indexesE[0], indexesE[1]);
#endif

        cudaMemset(indexesE[0], 0, indexesSize);

        Round<1, EDGES_A/4, EDGES_B/4><<<tp.trim.blocks, tp.trim.tpb>>>(3, (const uint2 *)bufferB, (uint2 *)bufferA, indexesE[1], indexesE[0]); // to .117
        if (abort) return false;
#if ROO_VERBOSE
        live_edges<<<1,1>>>(3, indexesE[0], indexesE[1]);
#endif

        cudaDeviceSynchronize();

        //return 0;  // about 70% time

        for (int round = 4; round < tp.ntrims; round += 2) {
            cudaMemset(indexesE[1], 0, indexesSize);
            Round<1, EDGES_B/4, EDGES_B/4><<<tp.trim.blocks, tp.trim.tpb>>>(round, (const uint2 *)bufferA, (uint2 *)bufferB, indexesE[0], indexesE[1]);
            if (abort) return false;
#if ROO_VERBOSE
            live_edges<<<1,1>>>(round, indexesE[0], indexesE[1]);
#endif
            cudaMemset(indexesE[0], 0, indexesSize);
            Round<1, EDGES_B/4, EDGES_B/4><<<tp.trim.blocks, tp.trim.tpb>>>(round+1, (const uint2 *)bufferB, (uint2 *)bufferA, indexesE[1], indexesE[0]);
            if (abort) return false;
#if ROO_VERBOSE
            live_edges<<<1,1>>>(round+1, indexesE[0], indexesE[1]);
#endif
        }

#if DUMP_ROUND175
#define ROUND175_INDEXE_FILE     "Round175-indexesE0.bin"
#define ROUND175_BUFFA_FILE      "Round175-bufferA.bin"
        {
            uint8_t* tmp = (uint8_t*)malloc(sizeA);
            assert(tmp);

            printf("indexesE[0] after Round(175):\n"); print_indexs<<<1,1>>>((uint32_t*)indexesE[0]);
            cudaMemcpy(tmp, indexesE[0], indexesSize, cudaMemcpyDeviceToHost);
            FILE* fp = fopen(ROUND175_INDEXE_FILE, "wb");
            fwrite(tmp, 1, indexesSize, fp);
            fclose(fp);

            cudaMemcpy(tmp, bufferA, sizeA, cudaMemcpyDeviceToHost);
            fp = fopen(ROUND175_BUFFA_FILE, "wb");
            fwrite(tmp, 1, sizeA, fp);
            fclose(fp);
            free(tmp);
        }
#endif
        cudaMemset(indexesE[1], 0, indexesSize);
        cudaDeviceSynchronize();

        Tail<EDGES_B/4><<<tp.tail.blocks, tp.tail.tpb>>>((const uint2 *)bufferA, (uint2 *)bufferB, indexesE[0], indexesE[1]);
        cudaMemcpy(&nedges, indexesE[1], sizeof(u32), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();


#if DUMP_TAIL
#define TAIL_BUFFB_FILE      "Tail-bufferB.bin"
        {
            uint8_t* tmp = (uint8_t*)malloc(sizeB);
            assert(tmp);

            cudaMemcpy(tmp, indexesE[0], indexesSize, cudaMemcpyDeviceToHost);
            uint32_t sum = 0;
            for (int i = 0; i < NX2; i ++) {
                sum += ((uint32_t*)tmp)[i];
            }

            printf("after Tail(): nedges = %d, indexesE[0] sum=%d\n", nedges, sum);
            assert(nedges == sum);

            cudaMemcpy(tmp, bufferB, sizeB, cudaMemcpyDeviceToHost);
            FILE* fp = fopen(TAIL_BUFFB_FILE, "wb");
            fwrite(tmp, 1, nedges * sizeof(uint2), fp);
            fclose(fp);

            free(tmp);
        }
#endif

        return nedges;
    }
};

struct solver_ctx {
    edgetrimmer trimmer;
    bool mutatenonce;
    uint2 *edges;
    graph<word_t> cg;
    uint2 soledges[PROOFSIZE];
    std::vector<u32> sols; // concatenation of all proof's indices

    solver_ctx(const trimparams tp, bool mutate_nonce) : trimmer(tp), cg(MAXEDGES, MAXEDGES, MAXSOLS, IDXSHIFT) {
        edges   = new uint2[MAXEDGES];
        mutatenonce = mutate_nonce;
    }

    void setheadernonce(char * const headernonce, const u32 len, const u32 nonce) {
	printf("setheadernonce(): headernonce=%s, len=%d, nonce=0x%08x, mutatenonce=%d\n", headernonce, len, nonce, mutatenonce);
	//printf("setheadernonce(): sipkeys=%d\n", trimmer.sipkeys);
        if (mutatenonce)
            ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
        setheader(headernonce, len, &trimmer.sipkeys);
#if NULL_SIPKEYS
        trimmer.sipkeys.k0 = 0;
        trimmer.sipkeys.k1 = 0;
        trimmer.sipkeys.k2 = 0;
        trimmer.sipkeys.k3 = 0;
#endif
        sols.clear();
    }
    ~solver_ctx() {
        delete[] edges;
    }

    int findcycles(uint2 *edges, u32 nedges) {
        cg.reset();
        for (u32 i = 0; i < nedges; i++)
            cg.add_compress_edge(edges[i].x, edges[i].y);
        for (u32 s = 0 ;s < cg.nsols; s++) {
            // print_log("Solution");
            for (u32 j = 0; j < PROOFSIZE; j++) {
                soledges[j] = edges[cg.sols[s][j]];
                // print_log(" (%x, %x)", soledges[j].x, soledges[j].y);
            }
            // print_log("\n");
            sols.resize(sols.size() + PROOFSIZE);
            cudaMemcpyToSymbol(recoveredges, soledges, sizeof(soledges));
            cudaMemset(trimmer.indexesE[1], 0, trimmer.indexesSize);
            Recovery<<<trimmer.tp.recover.blocks, trimmer.tp.recover.tpb>>>(*trimmer.dipkeys, (ulonglong4*)trimmer.bufferA, (int *)trimmer.indexesE[1]);
            cudaMemcpy(&sols[sols.size()-PROOFSIZE], trimmer.indexesE[1], PROOFSIZE * sizeof(u32), cudaMemcpyDeviceToHost);
            checkCudaErrors(cudaDeviceSynchronize());
            qsort(&sols[sols.size()-PROOFSIZE], PROOFSIZE, sizeof(u32), cg.nonce_cmp);
        }
        return 0;
    }

    int solve() {
        u64 time0, time1;
        u32 timems,timems2;

        trimmer.abort = false;
        time0 = timestamp();
        u32 nedges = trimmer.trim();
        if (!nedges)
            return 0;
        if (nedges > MAXEDGES) {
            print_log("OOPS; losing %d edges beyond MAXEDGES=%d\n", nedges-MAXEDGES, MAXEDGES);
            nedges = MAXEDGES;
        }
        cudaMemcpy(edges, trimmer.bufferB, sizeof(uint2[nedges]), cudaMemcpyDeviceToHost);
        time1 = timestamp(); timems  = (time1 - time0) / 1000000;
        time0 = timestamp();
        findcycles(edges, nedges);
        time1 = timestamp(); timems2 = (time1 - time0) / 1000000;
        print_log("findcycles edges %d time %d ms total %d ms\n", nedges, timems2, timems+timems2);
        return sols.size() / PROOFSIZE;
    }

    void abort() {
        trimmer.abort = true;
    }
};

#include <unistd.h>

// arbitrary length of header hashed into siphash key
#define HEADERLEN 80

typedef solver_ctx SolverCtx;

CALL_CONVENTION int run_solver(SolverCtx* ctx,
        char* header,
        int header_length,
        u32 nonce,
        u32 range,
        SolverSolutions *solutions,
        SolverStats *stats
        )
{
    u64 time0, time1;
    u32 timems;
    u32 sumnsols = 0;
    int device_id;
    if (stats != NULL) {
        cudaGetDevice(&device_id);
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, stats->device_id);
        stats->device_id = device_id;
        stats->edge_bits = EDGEBITS;
        strncpy(stats->device_name, props.name, MAX_NAME_LEN);
    }

    if (ctx == NULL || !ctx->trimmer.initsuccess){
        print_log("Error initialising trimmer. Aborting.\n");
        print_log("Reason: %s\n", LAST_ERROR_REASON);
        if (stats != NULL) {
            stats->has_errored = true;
            strncpy(stats->error_reason, LAST_ERROR_REASON, MAX_NAME_LEN);
        }
        return 0;
    }

    for (u32 r = 0; r < range; r++) {
        time0 = timestamp();
        ctx->setheadernonce(header, header_length, nonce + r);
        print_log("nonce %d k0 k1 k2 k3 %llx %llx %llx %llx\n", nonce+r, ctx->trimmer.sipkeys.k0, ctx->trimmer.sipkeys.k1, ctx->trimmer.sipkeys.k2, ctx->trimmer.sipkeys.k3);
        u32 nsols = ctx->solve();
        time1 = timestamp();
        timems = (time1 - time0) / 1000000;
        print_log("Time: %d ms\n", timems);
        for (unsigned s = 0; s < nsols; s++) {
            print_log("Solution");
            u32* prf = &ctx->sols[s * PROOFSIZE];
            for (u32 i = 0; i < PROOFSIZE; i++)
                print_log(" %jx", (uintmax_t)prf[i]);
            print_log("\n");
            if (solutions != NULL){
                solutions->edge_bits = EDGEBITS;
                solutions->num_sols++;
                solutions->sols[sumnsols+s].nonce = nonce + r;
                for (u32 i = 0; i < PROOFSIZE; i++) 
                    solutions->sols[sumnsols+s].proof[i] = (u64) prf[i];
            }
            int pow_rc = verify(prf, ctx->trimmer.sipkeys);
            if (pow_rc == POW_OK) {
                print_log("Verified with cyclehash ");
                unsigned char cyclehash[32];
                blake2b((void *)cyclehash, sizeof(cyclehash), (const void *)prf, sizeof(proof), 0, 0);
                for (int i=0; i<32; i++)
                    print_log("%02x", cyclehash[i]);
                print_log("\n");
            } else {
                print_log("FAILED due to %s\n", errstr[pow_rc]);
            }
        }
        sumnsols += nsols;
        if (stats != NULL) {
            stats->last_start_time = time0;
            stats->last_end_time = time1;
            stats->last_solution_time = time1 - time0;
        }
    }
    print_log("%d total solutions\n", sumnsols);
    return sumnsols > 0;
}

CALL_CONVENTION SolverCtx* create_solver_ctx(SolverParams* params) {
    trimparams tp;
    tp.ntrims = params->ntrims;
    tp.genA.blocks = params->genablocks;
    tp.genA.tpb = params->genatpb;
    tp.genB.tpb = params->genbtpb;
    tp.trim.tpb = params->trimtpb;
    tp.tail.tpb = params->tailtpb;
    tp.recover.blocks = params->recoverblocks;
    tp.recover.tpb = params->recovertpb;

    cudaDeviceProp prop;
    checkCudaErrors_N(cudaGetDeviceProperties(&prop, params->device));

    assert(tp.genA.tpb <= prop.maxThreadsPerBlock);
    assert(tp.genB.tpb <= prop.maxThreadsPerBlock);
    assert(tp.trim.tpb <= prop.maxThreadsPerBlock);
    // assert(tp.tailblocks <= prop.threadDims[0]);
    assert(tp.tail.tpb <= prop.maxThreadsPerBlock);
    assert(tp.recover.tpb <= prop.maxThreadsPerBlock);

    assert(tp.genA.blocks * tp.genA.tpb * EDGE_BLOCK_SIZE <= NEDGES); // check THREADS_HAVE_EDGES
    assert(tp.recover.blocks * tp.recover.tpb * EDGE_BLOCK_SIZE <= NEDGES); // check THREADS_HAVE_EDGES

    /*
     * in genA, a gpu-block writes to a column of buckets. assuming that edges are evenly distributed among
     * NX buckets in the row, for each _syncthreads(), there will be tpb new edges added into the tmp[NX][] array, before
     * flushing into global memory. this is to ensure that size in tmp[NX][FLUSHA*2] is big enough.  `*2` is a contingency
     * protection.
     */
    assert(tp.genA.tpb / NX <= FLUSHA); // check ROWS_LIMIT_LOSSES

    /*
     *
     */
    assert(tp.genB.tpb / NX <= FLUSHB); // check COLS_LIMIT_LOSSES

    cudaSetDevice(params->device);
    if (!params->cpuload)
        checkCudaErrors_N(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

    SolverCtx* ctx = new SolverCtx(tp, params->mutate_nonce);

    return ctx;
}

CALL_CONVENTION void destroy_solver_ctx(SolverCtx* ctx) {
    delete ctx;
}

CALL_CONVENTION void stop_solver(SolverCtx* ctx) {
    ctx->abort();
}

CALL_CONVENTION void fill_default_params(SolverParams* params) {
    trimparams tp;
    params->device = 0;
    params->ntrims = tp.ntrims;

    /*
     * NEDGES/EDGE_BLOCK_SIZE: number of edge-blocks
     * NEDGES/EDGE_BLOCK_SIZE/tp.genA.tpb: number of blocks
     * tp.genA.blocks = 4096, which is the up-limit of block numbers.
     * - in case of cuda19, cuda blocks=32, in genA, each thread just handles EDGE_BLOCK_SIZE (i.e., 64) edges
     * - in case of cuda29, cuda blocks=4096 (up-limit), each thread handles 8 EDGE_BLOCK_SIZE (i.e., 512) edges
     */
    params->genablocks = min(tp.genA.blocks, NEDGES/EDGE_BLOCK_SIZE/tp.genA.tpb);
    params->genatpb = tp.genA.tpb;


    params->genbtpb = tp.genB.tpb;
    params->trimtpb = tp.trim.tpb;
    params->tailtpb = tp.tail.tpb;
    params->recoverblocks = min(tp.recover.blocks, NEDGES/EDGE_BLOCK_SIZE/tp.recover.tpb);
    params->recovertpb = tp.recover.tpb;
    params->cpuload = false;
}

int main(int argc, char **argv) {
    trimparams tp;
    u32 nonce = 0;
    u32 range = 1;
    u32 device = 0;
    char header[HEADERLEN];
    u32 len;
    int c;

    printf("EDGES_A=%d, EDGES_B=%d\n", EDGES_A, EDGES_B);

    // set defaults
    SolverParams params;
    fill_default_params(&params);

    memset(header, 0, sizeof(header));
    while ((c = getopt(argc, argv, "scb:d:h:k:m:n:r:U:u:v:w:y:Z:z:")) != -1) {
        switch (c) {
            case 's':
                print_log("SYNOPSIS\n  cuda%d [-s] [-c] [-d device] [-h hexheader] [-m trims] [-n nonce] [-r range] [-U seedAblocks] [-u seedAthreads] [-v seedBthreads] [-w Trimthreads] [-y Tailthreads] [-Z recoverblocks] [-z recoverthreads]\n", NODEBITS);
                print_log("DEFAULTS\n  cuda%d -d %d -h \"\" -m %d -n %d -r %d -U %d -u %d -v %d -w %d -y %d -Z %d -z %d\n", NODEBITS, device, tp.ntrims, nonce, range, tp.genA.blocks, tp.genA.tpb, tp.genB.tpb, tp.trim.tpb, tp.tail.tpb, tp.recover.blocks, tp.recover.tpb);
                exit(0);
            case 'c':
                params.cpuload = false;
                break;
            case 'd':
                device = params.device = atoi(optarg);
                break;
            case 'h':
                len = strlen(optarg)/2;
                assert(len <= sizeof(header));
                for (u32 i=0; i<len; i++)
                    sscanf(optarg+2*i, "%2hhx", header+i); // hh specifies storage of a single byte
                break;
            case 'n':
                nonce = atoi(optarg);
                break;
            case 'm':
                params.ntrims = atoi(optarg) & -2; // make even as required by solve()
                break;
            case 'r':
                range = atoi(optarg);
                break;
            case 'U':
                params.genablocks = atoi(optarg);
                break;
            case 'u':
                params.genatpb = atoi(optarg);
                break;
            case 'v':
                params.genbtpb = atoi(optarg);
                break;
            case 'w':
                params.trimtpb = atoi(optarg);
                break;
            case 'y':
                params.tailtpb = atoi(optarg);
                break;
            case 'Z':
                params.recoverblocks = atoi(optarg);
                break;
            case 'z':
                params.recovertpb = atoi(optarg);
                break;
        }
    }

    int nDevices;
    checkCudaErrors(cudaGetDeviceCount(&nDevices));
    assert(device < nDevices);
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));
    u64 dbytes = prop.totalGlobalMem;
    int dunit;
    for (dunit=0; dbytes >= 102040; dbytes>>=10,dunit++) ;
    print_log("%s with %d%cB @ %d bits x %dMHz\n", prop.name, (u32)dbytes, " KMGT"[dunit], prop.memoryBusWidth, prop.memoryClockRate/1000);
    // cudaSetDevice(device);

    print_log("Looking for %d-cycle on cuckaroo%d(\"%s\",%d", PROOFSIZE, EDGEBITS, header, nonce);
    if (range > 1)
        print_log("-%d", nonce+range-1);
    print_log(") with 50%% edges, %d*%d buckets, %d trims, and %d thread blocks.\n", NX, NY, params.ntrims, NX);

    SolverCtx* ctx = create_solver_ctx(&params);

    u64 bytes = ctx->trimmer.globalbytes();
    int unit;
    for (unit=0; bytes >= 102400; bytes>>=10,unit++) ;
    print_log("Using %d%cB of global memory.\n", (u32)bytes, " KMGT"[unit]);

    run_solver(ctx, header, sizeof(header), nonce, range, NULL, NULL);

    return 0;
}
