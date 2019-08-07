#ifndef __CUCKOO_H__
#define __CUCKOO_H__

// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2017 John Tromp

#include <stdint.h> // for types uint32_t,uint64_t
#include <string.h> // for functions strlen, memset
#include <stdarg.h>
#include <stdio.h>
#ifndef __NVCC__
#include <chrono>
#include <ctime>
#endif // __NVCC__
#include "../crypto/blake2.h"
#include "../crypto/siphash.hpp"

#ifdef SIPHASH_COMPAT
#include <stdio.h>
#endif

// proof-of-work parameters
#ifndef EDGEBITS
// the main parameter is the 2-log of the graph size,
// which is the size in bits of the node identifiers
#define EDGEBITS 29
#endif
#ifndef PROOFSIZE
// the next most important parameter is the (even) length
// of the cycle to be found. a minimum of 12 is recommended
#define PROOFSIZE 42
#endif

// moved from simple.cpp
#define NNODES (2 * NEDGES)
#define NCUCKOO NNODES

// save some keystrokes since i'm a lazy typer
typedef uint32_t u32;
typedef uint64_t u64;

#ifndef MAX_SOLS
#define MAX_SOLS 4
#endif

#if EDGEBITS > 30
typedef uint64_t word_t;
#elif EDGEBITS > 14
typedef u32 word_t;
#else // if EDGEBITS <= 14
typedef uint16_t word_t;
#endif

// number of edges
#define NEDGES ((word_t)1 << EDGEBITS)
// used to mask siphash output
#define EDGEMASK ((word_t)NEDGES - 1)
#define NODE1MASK EDGEMASK

// Common Solver parameters, to return to caller
struct SolverParams {
        u32 nthreads = 0;
        u32 ntrims = 0;
        bool showcycle;
        bool allrounds;
        bool mutate_nonce = 1;
        bool cpuload = 1;

        // Common cuda params
        u32 device = 0;

        // Cuda-lean specific params
        u32 blocks = 0;
        u32 tpb = 0;

        // Cuda-mean specific params
        u32 expand = 0;
        u32 genablocks = 0;
        u32 genatpb = 0;
        u32 genbtpb = 0;
        u32 trimtpb = 0;
        u32 tailtpb = 0;
        u32 recoverblocks = 0;
        u32 recovertpb = 0;
};

// Solutions result structs to be instantiated by caller,
// and filled by solver if desired
struct Solution {
 u64 nonce = 0;
 u64 proof[PROOFSIZE];
};

struct SolverSolutions {
 u32 edge_bits = 0;
 u32 num_sols = 0;
 Solution sols[MAX_SOLS];
};

#define MAX_NAME_LEN 256

// last error reason, to be picked up by stats
// to be returned to caller
extern char LAST_ERROR_REASON[MAX_NAME_LEN];

// Solver statistics, to be instantiated by caller
// and filled by solver if desired
struct SolverStats {
        u32 device_id = 0;
        u32 edge_bits = 0;
        char plugin_name[MAX_NAME_LEN]; // will be filled in caller-side
        char device_name[MAX_NAME_LEN];
        bool has_errored = false;
        char error_reason[MAX_NAME_LEN];
        u32 iterations = 0;
        u64 last_start_time = 0;
        u64 last_end_time = 0;
        u64 last_solution_time = 0;
};


enum verify_code { POW_OK, POW_HEADER_LENGTH, POW_TOO_BIG, POW_TOO_SMALL, POW_NON_MATCHING, POW_BRANCH, POW_DEAD_END, POW_SHORT_CYCLE};
extern const char *errstr[];

#ifndef __NVCC__
// generate edge endpoint in cuckoo graph without partition bit
word_t sipnode(siphash_keys *keys, word_t edge, u32 uorv);
#endif
// verify that edges are ascending and form a cycle in header-generated graph
int verify(word_t edges[PROOFSIZE], siphash_keys *keys) ;
// convenience function for extracting siphash keys from header
void setheader(const char *header, const u32 headerlen, siphash_keys *keys) ;
// edge endpoint in cuckoo graph with partition bit
word_t sipnode_(siphash_keys *keys, word_t edge, u32 uorv) ;
u64 timestamp() ;

/////////////////////////////////////////////////////////////////
// Declarations to make it easier for callers to link as required
/////////////////////////////////////////////////////////////////

#ifndef C_CALL_CONVENTION
#define C_CALL_CONVENTION 0
#endif

// convention to prepend to called functions
#if C_CALL_CONVENTION
#define CALL_CONVENTION extern "C"
#else
#define CALL_CONVENTION
#endif

// Ability to squash printf output at compile time, if desired
#ifndef SQUASH_OUTPUT
#define SQUASH_OUTPUT 0
#endif

#if (0)
void print_log(const char *fmt, ...) {
        if (SQUASH_OUTPUT) return;
        va_list args;
        va_start(args, fmt);
        vprintf(fmt, args);
        va_end(args);
}
#else
void print_log(const char *fmt, ...) ;
#endif
//////////////////////////////////////////////////////////////////
// END caller QOL
//////////////////////////////////////////////////////////////////



#endif // __CUCKOO_H__
