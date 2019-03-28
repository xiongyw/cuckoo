// Cuckoo Cycle, a memory-hard proof-of-work
// Copyright (c) 2013-2019 John Tromp
// The edge-trimming memory optimization is due to Dave Andersen
// http://da-data.blogspot.com/2014/03/a-public-review-of-cuckoo-cycle.html
// The use of prefetching was suggested by Alexander Peslyak (aka Solar Designer)
// define SINGLECYCLING to run cycle finding single threaded which runs slower
// but avoids losing cycles to race conditions (not worth it in my testing)

#include "cuckoo.h"
#include "../crypto/siphashxN.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "../threads/barrier.hpp"
#include <assert.h>

typedef uint64_t u64; // save some typing

// atomic access to nodes' degree counter when parallelizing edge-trimming
#ifdef ATOMIC
#include <atomic>
typedef std::atomic<u32> au32;
typedef std::atomic<u64> au64;
#else
typedef u32 au32;
typedef u64 au64;
#endif

#ifndef SIZEOF_TWICE_ATOM
#define SIZEOF_TWICE_ATOM 4
#endif
#if SIZEOF_TWICE_ATOM == 8
typedef au64 atwice;
typedef u64 uatwice;
#elif SIZEOF_TWICE_ATOM == 4
typedef au32 atwice;
typedef u32 uatwice;
#elif SIZEOF_TWICE_ATOM == 1
typedef unsigned char atwice;
typedef unsigned char uatwice;
#else
#error not implemented
#endif

#include <set>

// algorithm/performance parameters; assume EDGEBITS < 31

const u32 NODEBITS = EDGEBITS + 1;
const word_t NODEMASK = (EDGEMASK << 1) | (word_t)1;

#ifndef PART_BITS
// #bits used to partition "edge set processing" to save memory
// a value of 0 does no partitioning and is fastest
// a value of 1 partitions in two, making twice_set the
// same size as shrinkingset at about 33% slowdown
// higher values are not that interesting
#define PART_BITS 0
#endif

#ifndef NPREFETCH
// how many prefetches to queue up
// before accessing the memory
// must be a multiple of NSIPHASH
#define NPREFETCH 32
#endif

#ifndef IDXSHIFT
// we want sizeof(cuckoo_hash) == sizeof(twice_set), so
// CUCKOO_SIZE * sizeof(u64)   == 2 * ONCE_BITS / 32
// CUCKOO_SIZE * 2             == 2 * ONCE_BITS / 32
// (NNODES >> IDXSHIFT) * 2      == 2 * ONCE_BITS / 32
// NNODES >> IDXSHIFT            == NEDGES >> PART_BITS >> 5
// IDXSHIFT                    == 1 + PART_BITS + 5
#define IDXSHIFT (PART_BITS + 6)
#endif
// grow with cube root of size, hardly affected by trimming
const u32 MAXPATHLEN = 8 << (NODEBITS/3);

const u32 PART_MASK = (1 << PART_BITS) - 1;
// number of nodes in one partition
const u64 ONCE_BITS = NEDGES >> PART_BITS;
// counter storage size (in bytes) for one partition
const u64 TWICE_BYTES = (2 * ONCE_BITS) / 8;
// counter storage size (in atwice) for one partition
const u64 TWICE_ATOMS = TWICE_BYTES / sizeof(atwice);
// number of node counters per atomic access: 1 byte (8-bit) can store degree count for 4 nodes, as each node uses 2 bits
const u32 TWICE_PER_ATOM = sizeof(atwice) * 4;

// twice_set class is used to store nodes' degree capping at 2. to increase
// the degree of a node, call set() once, and the degree will not increase
// anymore (capped) after two set() calls.
class twice_set {
public:
  atwice *bits;

  twice_set() {
    bits = new atwice[TWICE_ATOMS];
    assert(bits != 0);
  }
  void clear() {
    assert(bits);
    for (u32 i=0; i < TWICE_ATOMS; i++)
      bits[i] = 0;
  }
 void prefetch(word_t u) const {
#ifdef PREFETCH
    __builtin_prefetch((const void *)(&bits[u/TWICE_PER_ATOM]), /*READ=*/0, /*TEMPORAL=*/0);
#endif
  }

  void set(word_t u) {
    word_t idx = u/TWICE_PER_ATOM; // ATOM idx where the 2-bit resides
    uatwice bit = (uatwice)1 << (2 * (u%TWICE_PER_ATOM));
#ifdef ATOMIC
    uatwice old = std::atomic_fetch_or_explicit(&bits[idx], bit, std::memory_order_relaxed);
    if (old & bit) std::atomic_fetch_or_explicit(&bits[idx], bit<<1, std::memory_order_relaxed);
#else
    uatwice old = bits[idx];
    // suppose that bit=2'b01(u%TWICE_PER_ATOM==0), old=2'b00, then
    // 1st set: 00 | (01 + (00 & 01)) --> old = 01
    // 2nd set: 01 | (01 + (01 & 01)) --> old = 11
    // 3rd set: 11 | (01 + (11 & 01)) --> old = 11
    bits[idx] = old | (bit + (old & bit));
#endif
  }

  // return true if the node degree >=2; otherwise false
  bool test(word_t u) const {
#ifdef ATOMIC
    return ((bits[u/TWICE_PER_ATOM].load(std::memory_order_relaxed)
            >> (2 * (u%TWICE_PER_ATOM))) & 2) != 0;
#else
    return (bits[u/TWICE_PER_ATOM] >> (2 * (u%TWICE_PER_ATOM)) & 2) != 0;
#endif
  }

  ~twice_set() {
    delete[] bits;
  }
};

// set that starts out full and gets reset by threads on disjoint words
class shrinkingset {
public:
  u64 *bits;  // edge bit map, one bit per edge, 0 for alive, 1 for dead. a block contains 64 edge.
  u64 *cnt;   // one live-edge count per thread
  u32 nthreads;

  /**
   * better to use a variable for NEDGES, e.g. constructor(ne, nt)
   * - nt: number of threads
   */
  shrinkingset(const u32 nt) {
    bits = (u64 *)malloc(NEDGES/8);  // better to use a varaible for NEDGES
    cnt  = (u64 *)malloc(nt * sizeof(u64));
    nthreads = nt;
  }

  ~shrinkingset() {
    free(bits);
    free(cnt);
  }

  // set all edges to be live
  void clear() {
    memset(bits, 0, NEDGES/8);
    memset(cnt, 0, nthreads * sizeof(u64));
    cnt[0] = NEDGES;
  }

  // total number of live edges
  u64 count() const {
    u64 sum = 0LL;
    for (u32 i=0; i<nthreads; i++)
      sum += cnt[i];
    return sum;
  }

  /**
   * kill an edge
   * - n: edge index
   * - thread: the thread idx corresponding to the edge
   */
  void reset(word_t n, u32 thread) {
    bits[n/64] |= 1LL << (n%64);
    cnt[thread]--;
  }

  /**
   * return liveness an edge (by its idx)
   * return 1 if alive; otherwise dead
   */
  bool test(word_t n) const {
    return !((bits[n/64] >> (n%64)) & 1LL);
  }

  // return liveness of a block which contains the edge idx n
  // note that the liveness meaning is changed (why?)
  u64 block(word_t n) const {
    return ~bits[n/64];
  }
};

const u64 CUCKOO_SIZE = NEDGES >> (IDXSHIFT-1); // NNODES >> IDXSHIFT;
const u64 CUCKOO_MASK = CUCKOO_SIZE - 1;
// number of (least significant) key bits that survives leftshift by NODEBITS
const u32 KEYBITS = 64-NODEBITS;
const u64 KEYMASK = (1LL << KEYBITS) - 1;
const u64 MAXDRIFT = 1LL << (KEYBITS - IDXSHIFT);

class cuckoo_hash {
public:
  au64 *cuckoo;

  cuckoo_hash(void *recycle) {
    cuckoo = (au64 *)recycle;
    for (u32 i=0; i < CUCKOO_SIZE; i++)
      cuckoo[i] = 0;
  }

  void set(word_t u, word_t v) {
    u64 niew = (u64)u << NODEBITS | v;
    for (word_t ui = u >> IDXSHIFT; ; ui = (ui+1) & CUCKOO_MASK) {
#if !defined(SINGLECYCLING) && defined(ATOMIC)
      u64 old = 0;
      if (cuckoo[ui].compare_exchange_strong(old, niew, std::memory_order_relaxed))
        return;
      if ((old >> NODEBITS) == (u & KEYMASK)) {
        cuckoo[ui].store(niew, std::memory_order_relaxed);
        return;
      }
#else
      u64 old = cuckoo[ui];
      if (old == 0 || (old >> NODEBITS) == (u & KEYMASK)) {
        cuckoo[ui] = niew;
        return;
      }
#endif
    }
  }

  word_t operator[](word_t u) const {
    for (word_t ui = u >> IDXSHIFT; ; ui = (ui+1) & CUCKOO_MASK) {
#if !defined(SINGLECYCLING) && defined(ATOMIC)
      u64 cu = cuckoo[ui].load(std::memory_order_relaxed);
#else
      u64 cu = cuckoo[ui];
#endif
      if (!cu)
        return 0;
      if ((cu >> NODEBITS) == (u & KEYMASK)) {
        assert(((ui - (u >> IDXSHIFT)) & CUCKOO_MASK) < MAXDRIFT);
        return (word_t)(cu & NODEMASK);
      }
    }
  }
};

class cuckoo_ctx {
public:
  siphash_keys sip_keys;
  shrinkingset *alive;
  twice_set *nonleaf;
  cuckoo_hash *cuckoo;
  word_t (*sols)[PROOFSIZE];
  u32 nonce;
  u32 maxsols;
  au32 nsols;
  u32 nthreads;
  u32 ntrims;
  trim_barrier barry;

  cuckoo_ctx(u32 n_threads, u32 n_trims, u32 max_sols) : barry(n_threads) {
    nthreads = n_threads;
    alive = new shrinkingset(nthreads);
    cuckoo = 0;
    nonleaf = new twice_set;
    ntrims = n_trims;
    sols = (word_t (*)[PROOFSIZE])calloc(maxsols = max_sols, PROOFSIZE*sizeof(word_t));
    assert(sols != 0);
    nsols = 0;
  }

  void setheadernonce(char* headernonce, const u32 len, const u32 nce) {
    nonce = nce;
    ((u32 *)headernonce)[len/sizeof(u32)-1] = htole32(nonce); // place nonce at end
    setheader(headernonce, len, &sip_keys);
    alive->clear(); // set all edges to be alive
    nsols = 0;
  }

  ~cuckoo_ctx() {
    delete alive;
    delete nonleaf;
    delete cuckoo;
  }

  void barrier() {
    barry.wait();
  }

  void abort() {
    barry.abort();
  }

  //
  void prefetch(const u64 *hashes, const u32 part) const {
    for (u32 i=0; i < NSIPHASH; i++) {
      u32 u = hashes[i] & EDGEMASK;
      if ((u & PART_MASK) == part) {
        nonleaf->prefetch(u >> PART_MASK);
      }
    }
  }

  /**
   * capping-increase the degree of a bunch of nodes
   * - hashes[]: array storing the H(key, 2i) or H(key, 2i+1)
   * - nsiphash: number of hashes in the array above
   * - part: partition index the nodes belong to
   */
  void node_deg(const u64 *hashes, const u32 nsiphash, const u32 part) const {
    for (u32 i=0; i < nsiphash; i++) {
#ifdef SKIPZERO
    if (!hashes[i])
      continue;
#endif
      u32 u = hashes[i] & EDGEMASK;
      if ((u & PART_MASK) == part) {
        nonleaf->set(u >>= PART_BITS);
      }
    }
  }

  // update nodes' degree by thread `id` for `uorv` round on partition `part`
  // - id: thread idx
  // - uorv: even or odd round
  // - part: partition idx
  void count_node_deg(const u32 id, const u32 uorv, const u32 part) {
    alignas(64) u64 indices[NSIPHASH];
    alignas(64) u64 hashes[NPREFETCH];

    memset(hashes, 0, NPREFETCH * sizeof(u64)); // allow many nonleaf->set(0) to reduce branching
    u32 nidx = 0;
    // "To maintain efficient access to the bitmap of live edges, each thread
    // handles words spaced T apart."
    // a block is hard-coded as 64 bits (edges), and each thread handles block by block. e.g., suppose there
    // are totally 4 threads, then:
    // block  id: 0 1 2 3  4 5 6 7   8 9 10 11 ...
    // thread id: 0 1 2 3  0 1 2 3   0 1  2  3
    for (word_t edge_idx = id*64; edge_idx < NEDGES; edge_idx += nthreads*64) {
        u64 alive64 = alive->block(edge_idx);
        word_t nonce; // micro nonce (i.e., edge idx)
        for (nonce = edge_idx-1; alive64; ) { // -1 compensates for 1-based ffs
            u32 ffs = __builtin_ffsll(alive64);  // find first bit set, `man ffs`
            nonce += ffs;
            alive64 >>= ffs;
            indices[nidx++ % NSIPHASH] = 2*nonce + uorv;
            if (nidx % NSIPHASH == 0) {
                node_deg(hashes+nidx-NSIPHASH, NSIPHASH, part);
                // get N hashes into hashes[] at once
                siphash24xN(&sip_keys, indices, hashes+nidx-NSIPHASH);

                prefetch(hashes+nidx-NSIPHASH, part);
                nidx %= NPREFETCH;
            }
            if (ffs & 64) break; // can't shift by 64
        }
    }
    node_deg(hashes, NPREFETCH, part);
    if (nidx % NSIPHASH != 0) {
      siphash24xN(&sip_keys, indices, hashes+(nidx&-NSIPHASH));
      node_deg(hashes+(nidx&-NSIPHASH), nidx%NSIPHASH, part);
    }
  }

  //
  void kill(const u64 *hashes, const u64 *indices, const u32 nsiphash,
             const u32 part, const u32 id) const {
    for (u32 i=0; i < nsiphash; i++) {
#ifdef SKIPZERO
    if (!hashes[i])
      continue;
#endif
      u32 u = hashes[i] & EDGEMASK;
      if ((u & PART_MASK) == part && !nonleaf->test(u >> PART_BITS)) {
        alive->reset(indices[i]/2, id);
      }
    }
  }

  //
  void kill_leaf_edges(const u32 id, const u32 uorv, const u32 part) {
    alignas(64) u64 indices[NPREFETCH];
    alignas(64) u64 hashes[NPREFETCH];

    memset(hashes, 0, NPREFETCH * sizeof(u64)); // allow many nonleaf->test(0) to reduce branching
    u32 nidx = 0;
    for (word_t block = id*64; block < NEDGES; block += nthreads*64) {
      u64 alive64 = alive->block(block);
      for (word_t nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
        u32 ffs = __builtin_ffsll(alive64);
        nonce += ffs; alive64 >>= ffs;
        indices[nidx++] = 2*nonce + uorv;
        if (nidx % NSIPHASH == 0) {
          siphash24xN(&sip_keys, indices+nidx-NSIPHASH, hashes+nidx-NSIPHASH);
          prefetch(hashes+nidx-NSIPHASH, part);
          nidx %= NPREFETCH;
          kill(hashes+nidx, indices+nidx, NSIPHASH, part, id);
        }
        if (ffs & 64) break; // can't shift by 64
      }
    }
    const u32 pnsip = nidx & -NSIPHASH;
    if (pnsip != nidx) {
      siphash24xN(&sip_keys, indices+pnsip, hashes+pnsip);
    }
    kill(hashes, indices, nidx, part, id);
    const u32 nnsip = pnsip + NSIPHASH;
    kill(hashes+nnsip, indices+nnsip, NPREFETCH-nnsip, part, id);
  }

  //
  void solution(word_t *us, u32 nu, word_t *vs, u32 nv) {
    typedef std::pair<word_t,word_t> edge;
    std::set<edge> cycle;
    u32 n = 0;
    cycle.insert(edge(*us, *vs));
    while (nu--)
      cycle.insert(edge(us[(nu+1)&~1], us[nu|1])); // u's in even position; v's in odd
    while (nv--)
      cycle.insert(edge(vs[nv|1], vs[(nv+1)&~1])); // u's in odd position; v's in even
  #ifdef ATOMIC
    u32 soli = std::atomic_fetch_add_explicit(&nsols, 1U, std::memory_order_relaxed);
  #else
    u32 soli = nsols++;
  #endif
    for (word_t block = 0; block < NEDGES; block += 64) {
      u64 alive64 = alive->block(block);
      for (word_t nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
        u32 ffs = __builtin_ffsll(alive64);
        nonce += ffs; alive64 >>= ffs;
        edge e(sipnode_(&sip_keys, nonce, 0), sipnode_(&sip_keys, nonce, 1));
        if (cycle.find(e) != cycle.end()) {
          sols[soli][n++] = nonce;
  #ifdef SHOWSOL
          printf("e(%x)=(%x,%x)%c", nonce, e.first, e.second, n==PROOFSIZE?'\n':' ');
  #endif
          if (PROOFSIZE > 2)
            cycle.erase(e);
        }
        if (ffs & 64) break; // can't shift by 64
      }
    }
    assert(n==PROOFSIZE);
  }
};

typedef struct {
  u32 id;  // thread idx
  cuckoo_ctx *ctx;  // the ctx this thread belongs to
  pthread_t thread;
} thread_ctx;

u32 path(cuckoo_hash &cuckoo, word_t u, word_t *us) {
  u32 nu;
  for (nu = 0; u; u = cuckoo[u]) {
    if (nu >= MAXPATHLEN) {
      while (nu-- && us[nu] != u) ;
      if (!~nu)
        printf("maximum path length exceeded\n");
      else printf("illegal %4d-cycle\n", MAXPATHLEN-nu);
      pthread_exit(NULL);
    }
    us[nu++] = u;
  }
  return nu-1;
}

// vp is thread_ctx*
void *worker(void *vp) {
  thread_ctx *tp = (thread_ctx *)vp;
  cuckoo_ctx *ctx = tp->ctx;

  shrinkingset *alive = ctx->alive;
  // if (tp->id == 0) printf("initial size %d\n", NEDGES);
  for (u32 round=0; round < ctx->ntrims; round++) {
    // if (tp->id == 0) printf("round %2d partition sizes", round);
    for (u32 part = 0; part <= PART_MASK; part++) {
        // step 1: clear all counts: only let the 1st thread do this
        if (tp->id == 0) {
            ctx->nonleaf->clear(); // clear all counts
        }
        ctx->barrier();

        // step 2: each thread visit its lives edges, but can
        // update degree of any node (thus requires atomic operation)
        ctx->count_node_deg(tp->id,round&1,part);
        ctx->barrier();

        // step 3: each thread kill its edges, by refering the global nodes degree counters
        ctx->kill_leaf_edges(tp->id,round&1,part);
        ctx->barrier();
        // if (tp->id == 0) printf(" %c%d %d", "UV"[round&1], part, alive->count());
    }
    // if (tp->id == 0) printf("\n");
  }
  if (tp->id == 0) {
    u32 load = (u32)(100LL * alive->count() / CUCKOO_SIZE);
    printf("nonce %d: %d trims completed  final load %d%%\n", ctx->nonce, ctx->ntrims, load);
    if (load >= 90) {
      printf("overloaded! exiting...");
      pthread_exit(NULL);
    }
    ctx->cuckoo = new cuckoo_hash(ctx->nonleaf->bits);
  }
#ifdef SINGLECYCLING
  else pthread_exit(NULL);
#else
  ctx->barrier();
#endif
  cuckoo_hash &cuckoo = *ctx->cuckoo;
  word_t us[MAXPATHLEN], vs[MAXPATHLEN];
#ifdef SINGLECYCLING
  for (word_t block = 0; block < NEDGES; block += 64) {
#else
  for (word_t block = tp->id*64; block < NEDGES; block += ctx->nthreads*64) {
#endif
    u64 alive64 = alive->block(block);
    for (word_t nonce = block-1; alive64; ) { // -1 compensates for 1-based ffs
      u32 ffs = __builtin_ffsll(alive64);
      nonce += ffs; alive64 >>= ffs;
      word_t u0=sipnode_(&ctx->sip_keys, nonce, 0), v0=sipnode_(&ctx->sip_keys, nonce, 1);
      if (u0) {// ignore vertex 0 so it can be used as nil for cuckoo[]
        u32 nu = path(cuckoo, u0, us), nv = path(cuckoo, v0, vs);
        if (us[nu] == vs[nv]) {
          u32 min = nu < nv ? nu : nv;
          for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) ;
          u32 len = nu + nv + 1;
          printf("%4d-cycle found at %d:%d%%\n", len, tp->id, (u32)(nonce*100LL/NEDGES));
          if (len == PROOFSIZE && ctx->nsols < ctx->maxsols)
            ctx->solution(us, nu, vs, nv);
        } else if (nu < nv) {
          while (nu--)
            cuckoo.set(us[nu+1], us[nu]);
          cuckoo.set(u0, v0);
        } else {
          while (nv--)
            cuckoo.set(vs[nv+1], vs[nv]);
          cuckoo.set(v0, u0);
        }
      }
      if (ffs & 64) break; // can't shift by 64
    }
  }
  pthread_exit(NULL);
  return 0;
}
