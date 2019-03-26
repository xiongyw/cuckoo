#include <stdint.h> // for types uint32_t,uint64_t
#include <string.h> // for functions strlen, memset
#include <stdarg.h>
#include <stdio.h>
#include <chrono>
#include <ctime>
#include "../crypto/blake2.h"
#include "../crypto/siphash.hpp"

#ifdef SIPHASH_COMPAT
#include <stdio.h>
#endif

#include "cuckoo.h"

// last error reason, to be picked up by stats
// to be returned to caller
char LAST_ERROR_REASON[MAX_NAME_LEN];
const char *errstr[] = { "OK", "wrong header length", "edge too big", "edges not ascending", "endpoints don't match up", "branch in cycle", "cycle dead ends", "cycle too short"};

// generate edge endpoint in cuckoo graph without partition bit
word_t sipnode(siphash_keys *keys, word_t edge, u32 uorv) {
  return keys->siphash24(2*edge + uorv) & EDGEMASK;
}

// verify that edges are ascending and form a cycle in header-generated graph
int verify(word_t edges[PROOFSIZE], siphash_keys *keys) {
  word_t uvs[2*PROOFSIZE];
  word_t xor0 = 0, xor1 = 0;
  for (u32 n = 0; n < PROOFSIZE; n++) {
    if (edges[n] > EDGEMASK)
      return POW_TOO_BIG;
    if (n && edges[n] <= edges[n-1])
      return POW_TOO_SMALL;
    xor0 ^= uvs[2*n  ] = sipnode(keys, edges[n], 0);
    xor1 ^= uvs[2*n+1] = sipnode(keys, edges[n], 1);
  }
  if (xor0|xor1)              // optional check for obviously bad proofs
    return POW_NON_MATCHING;
  u32 n = 0, i = 0, j;
  do {                        // follow cycle
    for (u32 k = j = i; (k = (k+2) % (2*PROOFSIZE)) != i; ) {
      if (uvs[k] == uvs[i]) { // find other edge endpoint identical to one at i
        if (j != i)           // already found one before
          return POW_BRANCH;
        j = k;
      }
    }
    if (j == i) return POW_DEAD_END;  // no matching endpoint
    i = j^1;
    n++;
  } while (i != 0);           // must cycle back to start or we would have found branch
  return n == PROOFSIZE ? POW_OK : POW_SHORT_CYCLE;
}

// convenience function for extracting siphash keys from header
void setheader(const char *header, const u32 headerlen, siphash_keys *keys) {
  char hdrkey[32];
  // SHA256((unsigned char *)header, headerlen, (unsigned char *)hdrkey);
  blake2b((void *)hdrkey, sizeof(hdrkey), (const void *)header, headerlen, 0, 0);
#ifdef SIPHASH_COMPAT
  u64 *k = (u64 *)hdrkey;
  u64 k0 = k[0];
  u64 k1 = k[1];
  printf("k0 k1 %lx %lx\n", k0, k1);
  k[0] = k0 ^ 0x736f6d6570736575ULL;
  k[1] = k1 ^ 0x646f72616e646f6dULL;
  k[2] = k0 ^ 0x6c7967656e657261ULL;
  k[3] = k1 ^ 0x7465646279746573ULL;
#endif
  keys->setkeys(hdrkey);
}

// edge endpoint in cuckoo graph with partition bit
word_t sipnode_(siphash_keys *keys, word_t edge, u32 uorv) {
  return sipnode(keys, edge, uorv) << 1 | uorv;
}

u64 timestamp() {
        using namespace std::chrono;
        high_resolution_clock::time_point now = high_resolution_clock::now();
        auto dn = now.time_since_epoch();
        return dn.count();
}


void print_log(const char *fmt, ...) {
        if (SQUASH_OUTPUT) return;
        va_list args;
        va_start(args, fmt);
        vprintf(fmt, args);
        va_end(args);
}
