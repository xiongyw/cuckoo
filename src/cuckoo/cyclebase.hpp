#ifndef __CYCLEBASE_HPP__
#define __CYCLEBASE_HPP__

#include <utility>
#include <stdio.h>
#include <assert.h>
#include <set>
#include "cuckoo.h"

#ifndef MAXCYCLES
#define MAXCYCLES 64 // single byte
#endif

struct edge {
  u32 u;
  u32 v;
  edge() : u(0), v(0) { }
  edge(u32 x, u32 y) : u(x), v(y) { }
};

struct cyclebase {
  // should avoid different values of MAXPATHLEN in different threads of one process
  static const u32 MAXPATHLEN = 16 << (EDGEBITS/3);

  int ncycles;
  word_t *cuckoo;  // node array
  u32 *pathcount;  // see comments at <https://github.com/tromp/cuckoo/issues/90>
  edge cycleedges[MAXCYCLES];
  u32 cyclelengths[MAXCYCLES];
  u32 prevcycle[MAXCYCLES];
  u32 us[MAXPATHLEN];
  u32 vs[MAXPATHLEN];

  void alloc() {
    cuckoo = (word_t *)calloc(NCUCKOO, sizeof(word_t));
    pathcount = (u32 *)calloc(NCUCKOO, sizeof(u32));
  }

  void freemem() { // not a destructor, as memory may have been allocated elsewhere, bypassing alloc()
    free(cuckoo);
    free(pathcount);
  }

  void reset() {
    resetcounts();
  }

  void resetcounts() {
    memset(cuckoo, -1, NCUCKOO * sizeof(word_t)); // for prevcycle nil
    memset(pathcount, 0, NCUCKOO * sizeof(u32));
    ncycles = 0;
  }

  // return the path length start from node `u0`
  int path(u32 u0, u32 *us) const;

  // return the number of extra common nodes (except the last one) of the two pathes
  int pathjoin(u32 *us, int *pnu, u32 *vs, int *pnv) const;

  void addedge(u32 u0, u32 v0) ;

  void recordedge(const u32 i, const u32 u, const u32 v);

  void solution(u32 *us, int nu, u32 *vs, int nv);
  int sharedlen(u32 *us, int nu, u32 *vs, int nv);

  void cycles() ;
};


#endif // __CYCLEBASE_HPP__
