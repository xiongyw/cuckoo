#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include "cyclebase.hpp"
#include <set>

#define CYCLEBASE_VERBOSE      0

/**
 * return the path length start from node `u0`
 *
 * cuckoo[] idx: u0---->u1---->...---->uk
 *               ^      ^              ^
 *               |      |              |
 *              us[0] us[1] ...      us[k]
 */
int cyclebase::path(u32 u0, u32 *us) const {
    int nu = 0;  // path length
    u32 u; // current node
    us[0] = u0;
    for (u = u0; pathcount[u]; ) {
        pathcount[u]++;
        u = cuckoo[u];  // next node
        if (++nu >= (int)MAXPATHLEN) {
            while (nu-- && us[nu] != u) ;
            if (nu < 0)
                printf("maximum path length exceeded\n");
            else printf("illegal % 4d-cycle from node %d\n", MAXPATHLEN-nu, u0);
            exit(0);
        }
        us[nu] = u;
    }
    return nu;
}

// return the number of extra common nodes (except the first one) of the two pathes.
// *pnu and *pnv returns the number of edges (==number_of_nodes -1) in us[] and vs[].
// it does not actually alter the graph (i.e., join the two pathes)
int cyclebase::pathjoin(u32 *us, int *pnu, u32 *vs, int *pnv) const {
    int nu = *pnu, nv = *pnv;
    int min = nu < nv ? nu : nv;
    for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) min--;
    *pnu = nu; *pnv = nv;
    return min;
}

void cyclebase::addedge(u32 u0, u32 v0) {
    u32 u = u0 << 1, v = (v0 << 1) | 1;
    int nu = path(u, us), nv = path(v, vs);
    if (us[nu] == vs[nv]) {
#if CYCLEBASE_VERBOSE
        printf("before join, ns[%d]: ", nu);
        for (int i = 0; i <= nu; i ++) {
            printf("%x ", us[i]);
        }
        printf("\nbefore join, vs[%d]: ", nv);
        for (int i = 0; i <= nv; i ++) {
            printf("%x ", vs[i]);
        }
        printf("\n");
#endif
        // roots are the same, the current edge(u, v) is a cycle-forming edge, but
        // to keep the graph a directed forest, this edge is not to be added into the graph
        pathjoin(us, &nu, vs, &nv);
#if CYCLEBASE_VERBOSE
        printf(" after join, ns[%d]: ", nu);
        for (int i = 0; i <= nu; i ++) {
            printf("%x ", us[i]);
        }
        printf("\n after join, vs[%d]: ", nv);
        for (int i = 0; i <= nv; i ++) {
            printf("%x ", vs[i]);
        }
        printf("\n");
#endif
        int len = nu + nv + 1;
        printf("%d: %4d-cycle found\n", ncycles, len);

        // record the current cycle
        cycleedges[ncycles].u = u;
        cycleedges[ncycles].v = v;
        cyclelengths[ncycles++] = len;

        if (len == PROOFSIZE) {
            // print the solution
            solution(us, nu, vs, nv);
        }
        assert(ncycles < MAXCYCLES);
    } else if (nu < nv) {
        /* $6 in the paper: add the 8th edge: edge(u=10,v=11): 10->11
         *
         * nu = 1, ns: 11,12
         * nv = 2, vs: 10,5,8
         */
        pathcount[us[nu]]++;

        // reverse the shorter one
        while (nu--) {
            cuckoo[us[nu+1]] = us[nu];
        }
        // join the two path: shorter->longer
        cuckoo[u] = v;
    } else {  // nu !< nv
        pathcount[vs[nv]]++;
        // reverse vs[]
        while (nv--) {
            cuckoo[vs[nv+1]] = vs[nv];
        }
        // join the two path: u<---reversed(v)
        cuckoo[v] = u;
    }
}

void cyclebase::recordedge(const u32 i, const u32 u, const u32 v) {
    printf(" (%x,%x)", u, v);
}

// print the edges(u, v) of a cycle, given two cycle-forming-pathes:
// - us[nu] == vs[nv] is the same node
// - (us[0], vs[0]) is the new edge joining two pathes
// note that edges are print always with even node before the corresponding odd node.
void cyclebase::solution(u32 *us, int nu, u32 *vs, int nv) {
    printf("%s(nu=%d, nv=%d): Nodes", __FUNCTION__, nu, nv);
    u32 ni = 0;
    u32 u, v;

    recordedge(ni++, *us, *vs);  // edge(us[0], vs[0])

    printf("------");
    while (nu--) {
        // in us[], even indices store even nodes, odd indices store odd nodes
        // if (nu is odd) {
        //   u = nu + 1; v = nu;       <----
        // } else {
        //   u = nu;     v = nu + 1;   ---->
        // }
        u = (nu+1)&~1;
        v = nu|1;
        printf(" nu=%d, u=%d, v=%d", nu, u, v);
        //recordedge(ni++, us[(nu+1)&~1], us[nu|1]); // u's in even position; v's in odd
        recordedge(ni++, us[u], us[v]);
    }
    printf("------");
    while (nv--) {
        u = nv|1;
        v = (nv+1)&~1;
        printf(" nv=%d, u=%d, v=%d", nv, u, v);
        recordedge(ni++, vs[nv|1], vs[(nv+1)&~1]); // u's in odd position; v's in even
    }
    printf("\n");
#if 0
    for (u32 nonce = n = 0; nonce < NEDGES; nonce++) {
        edge e(2*sipnode(&sip_keys, nonce, 0), 2*sipnode(&sip_keys, nonce, 1)+1);
        if (cycle.find(e) != cycle.end()) {
            printf(" %x", nonce);
            cycle.erase(e);
        }
    }
    printf("\n");
#endif
}

int cyclebase::sharedlen(u32 *us, int nu, u32 *vs, int nv) {
    int len = 0;
    for (; nu-- && nv-- && us[nu] == vs[nv]; len++) ;
    return len;
}

/*
 * this is probably for finding more cycles than just a cycle base, which is
 * the "dead-end approach" mentioned in <https://github.com/tromp/cuckoo/issues/90>.
 * so we can safely ignore it for now...
 */
void cyclebase::cycles() {
    int len, len2;
    word_t us2[MAXPATHLEN], vs2[MAXPATHLEN];
    for (int i=0; i < ncycles; i++) {
        word_t u = cycleedges[i].u, v = cycleedges[i].v;
        int   nu = path(u, us),    nv = path(v, vs);
        /*
         * the current root (r1) may not be the previous root (r0) during addedge(); however,
         * the previous root (r0) must be still there.
         * u-->u-->u
         *            >---> r0 --> ... --> r1 --------> i
         *     v-->v
         */
        word_t root = us[nu]; assert(root == vs[nv]);

        prevcycle[i] = cuckoo[root];
        int i2 = cuckoo[root];

        /*
         * effectively we add manually an edge here, but as we don't update pathcount[root],
         * so path() will not follow this link.
         */
        cuckoo[root] = i;

#if CYCLEBASE_VERBOSE
        printf("\n%s(): i=%d, u=%x, nu=%d, v=%x, nv=%d; root=%x, cuckoo[root]=%x\n", __FUNCTION__, i, u, nu, v, nv, root, i2);
        printf("ns[%d]: ", nu);
        for (int i = 0; i <= nu; i ++) {
            printf("%x ", us[i]);
        }
        printf("\nvs[%d]: ", nv);
        for (int i = 0; i <= nv; i ++) {
            printf("%x ", vs[i]);
        }
        printf("\n");
#endif
        if (i2 < 0) { // the current root has outdegree 0
            continue;
        }

        int rootdist = pathjoin(us, &nu, vs, &nv);

        do  {
            printf("i=%d: chord found at cycle ids: %d %d\n", i, i, i2);
            word_t u2 = cycleedges[i2].u, v2 = cycleedges[i2].v;
            int nu2 = path(u2, us2), nv2 = path(v2, vs2);
            word_t root2 = us2[nu2]; assert(root2 == vs2[nv2] && root == root2);
            int rootdist2 = pathjoin(us2, &nu2, vs2, &nv2);
            if (us[nu] == us2[nu2]) {
                len  = sharedlen(us,nu,us2,nu2) + sharedlen(us,nu,vs2,nv2);
                len2 = sharedlen(vs,nv,us2,nu2) + sharedlen(vs,nv,vs2,nv2);
                if (len + len2 > 0) {
#if 0
                    word_t ubranch = us[nu-len], vbranch = vs[nv-len2];
                    addpath(ubranch, vbranch, len+len2);
                    addpath(ubranch, vbranch, len+len2);
#endif
                    printf(" % 4d-cycle found At %d%%\n", cyclelengths[i] + cyclelengths[i2] - 2*(len+len2), (int)(i*100L/ncycles));
                }
            } else {
                int rd = rootdist - rootdist2;
                if (rd < 0) {
                    if (nu+rd > 0 && us2[nu2] == us[nu+rd]) {
                        int len = sharedlen(us,nu+rd,us2,nu2) + sharedlen(us,nu+rd,vs2,nv2);
                        if (len) printf(" % 4d-cycle found At %d%%\n", cyclelengths[i] + cyclelengths[i2] - 2*len, (int)(i*100L/ncycles));
                    } else if (nv+rd > 0 && vs2[nv2] == vs[nv+rd]) {
                        int len = sharedlen(vs,nv+rd,us2,nu2) + sharedlen(vs,nv+rd,vs2,nv2);
                        if (len) printf(" % 4d-cycle found At %d%%\n", cyclelengths[i] + cyclelengths[i2] - 2*len, (int)(i*100L/ncycles));
                    }
                } else if (rd > 0) {
                    if (nu2-rd > 0 && us[nu] == us2[nu2-rd]) {
                        int len = sharedlen(us2,nu2-rd,us,nu) + sharedlen(us2,nu2-rd,vs,nv);
                        if (len) printf(" % 4d-cycle found At %d%%\n", cyclelengths[i] + cyclelengths[i2] - 2*len, (int)(i*100L/ncycles));
                    } else if (nv2-rd > 0 && vs[nv] == vs2[nv2-rd]) {
                        int len = sharedlen(vs2,nv2-rd,us,nu) + sharedlen(vs2,nv2-rd,vs,nv);
                        if (len) printf(" % 4d-cycle found At %d%%\n", cyclelengths[i] + cyclelengths[i2] - 2*len, (int)(i*100L/ncycles));
                    }
                } // else cyles are disjoint
            }
        } while ((i2 = prevcycle[i2]) >= 0);
    }
}

