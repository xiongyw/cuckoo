#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include "cyclebase.hpp"
#include <set>

/**
 * return the path length start from node `u0`
 *
 *  u0---->u1---->...---->uk
 *  ^      ^              ^
 *  |      |              |
 * us[0] us[1] ...      us[k]
 */
int cyclebase::path(u32 u0, u32 *us) const {
    int nu = 0;  // path length
    u32 u; // current node
    us[0] = u0;
    //printf("%s(u0=0x%08x), pathcount[u0]=%d\n", __FUNCTION__, u0, pathcount[u0]);
    for (u = u0; pathcount[u]; ) {
        pathcount[u]++;
#if (0)
        if (pathcount[u] > 1) {
            printf("######%s(), pathcount[%d]=%d\n", __FUNCTION__, u, pathcount[u]);
        }
#endif
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
    //printf("%s(u0=0x%08x)=%d, MAXPATHLEN=%d\n", __FUNCTION__, u0, nu, MAXPATHLEN);
    return nu;
}

// return the number of extra common nodes (except the last one) of the two pathes
int cyclebase::pathjoin(u32 *us, int *pnu, u32 *vs, int *pnv) {
    int nu = *pnu, nv = *pnv;
    int min = nu < nv ? nu : nv;
    for (nu -= min, nv -= min; us[nu] != vs[nv]; nu++, nv++) min--;
    *pnu = nu; *pnv = nv;
    return min;
}

void cyclebase::addedge(u32 u0, u32 v0) {
    u32 u = u0 << 1, v = (v0 << 1) | 1;
    int nu = path(u, us), nv = path(v, vs);
    printf("%s(), u=%08x, v=%08x, nu=%d, nv=%d\n", __FUNCTION__, u, v, nu, nv);
    if (us[nu] == vs[nv]) {  // roots are the same
        printf("%s(), us[nu]==vs[nv]\n", __FUNCTION__);
        pathjoin(us, &nu, vs, &nv);
        int len = nu + nv + 1;
        printf("a % 4d-cycle found\n", len);
        cycleedges[ncycles].u = u;
        cycleedges[ncycles].v = v;
        cyclelengths[ncycles++] = len;
        if (len == PROOFSIZE)
            solution(us, nu, vs, nv);
        assert(ncycles < MAXCYCLES);
    } else if (nu < nv) {
        printf("%s(), nu<nv\n", __FUNCTION__);
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
#if (0)
        if (pathcount[vs[nv]] > 1) {
            printf("%s(), pathcount[vs[nv]]=%d\n", __FUNCTION__, pathcount[vs[nv]]);
            exit(0);
        }
#endif
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

void cyclebase::solution(u32 *us, int nu, u32 *vs, int nv) {
    printf("%s(nu=%d, nv=%d): Nodes", __FUNCTION__, nu, nv);
    u32 ni = 0;
    u32 u, v;
    recordedge(ni++, *us, *vs);
    printf("------");
    while (nu--) {
        u = (nu+1)&~1;
        v = nu|1;
        printf(" nu=%d, u=%d, v=%d", nu, u, v);
        recordedge(ni++, us[(nu+1)&~1], us[nu|1]); // u's in even position; v's in odd
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

void cyclebase::cycles() {
    int len, len2;
    word_t us2[MAXPATHLEN], vs2[MAXPATHLEN];
    for (int i=0; i < ncycles; i++) {
        word_t u = cycleedges[i].u, v = cycleedges[i].v;
        int   nu = path(u, us),    nv = path(v, vs);
        word_t root = us[nu]; assert(root == vs[nv]);
        int i2 = prevcycle[i] = cuckoo[root];
        cuckoo[root] = i;
        if (i2 < 0) continue;
        int rootdist = pathjoin(us, &nu, vs, &nv);
        do  {
            printf("chord found at cycleids %d %d\n", i2, i);
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

