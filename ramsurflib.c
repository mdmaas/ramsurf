/*
 * Copyright (c) 2006, Dr. David C. Calvo <calvo@ccs.nrl.navy.mil> Copyright
 * (c) 2012, Serge Guelton <serge.guelton@quiet-oceans.com>
 * 
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *     Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *     Neither the name of the <ORGANIZATION> nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * 
 */

#include "ramsurf.h"

#include <stdio.h>
#include <stddef.h>
#include <complex.h>
#include <math.h>
#include <float.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <setjmp.h>

#ifdef __SSE3__
#include <xmmintrin.h>
#include <pmmintrin.h>
#endif


#ifndef max
   #define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
   #define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

/* M_PI dropped in c99 */
#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif




static const size_t m = 40;

typedef ram_cplx fcomplex;
typedef double complex dcomplex;

#if defined(RAMSURF_DOUBLE) && (defined(RAMSURF_PASCAL_TDMA) || defined(RAMSURF_OPENMP_PCR))
/* PoC switch: replace the fork's factored substitution (matrc LU + solve) with
   a raw complex tridiagonal solved by an external kernel. Double precision only,
   where ram_cplx == double complex. Kernel: the OpenMP PCR (pcr_omp.c) when
   RAMSURF_OPENMP_PCR, otherwise PaScaL_TDMA's serial complex Thomas (tdmas.f90). */
#define RAMSURF_USE_RAW_TDMA 1
#endif

#if defined(RAMSURF_OPENMP_PCR)
extern void pcr_tdma_complex(int n, double complex *a, double complex *b,
                             double complex *c, double complex *d);
#elif defined(RAMSURF_USE_RAW_TDMA)
/* In-place; (a,b,c,d) are consumed on return. */
extern void tdma_single_(ram_cplx *a, ram_cplx *b, ram_cplx *c, ram_cplx *d, int *n1);
#endif


static jmp_buf exception_env;

/* output management { */
typedef struct {
    ram_real ** buffer;
    ram_real ** current;
    ram_real ** end;
} output_t;

static
void output_init(output_t* o) {
    static const int init_size = 32;
    o->buffer = malloc(sizeof(*o->buffer)*init_size);
    o->current = o->buffer;
    o->end = o->current + init_size;
}
static
void output_destroy(output_t* o) {
    ram_real ** iter;
    for(iter = o->buffer; iter < o->current; ++iter)
        free(*iter);
    free(o->buffer);
}

static
ram_real** output_release(output_t* o) {
    int old_size = o->current - o->buffer;
    int new_size = old_size +1 ;
    o->buffer = realloc(o->buffer, new_size * sizeof(*o->buffer));
    o->buffer[old_size] = NULL;
    return o-> buffer;
}

static
void output_push_back( output_t* o, ram_real const * data, int count, ram_real r) {
    if( o->current == o->end) {
        int old_size = o->end - o->buffer;
        int new_size = old_size << 1;
        o->buffer = realloc(o->buffer, new_size * sizeof(*o->buffer));
        if(!o->buffer) {
    	    longjmp(exception_env, NOT_ENOUGH_MEMORY);
        }
        o->current = o->buffer + old_size;
        o->end = o->buffer + new_size;
    }
    *o->current = malloc((1+count) * sizeof(*data));
    (*o->current)[0] = r;                                   // first ram_real is the range
    memcpy((*o->current) + 1, data, count * sizeof(*data)); // others are the data
    ++o->current;
}
/* } */

//
//     This subroutine finds a root of a polynomial of degree n > 2
//     by Laguerre's method.
//
static
void guerre(double a[m][2], int n, double z[2], double err, int const nter) {
    double az[50][2],azz[50][2],dz[2],p[2],pz[2],pzz[2],f[2],g[2],h[2];
    double tmp;
    const double eps=1.0e-20;
    const double rn=n;


    //
    //     The coefficients of p'[z] and p''[z].
    //
    for(int i=1;i<=n;i++) {
        az[i-1][0]=i*a[i][0];
        az[i-1][1]=i*a[i][1];
    }
    for(int i=1;i<=n-1;i++) {
        azz[i-1][0]=i*az[i][0];
        azz[i-1][1]=i*az[i][1];
    }
    //
    int iter=0,jter=0;
    do {
        p[0]=a[n-1][0]+(a[n][0]*z[0]-a[n][1]*z[1]);
        p[1]=a[n-1][1]+(a[n][0]*z[1]+a[n][1]*z[0]);
        for(int i=n-1;i>=1;--i) {
            tmp=p[0];
            p[0]=a[i-1][0]+(z[0]*p[0]-z[1]*p[1]);
            p[1]=a[i-1][1]+(z[0]*p[1]+z[1]*tmp);
        }
        tmp=(p[0]*p[0]+p[1]*p[1]);
        if(sqrt(tmp)<eps) return;
        //
        pz[0]=az[n-2][0]+(az[n-1][0]*z[0]-az[n-1][1]*z[1]);
        pz[1]=az[n-2][1]+(az[n-1][0]*z[1]+az[n-1][1]*z[0]);
        for(int i=n-2;i>=1;i--) {
            tmp=pz[0];
            pz[0]=az[i-1][0]+(z[0]*pz[0]-z[1]*pz[1]);
            pz[1]=az[i-1][1]+(z[0]*pz[1]+z[1]*tmp);
        }
        //
        pzz[0]=azz[n-3][0]+(azz[n-2][0]*z[0]-azz[n-2][1]*z[1]);
        pzz[1]=azz[n-3][1]+(azz[n-2][0]*z[1]+azz[n-2][1]*z[0]);
        for(int i=n-3;i>=1;i--) {
            tmp=pzz[0];
            pzz[0]=azz[i-1][0]+(z[0]*pzz[0]-z[1]*pzz[1]);
            pzz[1]=azz[i-1][1]+(z[0]*pzz[1]+z[1]*tmp);
        }
        //
        //     The Laguerre perturbation.
        //
        double norm=(p[0]*p[0] + p[1]*p[1]);
        f[0]=(pz[0]*p[0] + pz[1]*p[1])/norm;
        f[1]=(pz[1]*p[0] - pz[0]*p[1])/norm;

        g[0]=(f[0]*f[0]-f[1]*f[1]) - (pzz[0]*p[0] + pzz[1]*p[1])/norm;
        g[1]=(f[0]*f[1]+f[1]*f[0]) - (pzz[1]*p[0] - pzz[0]*p[1])/norm;
        dcomplex H = rn*g[0] - (f[0]*f[0]-f[1]*f[1]) + I*( rn*g[1] - (f[0]*f[1]+f[1]*f[0]));
        H=csqrt((rn-1.0)*H);
        h[0]=creal(H);
        h[1]=cimag(H);
        double fph[2] = { f[0] + h[0], f[1] + h[1] };
        double fmh[2] = { f[0] - h[0], f[1] - h[1] };
        double amp1=fph[0]*fph[0] + fph[1]*fph[1];
        double amp2=fmh[0]*fmh[0] + fmh[1]*fmh[1];
        if(amp1>amp2) {
            dz[0]=-rn*fph[0]/amp1;
            dz[1]=+rn*fph[1]/amp1;
        }
        else {
            dz[0]=-rn*fmh[0]/amp2;
            dz[1]=+rn*fmh[1]/amp2;
        }
        //
        iter=iter+1;
        //
        //     Rotate by 90 degrees to avoid limit cycles. 
        //
        jter=jter+1;
        if(jter==10) {
            jter=1;
            tmp=dz[0];
            dz[0]=-dz[1];
            dz[1]=tmp;
        }
        z[0]+=dz[0];
        z[1]+=dz[1];
        //
        if(iter==100) {
    	    longjmp(exception_env, LAG_NOT_CON);
        }
        //
    } while ((sqrt(dz[0]*dz[0]+dz[1]*dz[1])>err) && (iter<nter));
    //
}

//
//     The root-finding subroutine. 
//
static
void fndrt(double a[m][2], int n, double z[m][2]) {
    double root[2];
    //
    if(n==1) {
        double norm = a[1][0]*a[1][0] +  a[1][1]*a[1][1];
        z[0][0]=-(a[0][0]*a[1][0]+a[0][1]*a[1][1])/norm;
        z[0][1]=-(a[0][1]*a[1][0]-a[0][0]*a[1][1])/norm;
        return;
    }
    if(n>2) {
        //
        for(int k=n;k>=3;--k) {
            //
            //     Obtain an approximate root.
            //
            root[0]=root[1]=0.;
            guerre(a,k,root, 1.0e-12, 1000);
            //
            //     Refine the root by iterating five more times.
            //
            guerre(a,k,root,0.,5);
            z[k-1][0]=root[0];
            z[k-1][1]=root[1];
            //
            //     Divide out the factor [z-root].
            //
            for(int i=k;i>=1;i--) {
                a[i-1][0]+=root[0]*a[i][0] - root[1]*a[i][1];
                a[i-1][1]+=root[0]*a[i][1] + root[1]*a[i][0];
            }
            for(int i=1;i<=k;i++) {
                a[i-1][0]=a[i][0];
                a[i-1][1]=a[i][1];
            }
            //
        }
    }
    //
    //     Solve the quadratic equation.
    //
    dcomplex A0 = a[0][0]+I*a[0][1];
    dcomplex A1 = a[1][0]+I*a[1][1];
    dcomplex A2 = a[2][0]+I*a[2][1];
    dcomplex SQRT = csqrt(A1*A1-4.0*A0*A2);
    dcomplex Z1 = 0.5*((-1)*A1+SQRT)/A2;
    dcomplex Z0 = 0.5*((-1)*A1-SQRT)/A2;
    z[1][0]=creal(Z1);
    z[1][1]=cimag(Z1);
    z[0][0]=creal(Z0);
    z[0][1]=cimag(Z0);
    //
}

//
//     Rows are interchanged for stability.
//
static
void pivot(int const n, int const i, double a[m][m][2], double b[m][2]){
    double temp[2];
    //
    int i0=i;
    double amp0=sqrt(a[i-1][i-1][0]*a[i-1][i-1][0]+a[i-1][i-1][1]*a[i-1][i-1][1]);
    for(int j=i+1;j<=n;j++) {
        double amp=sqrt(a[i-1][j-1][0]*a[i-1][j-1][0]+a[i-1][j-1][1]*a[i-1][j-1][1]);
        if(amp>amp0) {
            i0=j;
            amp0=amp;
        }
    }
    if(i0==i)
        return;
    //
    temp[0]=b[i-1][0];
    temp[1]=b[i-1][1];
    b[i-1][0]=b[i0-1][0];
    b[i-1][1]=b[i0-1][1];
    b[i0-1][0]=temp[0];
    b[i0-1][1]=temp[1];
    for(int j=i;j<=n;j++) {
        temp[0]=a[j-1][i-1][0];
        temp[1]=a[j-1][i-1][1];
        a[j-1][i-1][0]=a[j-1][i0-1][0];
        a[j-1][i-1][1]=a[j-1][i0-1][1];
        a[j-1][i0-1][0]=temp[0];
        a[j-1][i0-1][1]=temp[1];
    }
    //
}
//
//     Gaussian elimination.
//
static
void gauss(int const n, double a[m][m][2], double b[m][2]) {
    //
    //     Downward elimination.
    //
    for(int i=1;i<=n;i++) {
        if(i<n)
            pivot(n,i,a,b);
        double norm = a[i-1][i-1][0]*a[i-1][i-1][0]+a[i-1][i-1][1]*a[i-1][i-1][1];
        a[i-1][i-1][0]=(a[i-1][i-1][0])/norm;
        a[i-1][i-1][1]=-a[i-1][i-1][1]/norm;
        double tmp = b[i-1][0];
        b[i-1][0]=b[i-1][0]*a[i-1][i-1][0]-b[i-1][1]*a[i-1][i-1][1];
        b[i-1][1]=b[i-1][1]*a[i-1][i-1][0]+tmp*a[i-1][i-1][1];
        if(i<n){
            for(int j=i+1;j<=n;j++) {
                tmp = a[j-1][i-1][0];
                a[j-1][i-1][0]=a[j-1][i-1][0]*a[i-1][i-1][0]-a[j-1][i-1][1]*a[i-1][i-1][1];
                a[j-1][i-1][1]=a[j-1][i-1][1]*a[i-1][i-1][0]+tmp*a[i-1][i-1][1];
            }
            for(int k=i+1;k<=n;k++) {
                b[k-1][0]-=a[i-1][k-1][0]*b[i-1][0]-a[i-1][k-1][1]*b[i-1][1];
                b[k-1][1]-=a[i-1][k-1][1]*b[i-1][0]+a[i-1][k-1][0]*b[i-1][1];
                for(int j=i+1;j<=n;j++) {
                    a[j-1][k-1][0]-=a[i-1][k-1][0]*a[j-1][i-1][0]-a[i-1][k-1][1]*a[j-1][i-1][1];
                    a[j-1][k-1][1]-=a[i-1][k-1][1]*a[j-1][i-1][0]+a[i-1][k-1][0]*a[j-1][i-1][1];
                }
            }
        }
    }
    //
    //     Back substitution.
    //
    for(int i=n-1;i>=1;i--) {
        for(int j=i;j<n;j++) {
            b[i-1][0]-=a[j][i-1][0]*b[j][0]-a[j][i-1][1]*b[j][1];
            b[i-1][1]-=a[j][i-1][1]*b[j][0]+a[j][i-1][0]*b[j][1];
        }
    }
    //
}

//
//     The derivatives of the operator function at x=0.
//
static
void deriv(int n, ram_real sig, double alp, 
        double dg[m][2], double dh1[m][2], double dh2[m][2], double dh3[m][2],
        double bin[m][m], double nu) {
    //
    dh1[0][0]=0;
    dh1[0][1]=sig*0.5;
    double exp1 = -0.5;
    dh2[0][0]=alp;
    dh2[0][1]=0.;
    double exp2=-1.0;
    dh3[0][0]=-2.0*nu;
    dh3[0][1]=0.;
    double exp3=-1.0;
    for(int i=1;i<n;i++) {
        dh1[i][0]=dh1[i-1][0]*exp1;
        dh1[i][1]=dh1[i-1][1]*exp1;
        exp1=exp1-1.0;
        dh2[i][0]=dh2[i-1][0]*exp2;
        dh2[i][1]=dh2[i-1][1]*exp2;
        exp2=exp2-1.0;
        dh3[i][0]=-nu*dh3[i-1][0]*exp3;
        dh3[i][1]=-nu*dh3[i-1][1]*exp3;
        exp3=exp3-1.0;
    }
    //
    dg[0][0]=1.0;
    dg[0][1]=0.0;
    dg[1][0]=dh1[0][0]+dh2[0][0]+dh3[0][0];
    dg[1][1]=dh1[0][1]+dh2[0][1]+dh3[0][1];
    for(int i=2;i<=n;i++) {
        dg[i][0]=dh1[i-1][0]+dh2[i-1][0]+dh3[i-1][0];
        dg[i][1]=dh1[i-1][1]+dh2[i-1][1]+dh3[i-1][1];
        for(int j=1;j<=i-1;j++) {
            double tmpr=dh1[j-1][0]+dh2[j-1][0]+dh3[j-1][0];
            double tmpi=dh1[j-1][1]+dh2[j-1][1]+dh3[j-1][1];
            dg[i][0]+=bin[j-1][i-1]*(tmpr*dg[i-j][0]-tmpi*dg[i-j][1]);
            dg[i][1]+=bin[j-1][i-1]*(tmpi*dg[i-j][0]+tmpr*dg[i-j][1]);
        }
    }
    //
}

//
//     The coefficients of the rational approximation.
//
static
void epade(size_t mp, int np, int ns, int const ip, ram_real k0, ram_real dr,  ram_real pd1[mp][2],  ram_real pd2[mp][2]) {
    //
    double dg[m][2],dh1[m][2],dh2[m][2],dh3[m][2],a[m][m][2],b[m][2];
    double  z1, bin[m][m],fact[m];
    ram_real sig=k0*dr;
    int n=2*np;
    //
    double nu, alp;
    if(ip==1){
        nu=0.0;
        alp=0.0;
    }
    else {
        nu=1.0;
        alp=-0.25;
    }
    //
    //     The factorials.;
    //
    fact[0]=1.0;
    for(int i=1; i<n; i++)
        fact[i]=(i+1)*fact[i-1];
    //
    //     The binomial coefficients.;
    //
    for(int i=0;i<n+1;i++) {
        bin[0][i]=1.0;
        bin[i][i]=1.0;
    }
    for(int i=2;i<n+1;i++)
        for(int j=1;j<i;j++)
            bin[j][i]=bin[j-1][i-1]+bin[j][i-1];
    //
    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++) {
            a[i][j][0]=0.0;
            a[i][j][1]=0.0;
        }
    //
    //     The accuracy constraints.;
    //
    deriv(n, sig, alp, dg, dh1, dh2, dh3, bin, nu);
    //
    for(int i=0;i<n; i++) {
        b[i][0]=dg[i+1][0];
        b[i][1]=dg[i+1][1];
    }
    for(int i=1;i<=n;i++) {
        if(2*i-1<=n) {
            a[2*i-2][i-1][0]=fact[i-1];
            a[2*i-2][i-1][1]=0.;
        }
        for(int j=1;j<=i;j++) {
            if(2*j<=n) {
                a[2*j-1][i-1][0]=-bin[j][i]*fact[j-1]*dg[i-j][0];
                a[2*j-1][i-1][1]=-bin[j][i]*fact[j-1]*dg[i-j][1];
            }
        }
    }
    //
    //     The stability constraints.;
    //
    if(ns>=1){
        z1=-3.0;
        b[n-1][0]=-1.0;
        b[n-1][1]=0.0;
        for(int j=1;j<=np;j++) {
            a[2*j-2][n-1][0]=pow(z1,j);
            a[2*j-2][n-1][1]=0.;
            a[2*j-1][n-1][0]=0.0;
            a[2*j-1][n-1][1]=0.0;
        }
    }
    //
    if(ns>=2){
        z1=-1.5;
        b[n-2][0]=-1.0;
        b[n-2][1]=0.0;
        for(int j=1;j<=np;j++) {
            a[2*j-2][n-2][0]=pow(z1,j);
            a[2*j-2][n-2][1]=0.;
            a[2*j-1][n-2][0]=0.0;
            a[2*j-1][n-2][1]=0.0;
        }
    }
    //
    gauss(n,a, b);
    //
    dh1[0][0]=1.0;
    dh1[0][1]=0.0;
    for(int j=1;j<=np;j++) {
        dh1[j][0]=b[2*j-2][0];
        dh1[j][1]=b[2*j-2][1];
    }
    fndrt(dh1,np,dh2);
    for(int j=0;j<np;j++){
        double norm = dh2[j][0]*dh2[j][0]+dh2[j][1]*dh2[j][1];
        pd1[j][0]=-dh2[j][0]/norm;
        pd1[j][1]=+dh2[j][1]/norm;
    }
    //
    dh1[0][0]=1.0;
    dh1[0][1]=0.0;
    for(int j=1;j<=np;j++) {
        dh1[j][0]=b[2*j-1][0];
        dh1[j][1]=b[2*j-1][1];
    }
    fndrt(dh1,np,dh2);

    for(int j=0;j<np;j++) {
        double norm = dh2[j][0]*dh2[j][0]+dh2[j][1]*dh2[j][1];
        pd2[j][0]=-dh2[j][0]/norm;
        pd2[j][1]=+dh2[j][1]/norm;
    }
    //
}
//
//     Profile reader and interpolator.
//
static
void zread( ram_real const* data, size_t mz, int nz, ram_real dz, ram_real prof[mz]) {
    //
    for(int i=0;i<nz+2;i++)
        prof[i]=-1.0;
    ram_real zi = data[0], profi = data[1];
    //
    prof[0]=profi;
    int i=1.5+zi/dz;
    prof[i-1]=profi;
    int iold=i;
    int k=2;
    while(1) {
        zi = data[k++];
        profi = data[k++];
        if(zi<0.0) break;
        i=1.5+zi/dz;
        if(i == iold)i=i+1;
        prof[i-1]=profi;
        assert(i-1<mz);
        iold=i;
    }

    prof[nz+1]=prof[i-1];
    i=1;
    int j=1;
    do {
        do {
            i=i+1;
        } while (prof[i-1]<0.0);
        if(i-j!=1) {
            for(int k=j+1;k<=i-1;k++)
                prof[k-1]=prof[j-1]+((ram_real)(k-j))*(prof[i-1]-prof[j-1])/((ram_real)(i-j));
        }
        j=i;
    } while (j<nz+2);
}
//
//     Set up the profiles.
//
static
void profl(ramsurf_t const * const rsurf,       //< ramsurf input
           size_t const            profl_index, //< Index of the profile to extract
           size_t const            mz,          //< Size of array containing the extracted profiles
           int const               nz,          //< Number of depth points for the profile
           ram_real const             dz,          //< depth step
           ram_real const             omega,       //< Angular frequency (2 * PI * frequency)
           ram_real const             k0,          //< Wavenumber (omega / reference_celerity)
           // OUT parameters
           ram_real * const           rp,          //< Boundary of the extracted profile
           ram_real                   cw[mz],      //< Celerity profile in water
           ram_real                   cb[mz],      //< Celerity profile in bottom
           ram_real                   rhob[mz],    //< Density profile in bottom
           ram_real                   attn[mz],    //< Attenuation profile in the bottom
           ram_real                   alpw[mz],    //< Alpha profile in water
           ram_real                   alpb[mz],    //< Alpha profile in bottom
           ram_real                   ksqw[mz][2], //< Squared wavenumber in water
           ram_real                   ksqb[mz][2]  //< Squared wavenumber in bottom
          ) {
    //
    const ram_real eta=1.0/(40.0*M_PI*log10(exp(1.0)));
    zread(rsurf->cw[profl_index], mz, nz,dz,cw);
    zread(rsurf->cb[profl_index], mz, nz,dz,cb);
    zread(rsurf->rhob[profl_index], mz, nz,dz,rhob);
    zread(rsurf->attn[profl_index], mz, nz,dz,attn);
    *rp = rsurf->rp[profl_index];
    //
    for(int i=0; i< nz+2; i++) {
        ram_real rtmp = omega/cw[i], itmp;
        ksqw[i][0]= rtmp*rtmp - k0*k0;
        ksqw[i][1]= 0.f;
        rtmp = (omega/cb[i]);
        itmp = (omega/cb[i])*(eta*attn[i]);
        ksqb[i][0]=(rtmp*rtmp-itmp*itmp)-k0*k0;
        ksqb[i][1]=(rtmp*itmp+itmp*rtmp);
        alpw[i]=sqrt(cw[i]/rsurf->c0);
        alpb[i]=sqrt(rhob[i]*cb[i]/rsurf->c0);
    }
}

//
//     The tridiagonal matrices.
//
static
void matrc(size_t const mz,
           size_t const mp,
           int const    nz,
           int const    np,
           int const    iz,
           ram_real const  dz,
           ram_real const  k0,
           ram_real const  rhob[mz],    //< ^
           ram_real const  alpw[mz],    //< |
           ram_real const  alpb[mz],    //< | Profile description
           ram_real        ksq[mz][2],  //< |
           ram_real const  ksqw[mz][2], //< |
           ram_real const  ksqb[mz][2], //< v
           ram_real        f1[mz],
           ram_real        f2[mz],
           ram_real        f3[mz],
           ram_real        r1[mp][mz][2],
           ram_real        r2[mp][mz][2],
           ram_real        r3[mp][mz][2],
           ram_real        s1[mp][mz][2],
           ram_real        s2[mp][mz][2],
           ram_real        s3[mp][mz][2],
           ram_real        pd1[mp][2],
           ram_real        pd2[mz][2],
           int const    izsrf
          ) {
    ram_real d1[4] __attribute__((aligned(16))),d2[4] __attribute__((aligned(16))),d3[4] __attribute__((aligned(16))),rfact[2];
    //
    ram_real a1=k0*k0/6.0;
    ram_real a2=2.0*k0*k0/3.0;
    ram_real a3=k0*k0/6.0;
    ram_real cfact=0.5/(dz*dz);
    ram_real dfact=1.0/12.0;
    //
    for(int i=0;i<iz; i++) {
        f1[i]=1.0/alpw[i];
        f2[i]=1.0;
        f3[i]=alpw[i];
        ksq[i][0]=ksqw[i][0];
        ksq[i][1]=ksqw[i][1];
    }
    //
    for(int i=iz,ii=0;i<nz+2; i++,ii++) {
        f1[i]=rhob[ii]/alpb[ii];
        f2[i]=1.0/rhob[ii];
        f3[i]=alpb[ii];
        ksq[i][0]=ksqb[ii][0];
        ksq[i][1]=ksqb[ii][1];
    }

#if defined(__SSE3__) && !defined(RAMSURF_DOUBLE)
    //
    //     Discretization by Galerkin's method.
    //
    int ibound = nz/2;
    for(int i=1; i< ibound; i+=2) {
        ram_real c1= cfact*f1[i+0]*(f2[i-1]+f2[i])*f3[i-1];
        ram_real c2=-cfact*f1[i+0]*(f2[i-1]+2.0*f2[i]+f2[i+1])*f3[i];
        ram_real c3= cfact*f1[i+0]*(f2[i+0]+f2[i+1])*f3[i+1];
        ram_real c4= cfact*f1[i+1]*(f2[i+0]+f2[i+1])*f3[i+0];
        ram_real c5=-cfact*f1[i+1]*(f2[i+0]+2.0*f2[i+1]+f2[i+2])*f3[i+1];
        ram_real c6= cfact*f1[i+1]*(f2[i+1]+f2[i+2])*f3[i+2];

        d1[0]=c1+dfact*(ksq[i-1][0]+ksq[i+0][0]);
        d1[1]= 0+dfact*(ksq[i-1][1]+ksq[i+0][1]);
        d1[2]=c4+dfact*(ksq[i+0][0]+ksq[i+1][0]);
        d1[3]= 0+dfact*(ksq[i+0][1]+ksq[i+1][1]);

        d2[0]=c2+dfact*(ksq[i-1][0]+6.0*ksq[i+0][0]+ksq[i+1][0]);
        d2[1]= 0+dfact*(ksq[i-1][1]+6.0*ksq[i+0][1]+ksq[i+1][1]);
        d2[2]=c5+dfact*(ksq[i+0][0]+6.0*ksq[i+1][0]+ksq[i+2][0]);
        d2[3]= 0+dfact*(ksq[i+0][1]+6.0*ksq[i+1][1]+ksq[i+2][1]);

        d3[0]=c3+dfact*(ksq[i+0][0]+ksq[i+1][0]);
        d3[1]= 0+dfact*(ksq[i+0][1]+ksq[i+1][1]);
        d3[2]=c6+dfact*(ksq[i+1][0]+ksq[i+2][0]);
        d3[3]= 0+dfact*(ksq[i+1][1]+ksq[i+2][1]);
        //
        __m128 _a1 = _mm_load_ss(&a1); _a1=_mm_movelh_ps(_a1, _a1);
        __m128 _a2 = _mm_load_ss(&a2); _a2=_mm_movelh_ps(_a2, _a2);
        __m128 _a3 = _mm_load_ss(&a3); _a3=_mm_movelh_ps(_a3, _a3);
        __m128 _d1_imag = _mm_load_ps(&d1[0]);
        __m128 _d1_real = _mm_moveldup_ps(_d1_imag);
        _d1_imag = _mm_movehdup_ps(_d1_imag);
        __m128 _d2_imag = _mm_load_ps(&d2[0]);
        __m128 _d2_real = _mm_moveldup_ps(_d2_imag);
        _d2_imag = _mm_movehdup_ps(_d2_imag);
        __m128 _d3_imag = _mm_load_ps(&d3[0]);
        __m128 _d3_real = _mm_moveldup_ps(_d3_imag);
        _d3_imag = _mm_movehdup_ps(_d3_imag);
        __m128 xmm0, xmm1;
        for(int j=0;j<np;j++) {
            __m128 _pd1 = _mm_loadu_ps(&pd1[j][0]);
            _pd1 = _mm_movelh_ps(_pd1,_pd1);
            __m128 _pd1_shuf = _mm_shuffle_ps(_pd1,_pd1,_MM_SHUFFLE(2,3,0,1));
            __m128 _pd2 = _mm_loadu_ps(&pd2[j][0]);
            _pd2 = _mm_movelh_ps(_pd2,_pd2);
            __m128 _pd2_shuf = _mm_shuffle_ps(_pd2,_pd2,_MM_SHUFFLE(2,3,0,1));

            xmm0 = _mm_mul_ps(_pd1, _d1_real);
            xmm1 = _mm_mul_ps(_pd1_shuf, _d1_imag);
            xmm0 = _mm_addsub_ps(xmm0, xmm1);
            xmm0 = _mm_add_ps(xmm0, _a1);
            _mm_storeu_ps(&s1[j][i+0][0], xmm0);
            xmm0 = _mm_mul_ps(_pd2, _d1_real);
            xmm1 = _mm_mul_ps(_pd2_shuf, _d1_imag);
            xmm0 = _mm_addsub_ps(xmm0, xmm1);
            xmm0 = _mm_add_ps(xmm0, _a1);
            _mm_storeu_ps(&r1[j][i+0][0], xmm0);

            xmm0 = _mm_mul_ps(_pd1, _d2_real);
            xmm0 = _mm_mul_ps(_pd1, _d2_real);
            xmm1 = _mm_mul_ps(_pd1_shuf, _d2_imag);
            xmm0 = _mm_addsub_ps(xmm0, xmm1);
            xmm0 = _mm_add_ps(xmm0, _a2);
            _mm_storeu_ps(&s2[j][i+0][0], xmm0);
            xmm0 = _mm_mul_ps(_pd2, _d2_real);
            xmm1 = _mm_mul_ps(_pd2_shuf, _d2_imag);
            xmm0 = _mm_addsub_ps(xmm0, xmm1);
            xmm0 = _mm_add_ps(xmm0, _a2);
            _mm_storeu_ps(&r2[j][i+0][0], xmm0);


            xmm0 = _mm_mul_ps(_pd1, _d3_real);
            xmm1 = _mm_mul_ps(_pd1_shuf, _d3_imag);
            xmm0 = _mm_addsub_ps(xmm0, xmm1);
            xmm0 = _mm_add_ps(xmm0, _a3);
            _mm_storeu_ps(&s3[j][i+0][0], xmm0);
            xmm0 = _mm_mul_ps(_pd2, _d3_real);
            xmm1 = _mm_mul_ps(_pd2_shuf, _d3_imag);
            xmm0 = _mm_addsub_ps(xmm0, xmm1);
            xmm0 = _mm_add_ps(xmm0, _a3);
            _mm_storeu_ps(&r3[j][i+0][0], xmm0);
        }
    }
    for(int i=ibound; i< nz+1; i++) {
        ram_real c1=cfact*f1[i]*(f2[i-1]+f2[i])*f3[i-1];
        ram_real c2=-cfact*f1[i]*(f2[i-1]+2.0*f2[i]+f2[i+1])*f3[i];
        ram_real c3=cfact*f1[i]*(f2[i]+f2[i+1])*f3[i+1];
        d1[0]=c1+dfact*(ksq[i-1][0]+ksq[i][0]);
        d1[1]=dfact*(ksq[i-1][1]+ksq[i][1]);
        d2[0]=c2+dfact*(ksq[i-1][0]+6.0*ksq[i][0]+ksq[i+1][0]);
        d2[1]=dfact*(ksq[i-1][1]+6.0*ksq[i][1]+ksq[i+1][1]);
        d3[0]=c3+dfact*(ksq[i][0]+ksq[i+1][0]);
        d3[1]=dfact*(ksq[i][1]+ksq[i+1][1]);
        //
        for(int j=0;j<np;j++) {
            r1[j][i][0]=a1+pd2[j][0]*d1[0]-pd2[j][1]*d1[1];
            r1[j][i][1]= 0+pd2[j][1]*d1[0]+pd2[j][0]*d1[1];
            s1[j][i][0]=a1+pd1[j][0]*d1[0]-pd1[j][1]*d1[1];
            s1[j][i][1]= 0+pd1[j][1]*d1[0]+pd1[j][0]*d1[1];
            r2[j][i][0]=a2+pd2[j][0]*d2[0]-pd2[j][1]*d2[1];
            r2[j][i][1]= 0+pd2[j][1]*d2[0]+pd2[j][0]*d2[1];
            s2[j][i][0]=a2+pd1[j][0]*d2[0]-pd1[j][1]*d2[1];
            s2[j][i][1]= 0+pd1[j][1]*d2[0]+pd1[j][0]*d2[1];
            r3[j][i][0]=a3+pd2[j][0]*d3[0]-pd2[j][1]*d3[1];
            r3[j][i][1]= 0+pd2[j][1]*d3[0]+pd2[j][0]*d3[1];
            s3[j][i][0]=a3+pd1[j][0]*d3[0]-pd1[j][1]*d3[1];
            s3[j][i][1]= 0+pd1[j][1]*d3[0]+pd1[j][0]*d3[1];
        }
    }

#else

    //
    for(int i=1; i< nz+1; i++) {
        //
        //     Discretization by Galerkin's method.
        //
        ram_real c1=cfact*f1[i]*(f2[i-1]+f2[i])*f3[i-1];
        ram_real c2=-cfact*f1[i]*(f2[i-1]+2.0*f2[i]+f2[i+1])*f3[i];
        ram_real c3=cfact*f1[i]*(f2[i]+f2[i+1])*f3[i+1];
        d1[0]=c1+dfact*(ksq[i-1][0]+ksq[i][0]);
        d1[1]=dfact*(ksq[i-1][1]+ksq[i][1]);
        d2[0]=c2+dfact*(ksq[i-1][0]+6.0*ksq[i][0]+ksq[i+1][0]);
        d2[1]=dfact*(ksq[i-1][1]+6.0*ksq[i][1]+ksq[i+1][1]);
        d3[0]=c3+dfact*(ksq[i][0]+ksq[i+1][0]);
        d3[1]=dfact*(ksq[i][1]+ksq[i+1][1]);
        //
        for(int j=0;j<np;j++) {
            r1[j][i][0]=a1+pd2[j][0]*d1[0]-pd2[j][1]*d1[1];
            r1[j][i][1]= 0+pd2[j][1]*d1[0]+pd2[j][0]*d1[1];
            s1[j][i][0]=a1+pd1[j][0]*d1[0]-pd1[j][1]*d1[1];
            s1[j][i][1]= 0+pd1[j][1]*d1[0]+pd1[j][0]*d1[1];
            r2[j][i][0]=a2+pd2[j][0]*d2[0]-pd2[j][1]*d2[1];
            r2[j][i][1]= 0+pd2[j][1]*d2[0]+pd2[j][0]*d2[1];
            s2[j][i][0]=a2+pd1[j][0]*d2[0]-pd1[j][1]*d2[1];
            s2[j][i][1]= 0+pd1[j][1]*d2[0]+pd1[j][0]*d2[1];
            r3[j][i][0]=a3+pd2[j][0]*d3[0]-pd2[j][1]*d3[1];
            r3[j][i][1]= 0+pd2[j][1]*d3[0]+pd2[j][0]*d3[1];
            s3[j][i][0]=a3+pd1[j][0]*d3[0]-pd1[j][1]*d3[1];
            s3[j][i][1]= 0+pd1[j][1]*d3[0]+pd1[j][0]*d3[1];
        }
    }

#endif

    //
    //     The entries above the surface.
    //
    for(int j=0;j<np;j++) {
        for(int i=0;i<izsrf;i++) {
            r1[j][i][0]=0.0;
            r1[j][i][1]=0.0;
            r2[j][i][0]=1.0;
            r2[j][i][1]=0.0;
            r3[j][i][0]=0.0;
            r3[j][i][1]=0.0;
            s1[j][i][0]=0.0;
            s1[j][i][1]=0.0;
            s2[j][i][0]=0.0;
            s2[j][i][1]=0.0;
            s3[j][i][0]=0.0;
            s3[j][i][1]=0.0;
        }
    }
    //
    //     The matrix decomposition.
    //
#ifndef RAMSURF_USE_RAW_TDMA
    for(int j=0; j<np; j++) {
        for(int i=1;i<nz+1; i++) {
            ram_real denomr = (r2[j][i][0] - (r1[j][i][0] * r3[j][i-1][0] - r1[j][i][1] * r3[j][i-1][1])),
                  denomi = (r2[j][i][1] - (r1[j][i][0] * r3[j][i-1][1] + r1[j][i][1] * r3[j][i-1][0]));

            // Any other form of this computation yields slightly different results,
            // but this ends up with very different energy loss.
            fcomplex rfact = 1.f/(denomr + I*denomi);
            ram_real rfactr = creal(rfact),
                  rfacti = cimag(rfact);

            ram_real tr1 = r1[j][i][0];
            r1[j][i][0]= tr1 * rfactr - r1[j][i][1] * rfacti;
            r1[j][i][1]= tr1 * rfacti + r1[j][i][1] * rfactr;

            ram_real tr3 = r3[j][i][0];
            r3[j][i][0]= tr3 * rfactr - r3[j][i][1] * rfacti;
            r3[j][i][1]= tr3 * rfacti + r3[j][i][1] * rfactr;

            ram_real ts1 = s1[j][i][0];
            s1[j][i][0]= ts1 * rfactr - s1[j][i][1] * rfacti;
            s1[j][i][1]= ts1 * rfacti + s1[j][i][1] * rfactr;

            ram_real ts2 = s2[j][i][0];
            s2[j][i][0]= ts2 * rfactr - s2[j][i][1] * rfacti;
            s2[j][i][1]= ts2 * rfacti + s2[j][i][1] * rfactr;


            ram_real ts3 = s3[j][i][0];
            s3[j][i][0]= ts3 * rfactr - s3[j][i][1] * rfacti;
            s3[j][i][1]= ts3 * rfacti + s3[j][i][1] * rfactr;

        }
    }
#endif
}
//
//     Matrix updates.
//
static
void updat(ramsurf_t const * const rsurf,
           size_t * const          profl_index,
           size_t const            mr,
           size_t const            mz,
           size_t const            mp,
           int const               nz,
           int const               np,
           int * const             iz,
           int * const             ib,
           ram_real const             dr,
           ram_real const             dz,
           ram_real const             omega,
           ram_real const             k0, 
           ram_real const             r,
           ram_real * const           rp,
           ram_real * const           rs,
           ram_real const             rb[mr], //< ^ Bathymetry description
           ram_real const             zb[mr], //< v
           ram_real                   cw[mz],   //< ^
           ram_real                   cb[mz],   //< |
           ram_real                   rhob[mz], //< |
           ram_real                   attn[mz], //< |
           ram_real                   alpw[mz], //< | Placeholder for profile description
           ram_real                   alpb[mz], //< |
           fcomplex                ksq[mz],  //< |
           fcomplex                ksqw[mz], //< |
           fcomplex                ksqb[mz], //< v
           ram_real                   f1[mz],
           ram_real                   f2[mz],
           ram_real                   f3[mz],
           ram_real                   r1[mp][mz][2],
           ram_real                   r2[mp][mz][2],
           ram_real                   r3[mp][mz][2],
           ram_real                   s1[mp][mz][2],
           ram_real                   s2[mp][mz][2],
           ram_real                   s3[mp][mz][2],
           fcomplex                pd1[mp],
           fcomplex                pd2[mp],
           ram_real const             rsrf[mr], //< ^ Surface description
           ram_real const             zsrf[mr], //< v
           int * const             izsrf,
           int * const             isrf
          ) {
    //
    //     Varying bathymetry.
    //
    while(r>=rb[*ib]) (*ib)++;
    while(r>=rsrf[*isrf]) (*isrf)++;
    //
    int jzsrf=*izsrf;
    ram_real z=zsrf[*isrf-1]+(r+0.5*dr-rsrf[*isrf-1])*
        (zsrf[*isrf]-zsrf[*isrf-1])/(rsrf[*isrf]-rsrf[*isrf-1]);
    *izsrf=z/dz;
    //
    int jz=*iz;
    z=zb[*ib-1]+(r+0.5*dr-rb[*ib-1])*(zb[*ib]-zb[*ib-1])/(rb[*ib]-rb[*ib-1]);
    *iz=1.0+z/dz;
    *iz=max(2,*iz);
    *iz=min(nz,*iz);
    if((*iz!=jz) || (*izsrf != jzsrf))
        matrc(mz, mp, nz, np, *iz, dz, k0, rhob, alpw, alpb, (ram_real (*)[2])ksq, (ram_real const (*)[2])ksqw, (ram_real const (*)[2])ksqb, 
                f1, f2, f3, r1, r2, r3, s1, s2, s3, (ram_real (*)[2])pd1, (ram_real (*)[2])pd2, *izsrf);
    //
    //     Varying profiles.
    //
    /* Range profiles are specified on independently formed grids. Allow a
       few ulps at their boundaries rather than delaying a profile by a step. */
    const ram_real range_eps = sizeof(ram_real) == sizeof(double) ? DBL_EPSILON : FLT_EPSILON;
    while(r + 8 * range_eps * fmax(fabs(r), fabs(dr)) >= *rp){
        ++*profl_index;
        profl(rsurf, *profl_index, mz, nz, dz, omega, k0, rp, cw, cb, rhob, attn, 
                alpw, alpb, (ram_real (*)[2])ksqw, (ram_real (*)[2])ksqb);
        matrc(mz, mp, nz, np, *iz, dz, k0, rhob, alpw, alpb, (ram_real (*)[2])ksq, (ram_real const (*)[2])ksqw, (ram_real const (*)[2])ksqb, 
                f1, f2, f3, r1, r2, r3, s1, s2, s3, (ram_real (*)[2])pd1, (ram_real (*)[2])pd2, *izsrf);
    }
    //
    //     Turn off the stability constraints.
    //
    if(r>=*rs) {
        int ns=0;
        *rs=2.f*(rsurf->rmax);
        epade(mp, np, ns, 1, k0, dr, (ram_real (*)[2])pd1, (ram_real (*)[2])pd2);
        matrc(mz, mp, nz, np, *iz, dz, k0, rhob, alpw, alpb, (ram_real (*)[2])ksq, (ram_real const (*)[2])ksqw, (ram_real const (*)[2])ksqb, 
                f1, f2, f3, r1, r2, r3, s1, s2, s3, (ram_real (*)[2])pd1, (ram_real (*)[2])pd2, *izsrf);
    }
    //
}

//
//     Output transmission loss and complex field.
//
//     `tlg`/`out` receive the usual transmission loss (magnitude only); `cg`/
//     `outc` receive the complex envelope `ur = f3*u` as interleaved
//     (real, imag) pairs. The physical field is recovered from `ur` by
//     re-inserting the carrier that the parabolic approximation factored out:
//     `p(r,z) = ur(r,z) exp(i k0 r) / sqrt(r)` (the 1/sqrt(r) cylindrical
//     spreading is only present for a 3-D point source; the 2-D reference
//     solutions drop it).
//
static
void  outpt(output_t * const out,
            output_t * const outc,
            FILE * const     fdline,
            size_t const     mz,
            int * const      mdr,
            int const        ndr,
            int const        ndz,
            int const        nzplt,
            int const        lz,
            int const        ir,
            ram_real const      dir,
            ram_real const      r,
            ram_real const      f3[mz],
            ram_real const      u[mz][2],
            ram_real            tlg[mz],
            ram_real            cg[2*mz]) {
    const ram_real eps=1.0e-20;
    //
    fcomplex ur=(1.0f-dir)*(f3[ir-1]*(u[ir-1][0]+I*u[ir-1][1]))+dir*f3[ir]*(u[ir][0]+I*u[ir][1]);
    if(fdline) {
        ram_real tl=-20.0*log10(cabs(ur)+eps)+10.0*log10(r+eps);
        fprintf(fdline,"%f %f\n",r,tl);
    }
    //
    *mdr=*mdr+1;
    if(*mdr==ndr) {
        *mdr=0;
        //
        int j=0;
        for(int i=ndz-1; i< nzplt; i+=ndz) {
            ur=(u[i][0]+I*u[i][1])*f3[i];
            tlg[j]=-20*log10((cabs(ur)+eps)/sqrt(r+eps));
            cg[2*j]  =creal(ur);
            cg[2*j+1]=cimag(ur);
            j=j+1;
            //
        }
       output_push_back(out, tlg, lz, r);
       output_push_back(outc, cg, 2*lz, r);
    }
    //
}

//
//     The tridiagonal solver.
//
static
void solve(size_t mz, size_t mp, int nz, int const np, ram_real u[mz][2], 
        ram_real r1[mp][mz][2], ram_real r2[mp][mz][2], ram_real r3[mp][mz][2],
        ram_real s1[mp][mz][2], ram_real s2[mp][mz][2], ram_real s3[mp][mz][2]) {
#ifdef RAMSURF_USE_RAW_TDMA
    /* PoC: replace the fork's factored substitution with a raw complex
       tridiagonal solved by an external kernel (see the RAMSURF_USE_RAW_TDMA
       block near the top). matrc must be compiled *without* its LU (see the
       `#ifndef RAMSURF_USE_RAW_TDMA` around the decomposition above), so
       r1,r2,r3 hold the raw tridiagonal and s1,s2,s3 the Pade RHS operator.
       u[0] and u[nz+1] are the fixed boundary values; the top boundary is
       absent from the fork's own scheme (r1[1] is unused there), the bottom
       one is folded into the last row. */
    /* The fork's solve() must not modify r1,r2,r3 (selfs() solves the same
       system twice), so tdma_single_ works on private copies. */
    static ram_cplx *pa_d = NULL, *pa_A = NULL, *pa_B = NULL, *pa_C = NULL;
    static size_t pa_cap = 0;
    if (pa_cap < (size_t)(nz + 2)) {
        pa_cap = (size_t)(nz + 2);
        pa_d = realloc(pa_d, sizeof(ram_cplx) * pa_cap);
        pa_A = realloc(pa_A, sizeof(ram_cplx) * pa_cap);
        pa_B = realloc(pa_B, sizeof(ram_cplx) * pa_cap);
        pa_C = realloc(pa_C, sizeof(ram_cplx) * pa_cap);
    }
    for (int j = 0; j < np; j++) {
        for (int i = 1; i <= nz; i++) {
            pa_A[i] = *(ram_cplx *)&r1[j][i][0];
            pa_B[i] = *(ram_cplx *)&r2[j][i][0];
            pa_C[i] = *(ram_cplx *)&r3[j][i][0];
            pa_d[i] = (*(ram_cplx *)&s1[j][i][0]) * (*(ram_cplx *)&u[i-1][0])
                    + (*(ram_cplx *)&s2[j][i][0]) * (*(ram_cplx *)&u[i  ][0])
                    + (*(ram_cplx *)&s3[j][i][0]) * (*(ram_cplx *)&u[i+1][0]);
        }
        /* Fixed boundary values: only the bottom row couples outward. */
        pa_d[nz] -= pa_C[nz] * (*(ram_cplx *)&u[nz+1][0]);
        pa_C[nz] = 0.0;
        pa_A[1]  = 0.0;   /* r1[1] unused by the fork's scheme */

#ifdef RAMSURF_OPENMP_PCR
        pcr_tdma_complex(nz, pa_A + 1, pa_B + 1, pa_C + 1, pa_d + 1);
#else
        int nn = nz;
        tdma_single_(pa_A + 1, pa_B + 1, pa_C + 1, pa_d + 1, &nn);
#endif

        for (int i = 1; i <= nz; i++) {
            *(ram_cplx *)&u[i][0] = pa_d[i];
        }
    }
#else
    for(int j=0; j<np; j++) {

        ram_real vr = s1[j][1][0]*u[0][0]-s1[j][1][1]*u[0][1] + s2[j][1][0]*u[1][0] - s2[j][1][1]*u[1][1] + s3[j][1][0]*u[2][0] - s3[j][1][1]*u[2][1];
        ram_real vc = s1[j][1][0]*u[0][1]+s1[j][1][1]*u[0][0] + s2[j][1][0]*u[1][1] + s2[j][1][1]*u[1][0] + s3[j][1][0]*u[2][1] + s3[j][1][1]*u[2][0];

        for(int i=2; i< nz+1; i++) {
            ram_real v0, v1;
            v0 = s1[j][i][0]*u[i-1][0] - s1[j][i][1]*u[i-1][1];
            v0+= s2[j][i][0]*u[i+0][0] - s2[j][i][1]*u[i][1];
            v0+= s3[j][i][0]*u[i+1][0] - s3[j][i][1]*u[i+1][1];
            v0-= r1[j][i][0]*vr        - r1[j][i][1]*vc;
            v1 = s1[j][i][0]*u[i-1][1] + s1[j][i][1]*u[i-1][0];
            v1+= s2[j][i][0]*u[i+0][1] + s2[j][i][1]*u[i][0];
            v1+= s3[j][i][0]*u[i+1][1] + s3[j][i][1]*u[i+1][0];
            v1-= r1[j][i][0]*vc        + r1[j][i][1]*vr;
            u[i-1][0] = vr;
            u[i-1][1] = vc;
            vr = v0;
            vc = v1;
        }
        u[nz][0] = vr;
        u[nz][1] = vc;
        //
        ram_real ur = u[nz+1][0],
              uc = u[nz+1][1];
        for(int i=nz; i>=1; i--) {
            u[i][0] -= (r3[j][i][0] * ur - r3[j][i][1] * uc);
            u[i][1] -= (r3[j][i][0] * uc + r3[j][i][1] * ur);
            ur = u[i][0];
            uc = u[i][1];
        }
        //
    }
#endif
    //
}

//
//     The self-starter.
//
static
void selfs(size_t const mz,
           size_t const mp,
           int const    nz,
           int const    np,
           int const    ns,
           int const    iz,
           ram_real const  zs,
           ram_real const  dr,
           ram_real const  dz,
           ram_real const  k0,
           ram_real        rhob[mz],
           ram_real        alpw[mz], 
           ram_real        alpb[mz],
           fcomplex     ksq[mz],
           fcomplex     ksqw[mz],
           fcomplex     ksqb[mz],
           ram_real        f1[mz],
           ram_real        f2[mz],
           ram_real        f3[mz],
           ram_real        u[mz][2], 
           ram_real        r1[mp][mz][2],
           ram_real        r2[mp][mz][2],
           ram_real        r3[mp][mz][2],
           ram_real        s1[mp][mz][2],
           ram_real        s2[mp][mz][2],
           ram_real        s3[mp][mz][2],
           fcomplex     pd1[mp],
           fcomplex     pd2[mp], 
           int const    izsrf
          ) {
    //
    //     Conditions for the delta function.
    //
    ram_real si=1.0+zs/dz;
    int is=(int)si;
    ram_real dis=si-(ram_real)is;
    u[is-1][0]=(1.0-dis)*sqrt(2.0*M_PI/k0)/(dz*alpw[is-1]);
    u[is-1][1]=0.f;
    u[is][0]=dis*sqrt(2.0*M_PI/k0)/(dz*alpw[is-1]);
    u[is][1]=0.f;
    //
    //     Divide the delta function by [1-X]**2 to get a smooth rhs.
    //
    pd1[0]=0.0;
    pd2[0]=-1.0;
    matrc(mz, mp, nz, 1, iz, dz, k0, rhob, alpw, alpb, (ram_real (*)[2])ksq, (ram_real const (*)[2])ksqw, (ram_real const (*)[2])ksqb, 
            f1, f2, f3, r1, r2, r3, s1, s2, s3, (ram_real (*)[2])pd1, (ram_real (*)[2])pd2, izsrf);
    solve(mz, mp, nz, 1, u, r1, r2, r3, s1, s2, s3);
    solve(mz, mp, nz, 1, u, r1, r2, r3, s1, s2, s3);
    //
    //     Apply the operator [1-X]**2*[1+X]**[-1/4]*exp[I*k0*r*sqrt[1+X]].
    //
    epade(mp, np,ns,2,k0,dr,(ram_real (*)[2])pd1,(ram_real (*)[2])pd2);
    matrc(mz, mp, nz, np, iz, dz, k0, rhob, alpw, alpb, (ram_real (*)[2])ksq, (ram_real const (*)[2])ksqw, (ram_real const (*)[2])ksqb, 
            f1, f2, f3, r1, r2, r3, s1, s2, s3, (ram_real (*)[2])pd1, (ram_real (*)[2])pd2, izsrf);
    solve(mz, mp, nz, np, u, r1, r2, r3, s1, s2, s3);
    //
}


//
//     The Gaussian-beam starter.
//
//     RAM's built-in self-starter launches an isotropic line/point source at
//     depth zs. This alternative injects a directional Herglotz
//     (angular-spectrum) Gaussian beam, matching the incident field used by
//     the SWKB reference solutions. The initial envelope at x = 0 is
//
//         u(0,z) = int exp(-(phi-phi0)^2/(2 sigma^2))
//                    * exp(i k0 (z-z0) sin(phi)) dphi,
//
//     i.e. the x = 0 slice of the field
//         u_inc(x,z) = int exp(-(phi-phi0)^2/(2 sigma^2))
//                        * exp(i k0 (x cos(phi) + (z-z0) sin(phi))) dphi.
//     Because RAM marches the envelope u = p exp(-i k0 x), this slice is
//     exactly the correct wide-angle PE initial condition. The field is then
//     advanced one range step dr so that, like the self-starter, it is ready
//     at the first output range.
//
//     The angular quadrature mirrors SWKB's `gaussian_beam` construction.
//
static
void beams(size_t const mz,
           size_t const mp,
           int const    nz,
           int const    np,
           int const    ns,
           int const    iz,
           ram_real const  sigma,
           ram_real const  phi0,
           ram_real const  z0,
           ram_real const  dr,
           ram_real const  dz,
           ram_real const  k0,
           ram_real        rhob[mz],
           ram_real        alpw[mz],
           ram_real        alpb[mz],
           fcomplex     ksq[mz],
           fcomplex     ksqw[mz],
           fcomplex     ksqb[mz],
           ram_real        f1[mz],
           ram_real        f2[mz],
           ram_real        f3[mz],
           ram_real        u[mz][2],
           ram_real        r1[mp][mz][2],
           ram_real        r2[mp][mz][2],
           ram_real        r3[mp][mz][2],
           ram_real        s1[mp][mz][2],
           ram_real        s2[mp][mz][2],
           ram_real        s3[mp][mz][2],
           fcomplex     pd1[mp],
           fcomplex     pd2[mp],
           int const    izsrf
          ) {
    //
    //     Angular quadrature nodes (uniform). For sigma << 1 the Gaussian
    //     spectrum is negligible outside [phi0-10 sigma, phi0+10 sigma], as in
    //     SWKB's gaussian_beam; broad spectra fall back to [0, 2 pi].
    //
    // Quadrature nodes. A fixed 2001 was too coarse for broad spectra (the
    // phase k0*z*sin(phi) aliases and the starter amplitude collapses, e.g. to
    // ~0.5 of the analytic Herglotz value for sigma = 18 deg); scale the node
    // count with the angular span so the phase is resolved for any sigma.
    const int Nphi = 2001 + (int)(4000.0 * (2.0 * 10.0 * sigma) / (2.0 * M_PI));
    double lo, hi;
    if(10.0*sigma < M_PI) {
        lo = (double)phi0 - 10.0*sigma;
        hi = (double)phi0 + 10.0*sigma;
    } else {
        lo = 0.0;
        hi = 2.0*M_PI;
    }
    double dphi = (Nphi > 1) ? (hi - lo)/(Nphi - 1) : 0.0;
    double inv2s2 = 1.0/(2.0*(double)sigma*(double)sigma);
    double k0d = (double)k0, z0d = (double)z0, dzd = (double)dz;

    for(int i=0;i<nz+2;i++) {
        double zi = (double)i*dzd - z0d;
        double sr = 0.0, si = 0.0;
        for(int j=0;j<Nphi;j++) {
            double phi = lo + (double)j*dphi;
            double d = phi - (double)phi0;
            double wgt = exp(-d*d*inv2s2);
            double phase = k0d*zi*sin(phi);
            sr += wgt*cos(phase);
            si += wgt*sin(phase);
        }
        u[i][0] = (ram_real)(dphi*sr);
        u[i][1] = (ram_real)(dphi*si);
    }
    //
    //     Advance one range step to r = dr with the propagation operator,
    //     exactly as the main marching loop does.
    //
    epade(mp, np, ns, 1, k0, dr, (ram_real (*)[2])pd1, (ram_real (*)[2])pd2);
    matrc(mz, mp, nz, np, iz, dz, k0, rhob, alpw, alpb,
            (ram_real (*)[2])ksq, (ram_real const (*)[2])ksqw, (ram_real const (*)[2])ksqb,
            f1, f2, f3, r1, r2, r3, s1, s2, s3,
            (ram_real (*)[2])pd1, (ram_real (*)[2])pd2, izsrf);
    solve(mz, mp, nz, np, u, r1, r2, r3, s1, s2, s3);
    //
}


//
//     Initialize the parameters, acoustic field, and matrices.
//
static
void setup(ramsurf_t const * const rsurf,
           size_t * const          profl_index,
           output_t * const        out,
           output_t * const        outc,
           FILE * const            fdline,
           size_t const            mr,
           size_t const            mz,
           size_t const            mp,
           int * const             nz,
           int * const             np,
           int * const             ns,
           int * const             mdr,
           int * const             ndr,
           int * const             ndz,
           int * const             iz,
           int * const             nzplt,
           int * const             lz,
           int * const             ib,
           int * const             ir,
           ram_real * const           dir,
           ram_real * const           dr,
           ram_real * const           dz,
           ram_real * const           omega, 
           ram_real * const           k0,
           ram_real * const           r,
           ram_real * const           rp,
           ram_real * const           rs,
           ram_real                   rb[mr],
           ram_real                   zb[mr],
           ram_real                   cw[mz],
           ram_real                   cb[mz],
           ram_real                   rhob[mz],
           ram_real                   attn[mz],
           ram_real                   alpw[mz],
           ram_real                   alpb[mz],
           fcomplex                ksq[mz],
           fcomplex                ksqw[mz],
           fcomplex                ksqb[mz],
           ram_real                   f1[mz],
           ram_real                   f2[mz],
           ram_real                   f3[mz],
           ram_real                   u[mz][2], 
           ram_real                   r1[mp][mz][2],
           ram_real                   r2[mp][mz][2],
           ram_real                   r3[mp][mz][2],
           ram_real                   s1[mp][mz][2],
           ram_real                   s2[mp][mz][2],
           ram_real                   s3[mp][mz][2],
           fcomplex                pd1[mp],
           fcomplex                pd2[mp],
           ram_real                   tlg[mz],
           ram_real                   cg[2*mz],
           ram_real                   rsrf[mr],
           ram_real                   zsrf[mr],
           int * const             izsrf,
           int * const             isrf
          ) {
    ram_real zr,zmax,zmplt;

    zr = rsurf->zr;
    *dr = rsurf->dr ; *ndr = rsurf->ndr;
    zmax = rsurf->zmax ; *dz = rsurf->dz ; *ndz = rsurf->ndz ; zmplt = rsurf->zmplt;
    *np = rsurf->np; *ns = rsurf->ns; *rs = rsurf->rs;

    //
    int i=0;
    while(1) {
        rsrf[i] = rsurf->rsrf[i];
        zsrf[i] = rsurf->zsrf[i];
        if( rsrf[i] < 0.0) break;
        i=i+1;
    }

    rsrf[i]=2.0*(rsurf->rmax);
    zsrf[i]=zsrf[i-1];
    //
    i=0;
    while(1) {
        rb[i] = rsurf->rb[i];
        zb[i] = rsurf->zb[i];
        if( rb[i] < 0.0) break;
        i=i+1;
    }

    rb[i]=2.0*(rsurf->rmax);
    zb[i]=zb[i-1];
    //
    *ib=1;
    *isrf=1;
    *mdr=0;
    *r=*dr;
    *omega=2.0*M_PI*rsurf->freq;
    ram_real ri=1.0+zr/(*dz);
    *ir=(int)ri;
    *dir=ri-(ram_real)(*ir);
    *k0=*omega/(rsurf->c0);
    *nz=zmax/(*dz)-0.5;
    assert(*nz +2 < mz);
    *nzplt=zmplt/(*dz)-0.5;
    //
    ram_real z=zsrf[0];
    *izsrf=1.0+z/(*dz);
    //
    z=zb[0];
    *iz=1.0+z/(*dz);
    *iz=max(2,*iz);
    *iz=min(*nz,*iz);
    if(*rs < *dr){*rs=2.0*(rsurf->rmax);}
    //
    for(size_t j=0;j<mp; j++) {
        r3[j][0][0]=0.0;
        r3[j][0][1]=0.0;
        r1[j][*nz+1][0]=0.0;
        r1[j][*nz+1][1]=0.0;
    }
    for(int i=0; i< *nz+2; i++) {
        u[i][0]=0.0;
        u[i][1]=0.0;
    }

    *lz=0;
    for(int i=(*ndz); i<=(*nzplt); i+=(*ndz)) {
        *lz=*lz+1;
    }

    //
    //     The initial profiles and starting field.
    //
    *profl_index=0;
    profl(rsurf, *profl_index, mz, *nz, *dz, *omega, *k0, rp, cw, cb, rhob, attn, 
            alpw, alpb, (ram_real (*)[2])ksqw, (ram_real (*)[2])ksqb);
    if(rsurf->nfield > 0) {
        // The caller supplies physical pressure. matrc builds f3, whereas
        // RAM advances the energy-scaled variable pressure/f3.
        epade(mp, *np, *ns, 1, *k0, *dr, (ram_real (*)[2])pd1, (ram_real (*)[2])pd2);
        matrc(mz, mp, *nz, *np, *iz, *dz, *k0, rhob, alpw, alpb,
                (ram_real (*)[2])ksq, (ram_real const (*)[2])ksqw, (ram_real const (*)[2])ksqb,
                f1, f2, f3, r1, r2, r3, s1, s2, s3,
                (ram_real (*)[2])pd1, (ram_real (*)[2])pd2, *izsrf);
        for(int i=0;i<*nz+2;i++) {
            u[i][0] = rsurf->field_re[i] / f3[i];
            u[i][1] = rsurf->field_im[i] / f3[i];
        }
        solve(mz, mp, *nz, *np, u, r1, r2, r3, s1, s2, s3);
    } else if(rsurf->beam) {
        beams(mz, mp, *nz, *np, *ns, *iz, rsurf->beam_sigma, rsurf->beam_phi0,
                rsurf->beam_z0, *dr, *dz, *k0, rhob, alpw, alpb, ksq,
                ksqw, ksqb, f1, f2, f3, u, r1, r2, r3, s1, s2, s3, pd1, pd2, *izsrf);
    } else {
        selfs(mz, mp, *nz, *np, *ns, *iz, rsurf->zs, *dr, *dz, *k0, rhob, alpw, alpb, ksq,
                ksqw, ksqb, f1, f2, f3, u, r1, r2, r3, s1, s2, s3, pd1, pd2, *izsrf);
    }
    outpt(out, outc, fdline, mz, mdr, *ndr, *ndz, *nzplt, *lz, *ir, *dir, *r, f3, (ram_real const (*)[2])u, tlg, cg);
    //
    //     The propagation matrices.
    //
    epade(mp, *np, *ns, 1, *k0, *dr, (ram_real (*)[2])pd1, (ram_real (*)[2])pd2);
    matrc(mz, mp, *nz, *np, *iz, *dz, *k0, rhob, alpw, alpb, (ram_real (*)[2])ksq, (ram_real const (*)[2])ksqw, (ram_real const (*)[2])ksqb, 
            f1, f2, f3, r1, r2, r3, s1, s2, s3, (ram_real (*)[2])pd1, (ram_real (*)[2])pd2, *izsrf);
    //
}

static
void fix_zmax(ram_real *zmax, ram_real**data) {
    while(*data) {
        ram_real * iter = *data;
        while(*iter != -1.f) iter +=2;
        if(*zmax < iter[-2]) {
            *zmax = iter[-2];
        }
        ++data;
    }
}

int ramsurf(ramsurf_t const* rsurf, int * lz, ram_real *** ogrid, ram_real *** ocgrid, FILE *fdline)
{
    ram_real k0;
    int errorCode;
    ram_real lzmax = rsurf->zmax;
    fix_zmax(&lzmax, rsurf->cw);
    fix_zmax(&lzmax, rsurf->cb);
    fix_zmax(&lzmax, rsurf->rhob);
    fix_zmax(&lzmax, rsurf->attn);

#ifdef __SSE3__
    // this sets the "flush to zero" and "denormals are zero" bits (15,6) in the MXCSR register.
    // otherwise we get very bad timings...
    int flush_mode = _MM_GET_FLUSH_ZERO_MODE( );
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
#endif

    size_t mr = rsurf->mr,
           mz = lzmax / rsurf->dz + 2.5f,
           mp = rsurf->np;



    output_t out;
    output_init(&out);

    output_t outc;
    output_init(&outc);

    // allocation step 
    void *scratch = malloc(sizeof(ram_real) * ( 2*mz*3 + 2*mz*mp*3 + 2*mp*2 + mr*4 + mz *10 + 2*mz*1 + 2*mp*mz*3 + 2*mz));
    void *oscratch = scratch;
    fcomplex (*ksq)[mz] = scratch; scratch += sizeof(ram_real) * 2*mz;
    fcomplex (*ksqb)[mz] = scratch; scratch += sizeof(ram_real) * 2*mz;
    fcomplex (*ksqw)[mz] = scratch; scratch += sizeof(ram_real) * 2*mz;
    ram_real (*r1)[mp][mz][2]= scratch; scratch += sizeof(ram_real) * 2*mz*mp;
    ram_real (*r2)[mp][mz][2]= scratch; scratch += sizeof(ram_real) * 2*mz*mp;
    ram_real (*r3)[mp][mz][2]= scratch; scratch += sizeof(ram_real) * 2*mz*mp;
    fcomplex (*pd1)[mp]= scratch; scratch += sizeof(ram_real) * 2*mp;
    fcomplex (*pd2)[mp]= scratch; scratch += sizeof(ram_real) * 2*mp;
    ram_real (*rb)[mr]= scratch; scratch += sizeof(ram_real) * mr;
    ram_real (*zb)[mr]= scratch; scratch += sizeof(ram_real) * mr;
    ram_real (*rsrf)[mr]= scratch; scratch += sizeof(ram_real) * mr;
    ram_real (*zsrf)[mr]= scratch; scratch += sizeof(ram_real) * mr;
    ram_real (*cw)[mz]= scratch; scratch += sizeof(ram_real) * mz;
    ram_real (*cb)[mz]= scratch; scratch += sizeof(ram_real) * mz;
    ram_real (*rhob)[mz]= scratch; scratch += sizeof(ram_real) * mz;
    ram_real (*attn)[mz]= scratch; scratch += sizeof(ram_real) * mz;
    ram_real (*alpw)[mz]= scratch; scratch += sizeof(ram_real) * mz;
    ram_real (*alpb)[mz]= scratch; scratch += sizeof(ram_real) * mz;
    ram_real (*f1)[mz]= scratch; scratch += sizeof(ram_real) * mz;
    ram_real (*f2)[mz]= scratch; scratch += sizeof(ram_real) * mz;
    ram_real (*f3)[mz]= scratch; scratch += sizeof(ram_real) * mz;
    ram_real (*tlg)[mz]= scratch; scratch += sizeof(ram_real) * mz;
    ram_real (*cg)[2*mz]= scratch; scratch += sizeof(ram_real) * 2*mz;
    ram_real (*u)[mz][2]= scratch; scratch += sizeof(ram_real) * 2*mz;

    ram_real (*s1)[mp][mz][2]= scratch; scratch += sizeof(ram_real) * 2*mz*mp;
    ram_real (*s2)[mp][mz][2]= scratch; scratch += sizeof(ram_real) * 2*mz*mp;
    ram_real (*s3)[mp][mz][2]= scratch; scratch += sizeof(ram_real) * 2*mz*mp;

    int nz,np,ns,mdr,ndr,ndz,iz,nzplt,ib,ir,izsrf,isrf;
    size_t profl_index;
    ram_real omega, r, rp, rs, dr, dz, dir;
    if(!(errorCode=setjmp(exception_env)))
    {
        setup(rsurf, &profl_index, &out, &outc, fdline, mr, mz, mp, &nz, &np, &ns, &mdr, &ndr, &ndz, &iz,
                &nzplt, lz, &ib, &ir,
                &dir, &dr, &dz, &omega, 
                &k0, &r, &rp, &rs, *rb, *zb, *cw, *cb, *rhob, 
                *attn, *alpw, *alpb, *ksq, *ksqw, *ksqb, 
                *f1, *f2, *f3, 
                *u, 
                *r1, *r2, *r3, *s1, *s2, *s3, 
                *pd1, *pd2, *tlg, *cg, *rsrf, *zsrf, &izsrf, &isrf);
        //
        //     March the acoustic field out in range.
        //
        size_t march_step = 1;
        while (r < rsurf->rmax) {
            updat(rsurf, &profl_index, mr, mz, mp, nz, np, &iz, &ib, dr, dz, omega, k0, r, 
                    &rp, &rs, *rb, *zb, *cw, *cb, *rhob, *attn, *alpw, *alpb, *ksq, *ksqw, *ksqb, *f1, *f2, *f3, 
                    *r1, *r2, *r3, *s1, *s2, *s3, *pd1, *pd2, *rsrf, *zsrf, &izsrf, &isrf);
            solve(mz, mp, nz, np, *u, *r1, *r2, *r3, *s1, *s2, *s3);
            r=(++march_step)*dr;
            outpt(&out, &outc, fdline, mz,  &mdr, ndr, ndz, nzplt, *lz, ir, dir, r, *f3, (ram_real const (*)[2]) *u, *tlg, *cg);
        }
        if(errorCode) {
            output_destroy(&out);
            output_destroy(&outc);
        } else {
            *ogrid = output_release(&out);
            *ocgrid = output_release(&outc);
        }
    }

    // deallocation step 
    free(oscratch);
#ifdef __SSE3__
    _MM_SET_FLUSH_ZERO_MODE(flush_mode);
#endif

    return errorCode;
}

