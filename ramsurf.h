#ifndef RAMSURF_H
#define RAMSURF_H

#include <stdio.h>
#include <complex.h>

// Working precision of the PE march. Compile with -DRAMSURF_DOUBLE for a
// double-precision library (used by the `libramsurfdp` build); the default is
// the original NRL RAM single precision. Only the field, the propagation
// matrices and the profiles/output change precision; the Pade-coefficient
// solvers (epade/deriv/gauss/fndrt) are always double internally.
#ifdef RAMSURF_DOUBLE
typedef double ram_real;
typedef double complex ram_cplx;
#else
typedef float ram_real;
typedef float complex ram_cplx;
#endif

#ifdef __cplusplus
extern "C" {
#endif

    enum {
        //laguerre is not converging
        LAG_NOT_CON = 2,
        NOT_ENOUGH_MEMORY
    };


    typedef struct {
        size_t mr; // max number of elements between rb and rsrf
        ram_real freq, zs, zr; //frequency in Hz, source depth, receiver depth (data at this depth goes into tl.line)
        ram_real rmax, dr; int ndr; // maximum range (rmax), range step (delta r or dr), ndr (writes output every ndr range steps)
        ram_real zmax, dz; int ndz; ram_real zmplt; // maximum depth of the whole computational domain (zmax), depth step (dz), ndz (output data at every ndz vertical point at a given range), vertical extent of the domain to output in tl.grid (zmplt).

        ram_real c0; int np, ns; ram_real rs; // reference sound speed, number of Pade terms (np), # of stability constraints (ns), radius of stability constraint (rs)

        // Optional Gaussian-beam starter ("beam.in"). When beam != 0 the
        // isotropic point-source self-starter is replaced by the Herglotz
        // (angular-spectrum) beam below, evaluated at x = 0:
        //   u(0,z) = int exp(-(phi-phi0)^2/(2 sigma^2))
        //              * exp(i k0 (z-z0) sin(phi)) dphi
        int beam; ram_real beam_sigma, beam_phi0, beam_z0;

        ram_real *rsrf, *zsrf; // surface depth: rsrf contains ranges and zsrf the depths. Both terminate by -1
        ram_real *rb, *zb; // bathymetry: rb contains ranges , zb contains bathymetry values. Both teminate by -1
        // array of ram_real arrays. Each subarray contains a profile, as an array of ram_reals containing depth and informations alternatively.
        ram_real **cw, **cb, **rhob, **attn; // sound speed in water, sound speed in bottom, density of bottom, attenuation of bottom, 
        ram_real *rp; // new profile boundaries (use new profile when a value is reached)
    } ramsurf_t;


    int ramsurf(ramsurf_t const* rsurf, //ram parameters
            int *lz, // number of rows in the NULL terminated ogrid
            ram_real ***ogrid, // the entire range-depth transmission loss field (pointer to the unallocated matrix)
            ram_real ***ocgrid, // complex envelope field f3*u as interleaved (re,im) pairs, 2*lz ram_reals per row
            FILE *fdline  // transmission loss versus range at a specified receiver depth
            );

#ifdef __cplusplus
}
#endif

#endif
