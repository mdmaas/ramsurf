RAM PARABOLIC EQUATION
======================

This source code is a translation in optimized C of the original ram code from
Dr. David C. Calvo. See `readme.orig` for original instructions.

The original code shipped five sources:

- `ram1.5.f`
- `ramclr.f`
- `rams0.5.f`
- `ramsurf1.5.f`
- `ramsurfclr2.0.f`

The aim of this project is to make these program run faster by using several
optimization techniques, while keeping the original behavior. This implies a
move to the C language.

Download
--------

Download the latest release with the following link: ramsurf-1.0.tar.gz_

.. _ramsurf-1.0.tar.gz: https://github.com/quiet-oceans/ramsurf/blob/releases/ramsurf-1.0.tar.gz?raw=true 

Building
--------

Simply run::

    ./configure
    make

You can install all the ramsurf variant as well as the headers using::

    sudo make install

To turn on the vectorized C code, run::


    make CFLAGS="-O2 -march=native"

All the traditional autotool flags work, including not building in the sources.

Testing
-------

Compare the output produced by the FORTRAN code and the C code using::

    make check
    
Using the Ramsurf Program
--------------------------

The C version mimics the FORTRAN interface and assumes all inputs are described
in `ram.in`. It outputs three files: `tl.line` as plain text, `tl.grid` (the
transmission-loss field) and `field.grid` (the complex envelope), both using the
binary format used by FORTRAN.

Complex field output
--------------------

`field.grid` stores the complex envelope ``ur = f3*u`` that RAM marches
internally and that `tl.grid` throws away (it keeps only ``-20 log10|ur|``).
The layout mirrors `tl.grid`: a leading Fortran record with the integer ``lz``
(number of depth samples), then one record per output range containing ``2*lz``
single-precision floats, interleaved as ``(real, imag)`` pairs.

Re-inserting the carrier factored out by the parabolic approximation recovers
the physical pressure up to the cylindrical-spreading factor,

    p(r,z) = ur(r,z) * exp(i k0 r) / sqrt(r),      k0 = 2 pi freq / c0 .

The ``1/sqrt(r)`` is present only for a 3-D point source; the 2-D line-source
references (SWKB, spectral fullwave) drop it, and are therefore compared against
``ur * exp(i k0 r)``. The factor ``f3`` is real (a density/celerity ratio), so
``arg(ur) = arg(u)``.

See ``SWKB/part_1_examples/ex8_ram_phase.jl`` for a worked example on the
example-1 lens.

The input format is *the same* as the one described in `readme.orig`::
    
    # sample config. The first line is a header and is skipped. 
    freq zs zr          #frequency in Hz, source depth, receiver depth (data at this depth goes into tl.line)
    rmax dr ndr         #maximum range (rmax), range step (delta r or dr), ndr (writes output every ndr range steps)
    zmax dz ndz zmplt   #maximum depth of the whole computational domain (zmax), depth step (dz), ndz (output data at every ndz vertical point at a given range), vertical extent of the domain to output in tl.grid (zmplt).
    c0 np ns rs         # reference sound speed, number of Pade terms (np), # of stability constraints (ns), radius of stability constraint (rs)
    surface_range0 surface_depth0 # surface modeling
    ...
    surface_rangeN surface_depthN 
    -1 -1               # END of SECTION marker
    bathymetry_range0 bathymetry_depth0 #bathymetry modeling
    ...
    bathymetry_rangeM bathymetry_depthM
    -1 -1
    ssp_depth0 ssp_value0       # sound speed profile in water
    ...
    -1 -1
    bssp_depth0 bssp_value0    # sound speed profile in bottom
    ...
    -1 -1
    bd_depth0 bd_value0    # density of bottom
    ...
    -1 -1
    ba_depth0 ba_value0    # attenuation of bottom
    ...
    -1 -1
    optional_range # if given, starts a new ssp description starting from this range, otherwise the previous values are valid for all remaining ranges
    [...]


Gaussian-beam starter
---------------------

By default RAM launches an isotropic point/line source at depth ``zs`` with its
built-in self-starter. Placing a file named ``beam.in`` in the working directory
(next to ``ram.in``) replaces it with a directional Gaussian beam built as a
Herglotz (angular-spectrum) superposition of plane waves::

    sigma phi0 z0

``sigma`` is the angular spread in radians, ``phi0`` the central launch angle
(radians, measured from the x-axis) and ``z0`` the beam-centre depth in metres.
The initial envelope at ``x = 0`` is

    u(0,z) = int exp(-(phi-phi0)^2/(2 sigma^2)) exp(i k0 (z-z0) sin(phi)) dphi,

with ``k0 = 2 pi freq / c0``. This matches the Gaussian beam used by the SWKB
reference solutions. If ``beam.in`` is absent, or ``sigma <= 0``, the original
point-source self-starter is used.


Using the Ramsurf Library
-------------------------

The C version can be used as a C library. Link with `-lramsurflib` and include
`ramsurf.h`. The interface documentation is contained in the header file.
