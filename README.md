# linalg

This project implements linear algebra using a design
pattern following SLATE: https://icl.utk.edu/slate/

Matrices are maps from indices to tiles.
Individual tiles may be resident on any
MPI rank or co-processor devices.  Each tile has a
designated home location, but can be copied, moved,
or updated at will during a parallel algorithm.

