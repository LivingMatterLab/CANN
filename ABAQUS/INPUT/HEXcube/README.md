# CANN/ABAQUS/INPUT/HEXcube
@author:mpeirlinck

This folder contains all the necessary input files to run the single hex cube element benchmark simulations in the  
"On automated model discovery and a universal material subroutine" paper by M. Peirlinck, K. Linka, J.A. Hurtado, E. Kuhl

To run any of these single cube benchmark files, run:  
'abaqus -j HEXcube-UANIuniversal-***.inp -user UANIuniversal.f inter'

Here, the CXblatzko, CXdemiray, CXgent, CXholzapfel, CXmooneyrivlin, CXneohooke benchmark input files incorporate the CANN parameters discovered in Table 1 of the paper.
The CXcann and the CRcann benchmark input files incorporate the ten+com+shr gray and white matter CANN parameters discovered in Table 2 of the paper respectively.  