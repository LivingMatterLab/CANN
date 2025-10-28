# A universal material model subroutine for soft matter systems
When using, please cite  
**"A universal material model subroutine for soft matter systems",  
M. Peirlinck, J.A. Hurtado, M.K. Rausch, A. Buganza Tepole, E. Kuhl,  
Engineering with Computers, 2024**,
[DOI/URL](https://doi.org/10.1007/s00366-024-02031-w)

All input data and code is provided in [LINK](https://github.com/peirlincklab/universalmatsubroutine).

~~~bibtex
@article{Peirlinck2024,
  title = {A universal material model subroutine for soft matter systems},
  volume = {41},
  ISSN = {1435-5663},
  url = {http://dx.doi.org/10.1007/s00366-024-02031-w},
  DOI = {10.1007/s00366-024-02031-w},
  number = {2},
  journal = {Engineering with Computers},
  publisher = {Springer Science and Business Media LLC},
  author = {Peirlinck,  Mathias and Hurtado,  Juan A. and Rausch,  Manuel K. and Tepole,  Adrián Buganza and Kuhl,  Ellen},
  year = {2024},
  month = sep,
  pages = {905–927}
}
~~~

Original isotropic subroutine:
## On automated model discovery and a universal material subroutine

Supplementary software, data, and input files for the paper  
"On automated model discovery and a universal material subroutine"  
by Mathias Peirlinck, Kevin Linka, Juan A. Hurtado, Ellen Kuhl  
Publication: https://doi.org/10.1016/j.cma.2023.116534

**When using/modifying our universal material subroutine, simulation files, and/or data,  
please cite our paper above!**

> This repository is organized as follows:  
> /DATA contains the original experimental data used to autonomously learn the constitutive artifical neural network model for gray and white brain matter tissue  
> /INPUT/HEXcube contains the hexahedral cube benchmark simulation files  
> /INPUT/BRAINslices contains the brain slice simulation files  
> /SUBROUTINE contains the raw fortran universal uanisohyper subroutine, as well as a Windows/Abaqus2022 pre-compiled subroutine for users working on a system without a Fortran compiler installed  
