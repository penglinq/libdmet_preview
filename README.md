libDMET
===============================================
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/libdmet/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/libdmet/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/libDMET/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/libDMET/branch/master)
[Build Status](https://github.com/zhcui/libdmet_solid/workflows/CI/badge.svg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A library of density matrix embedding theory (DMET) for lattice models and realistic solids.

Installation
------------

* Prerequisites
    - [PySCF](https://github.com/pyscf/pyscf) 2.0 or higher.

* Add libdmet top-level directory to your `PYTHONPATH` and you are all set!
  e.g. if libdmet_preview is installed in `/opt`, your `PYTHONPATH` should be

        export PYTHONPATH=/opt/libdmet_preview:$PYTHONPATH
	
* Extensions
    - [Wannier90](https://github.com/wannier-developers/wannier90): optional, for wannier functions as local orbitals.
	- [Block2](https://github.com/block-hczhai/block2-preview.git): optional, for DMRG solver.
	- [Stackblock](https://github.com/sanshar/StackBlock): optional, for DMRG solver.
	- [Arrow](https://github.com/QMC-Cornell/shci/tree/master): optional, for SHCI solver.

Reference
------------

The following papers should be cited in publications utilizing the libDMET program package:

Zhi-Hao Cui, Tianyu Zhu, Garnet Kin-Lic Chan, Efficient Implementation of Ab Initio Quantum Embedding in Periodic Systems: 
Density Matrix Embedding Theory, [J. Chem. Theory Comput. 2020, 16, 1, 119-129.](https://pubs.acs.org/doi/10.1021/acs.jctc.9b00933)

Tianyu Zhu, Zhi-Hao Cui, Garnet Kin-Lic Chan, Efficient Formulation of Ab Initio Quantum Embedding in Periodic Systems: 
Dynamical Mean-Field Theory, [J. Chem. Theory Comput. 2020, 16, 1, 141-153.](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.9b00934)

Original [libDMET](https://bitbucket.org/zhengbx/libdmet) by Bo-Xiao Zheng.

Bug reports and feature requests
--------------------------------
Please submit tickets on the issues page.

### Copyright

Copyright (c) 2022, Linqing Peng, Zhihao Cui


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
