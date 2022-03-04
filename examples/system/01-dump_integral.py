#! /usr/bin/env python

"""
Dump integral.
"""

import numpy as np
np.set_printoptions(3, linewidth=1000, suppress=True)
from libdmet.system import lattice as latt
from libdmet.system import hamiltonian as ham
from libdmet.system import integral

# 1. a 10 site 1D Hubbard model
LatSize = 10
ImpSize = 10
Lat = latt.ChainLattice(LatSize, ImpSize)
nao = Lat.nao
nelec = nao
U = 6.0
LatHam = ham.HubbardHamiltonian(Lat, U, obc=False)
H1_ao = LatHam.getH1()
H2_ao = LatHam.getH2()

Ham = integral.Integral(nao, restricted=False, bogoliubov=False, H0=0.0, \
        H1={"cd": np.asarray((H1_ao[0], H1_ao[0]))}, \
        H2={"ccdd": np.asarray((H2_ao,) * 3)})

# 2. UHF
from libdmet.solver import scf_solver, scf
scfsolver = scf_solver.SCFSolver(restricted=False, Sz=0, tol=1e-9)
# AFM guess
dm0 = np.zeros((2, nao, nao))
for i in range(nao):
    if i % 2 == 0:
        dm0[0, i, i] = 1.0
        dm0[1, i, i] = 0.0
    else:
        dm0[0, i, i] = 0.0
        dm0[1, i, i] = 1.0

rdm1, E_mf = scfsolver.run(Ham, dm0=dm0)

# 3. ao2mo and dump
Ham_uhf = scf.ao2mo_Ham(Ham, scfsolver.scfsolver.mf.mo_coeff, compact=False, in_place=False)
integral.dumpFCIDUMP("FCIDUMP", Ham_uhf, thr=1e-12)

