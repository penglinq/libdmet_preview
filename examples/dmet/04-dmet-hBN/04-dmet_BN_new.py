'''
Example of DMET for boron nitride with gth-dzv basis.
'''

import os, sys
import numpy as np
import scipy.linalg as la

from pyscf import lib
from pyscf.pbc.lib import chkfile
from pyscf.pbc import scf, gto, df, cc, tools

from libdmet.system import lattice
from libdmet.lo.iao import reference_mol

from libdmet.utils import logger as log
import libdmet.dmet.Hubbard as Hubbard
import libdmet.run_dmet as dmet

log.verbose = "DEBUG2"
np.set_printoptions(4, linewidth=1000, suppress=True)

### ************************************************************
### System settings
### ************************************************************

max_memory = 119000 # 119 G
cell = gto.Cell()
Lz = 20.0
cell.build(unit = 'angstrom',
           a = [[2.50, 0.0, 0.0], [-1.25, 2.1650635094610964, 0.0], [0.0, 0.0, Lz]],
           atom = 'B 0.0 0.0 0.0; N 1.25 0.721687836487032 0.0',
           dimension = 3,
           max_memory = max_memory,
           verbose = 5,
           pseudo = 'gth-pade',
           basis='gth-dzv',
           precision = 1e-12)
cell_mesh = [1, 1, 1]
ncell_sc = np.prod(cell_mesh)
cell = tools.pbc.super_cell(cell, cell_mesh)
natom_sc = cell.natm

kmesh = [3, 3, 1]
Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts

minao = 'gth-szv'
pmol = reference_mol(cell, minao=minao)
ncore = 0
nval = pmol.nao_nr()
nvirt = cell.nao_nr() - ncore - nval
Lat.set_val_virt_core(nval, nvirt, ncore)

### ************************************************************
### SCF Mean-field calculation
### ************************************************************

log.section("\nSolving SCF mean-field problem\n")
       
exxdiv = None
kmf_conv_tol = 1e-12
kmf_max_cycle = 300
gdf_fname = 'BN331_gdf_ints.h5'
chkfname = 'BN331.chk'

gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
if not os.path.isfile(gdf_fname):
    gdf.build()

if os.path.isfile(chkfname):
    kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    kmf.conv_tol = kmf_conv_tol
    kmf.max_cycle = kmf_max_cycle
    data = chkfile.load(chkfname, 'scf')
    kmf.__dict__.update(data)
else:
    kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    kmf.conv_tol = kmf_conv_tol
    kmf.max_cycle = kmf_max_cycle
    kmf.chkfile = chkfname
    kmf.kernel()
    assert(kmf.converged)

log.result("kmf electronic energy: %20.12f", (kmf.energy_tot()-kmf.energy_nuc())/ncell_sc)

### ************************************************************
### DMET embedding calculatiun
### ************************************************************

cc_etol = cell.natm * 1e-9
cc_ttol = cell.natm * 1e-6
solver = Hubbard.impurity_solver.CCSD(restricted=True, tol=cc_etol, \
    tol_normt=cc_ttol, max_memory=max_memory) ## add direct ref to solver
mydmet = dmet.DMET(kmf, Lat, solver)
mydmet.minao = 'gth-szv'
mydmet.max_memory = max_memory
mydmet.ncell_sc = ncell_sc
e_dmet, conv = mydmet.kernel() # e_dmet = energy per cell



