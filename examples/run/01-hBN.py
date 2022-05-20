#!/usr/bin/env python

'''
Example of DMET for periodic hexagonal boron nitride.
To run a DMET calculation:
1. Define a lattice object that includes a unit cell and kmesh and that 
   defines valence, virtual, and core orbitals.
2. Perform a low-level calculation, e.g. HF here, on the same cell and k-points.
3. Instantiate a DMET object with the lattice, a low-level calculation, and  
   selecting a high-level solver.
4. Use .kernel() to run DMET. 
5. Properties e.g. density matrices and energies can be obtained by the 
   corresponing method following PySCF's standard, e.g. .make_rdm1(), .energy_cell().
'''

import os, sys
import numpy as np
import scipy.linalg as la
from pyscf import lib
from pyscf.pbc.lib import chkfile
from pyscf.pbc import scf, gto, df, cc, tools

from libdmet.system import lattice
from libdmet.basis_transform import make_basis
from libdmet.lo.iao import reference_mol
from libdmet.dmet import Hubbard, dmet
from libdmet.solver import impurity_solver
from libdmet.utils import logger as log

log.verbose = "DEBUG2"
np.set_printoptions(4, linewidth=1000, suppress=True)

### ************************************************************
### Lattice settings
### ************************************************************

'''
    To create a lattice object lat, first build a PySCF cell object for the unit 
    cell, and then use both cell and kmesh to instantiate the Lattice class.
        >>> lat = lattice.Lattice(cell, kmesh)

'''
# Build the unit cell
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
cell = tools.pbc.super_cell(cell, cell_mesh)

# Create the lattice 
kmesh = [3, 3, 1]
lat = lattice.Lattice(cell, kmesh) 
lat.ncell_sc = np.prod(cell_mesh)  # Only change energy evaluation
kpts = lat.kpts  # To be used in low-level HF calculation

# Define valence, virtual and core orbitals
minao = 'gth-szv'
pmol = reference_mol(cell, minao=minao)
ncore = 0
nval = pmol.nao_nr()
nvirt = cell.nao_nr() - ncore - nval
lat.set_val_virt_core(nval, nvirt, ncore)

### ************************************************************
### SCF Mean-field calculation
### ************************************************************

log.section("\nSolving SCF mean-field problem\n")
       
# SCF parameters
exxdiv = None
kmf_conv_tol = 1e-12
kmf_max_cycle = 300
gdf_fname = 'BN331_gdf_ints.h5'
chkfname = 'BN331.chk'

# Run HF with Gaussian density fitting
# This may take a while depending on the lattice size
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

log.result("kmf electronic energy per unit cell: %20.12f", 
        (kmf.energy_tot()-kmf.energy_nuc())/lat.ncell_sc)

### ************************************************************
### DMET embedding calculatiun
### ************************************************************

'''
    Create a DMET object with the low-level calculation, the lattice object, 
    and a high-level solver and run with .kernel().. 
        >>> mydmet = dmet.DMET(kmf, lat, solver) 
        >>> mydmet.kernel()
   
    The solver could be an object of the built-in solver under the sovler folder, 
    an object of a child of the dmetsolvermixin class or a string of one of the 
    following: "ccsd" or "fci".

    By default DMET uses Intrinsic Atomic Orbital (IAO) to localize atomic orbitals.
    One can specify the minimal basis for IAO construction by setting mydmet.minao
    or directly input a predetermined set of localized orbitals (LO) by setting
    mydmet.C_ao_lo to be the basis transformation matrix from AO to LO.

    By default the DMET calculation runs self-consistently. For the single-shot DMET, 
    set dmet.max_iter = 1.
'''
# Initialize a CCSD solver 
solver = impurity_solver.CCSD()
solver.restricted = True
solver.conv_tol = cell.natm * 1e-9
solver.conv_tol_normt = cell.natm * 1e-6
solver.max_memory = max_memory                                    

# Run DMET
mydmet = dmet.DMET(kmf, lat, solver)
mydmet.minao = 'gth-szv' # to construct IAO localization 
e_dmet, conv = mydmet.kernel() # e_dmet = energy per cell



