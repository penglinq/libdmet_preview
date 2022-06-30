#!/usr/bin/env python

'''
Example of wannierization for Si valence bands.
'''

import os, sys
import numpy as np
import scipy.linalg as la

from pyscf.pbc.lib import chkfile
from pyscf.pbc import scf, dft, gto, df, cc, tools

from libdmet.system import lattice
from libdmet.basis_transform import make_basis
from libdmet.lo.iao import reference_mol
from libdmet.lo import pywannier90
from libdmet.utils import logger as log

log.verbose = "DEBUG2"
np.set_printoptions(4, linewidth=1000, suppress=True)

vol_target = 40 
basis_ao = 'gth-dzvp'
cell_mesh = np.asarray([1, 1, 1], dtype=np.int)
exxdiv = None
total_mesh = np.asarray([4, 4, 4], dtype=np.int)
kmesh = total_mesh // cell_mesh

gdf_fname = './Sic%s%s%s_k%s%s%s_gdf_ints.h5'%(tuple(cell_mesh)+tuple(kmesh))
chkfname = './Sic%s%s%s_k%s%s%s_scf.chk'%(tuple(cell_mesh)+tuple(kmesh))

### ************************************************************
### System settings
### ************************************************************
latt_vec = np.array([[0.000000  ,   2.708337  ,   2.708337],
                     [2.708337  ,   0.000000  ,   2.708337],
                     [2.708337  ,   2.708337  ,   0.000000]])
vol_0 = abs(la.det(latt_vec))

scal_factor = (vol_target/vol_0) ** (1.0/3.0)
latt_vec *= scal_factor

vol = abs(la.det(latt_vec))
log.info("Target volume vol = %20.12f", vol)

atom_1 = np.array([0.375, 0.375, 0.375])
atom_2 = np.array([0.625, 0.625, 0.625])
coord_1 = latt_vec.dot(atom_1)
coord_2 = latt_vec.dot(atom_2)

max_memory = 255000 # 170 G
cell = gto.Cell()
cell.build(unit = 'angstrom',
           a = latt_vec,
           atom = 'Si %20.12f %20.12f %20.12f; Si %20.12f %20.12f %20.12f'%(coord_1[0], coord_1[1], coord_1[2],
               coord_2[0], coord_2[1], coord_2[2]),
           max_memory = max_memory,
           verbose = 5,
           pseudo = 'gth-pade',
           basis=basis_ao,
           precision=1e-11)

ncell_sc = np.prod(cell_mesh)
cell = tools.pbc.super_cell(cell, cell_mesh)
natom_sc = cell.natm

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

kmf_conv_tol = 1e-10
kmf_max_cycle = 300

### ************************************************************
### DMET settings 
### ************************************************************

# system
Filling = cell.nelectron / (Lat.nscsites*2.0)
restricted = True
bogoliubov = False
nscsites = Lat.nscsites
beta = np.inf

### ************************************************************
### SCF Mean-field calculation
### ************************************************************

log.section("\nSolving SCF mean-field problem\n")
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

Enuc = kmf.energy_nuc() / ncell_sc
Emf = (kmf.e_tot) / ncell_sc - Enuc
log.result("kmf nuclear energy: %20.12f", Enuc)
log.result("kmf electronic energy: %20.12f", Emf)

num_wann = 4
keywords = \
'''
num_iter = 400
begin projections
f=0.5, 0.5, 0.0:s
f=0.0, 0.5, 0.5:s
f=0.5, 0.0, 0.5:s
f=0.25, 0.25, 0.75:s
end projections
exclude_bands : 5-%s
'''%(kmf.cell.nao_nr())

# wannier run
w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords = keywords)
#w90.use_bloch_phases = True
#w90.use_scdm = True
#w90.guiding_centres = False
w90.kernel()

C_ao_mo = np.asarray(w90.mo_coeff)[:, :, w90.band_included_list]
C_mo_lo = make_basis.tile_u_matrix(np.array(w90.U_matrix.transpose(2, 0, 1), \
    order='C'), u_virt=None, u_core=None)
C_ao_lo = make_basis.multiply_basis(C_ao_mo, C_mo_lo)

# plot
from libdmet.utils.plot import plot_orb_k_all
plot_orb_k_all(cell, 'Si_wannier_val', C_ao_lo, Lat.kpts, \
        nx=50, ny=50, nz=50, resolution=None, margin=5.0)

