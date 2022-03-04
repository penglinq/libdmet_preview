#!/usr/bin/env python

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
from libdmet.basis_transform import make_basis
from libdmet.lo.iao import reference_mol

from libdmet.utils import logger as log
import libdmet.dmet.Hubbard as dmet

lib.param.TMPDIR = "/scratch/global/zhcui"
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

exxdiv = None
kmf_conv_tol = 1e-12
kmf_max_cycle = 300

gdf_fname = 'BN331_gdf_ints.h5'
chkfname = 'BN331.chk'

### ************************************************************
### DMET settings 
### ************************************************************

# system
Filling = cell.nelectron / (Lat.nscsites*2.0)
restricted = True
bogoliubov = False
int_bath = True
add_vcor = False
nscsites = Lat.nscsites
Mu = 0.0
last_dmu = 0.0
beta = np.inf

# DMET SCF control
MaxIter = 100
u_tol = 5.0e-5
E_tol = 5.0e-6 # energy diff per orbital
iter_tol = 4

# DIIS
adiis = lib.diis.DIIS()
adiis.space = 4
diis_start = 4
dc = dmet.FDiisContext(adiis.space)
trace_start = 3

# solver and mu fit
ncas = nscsites + nval
nelecas = (Lat.ncore+Lat.nval)*2
cc_etol = natom_sc * 1e-9
cc_ttol = natom_sc * 1e-6
cisolver = dmet.impurity_solver.CCSD(restricted=True, tol=cc_etol, \
        tol_normt=cc_ttol, max_memory=max_memory)
solver = cisolver
nelec_tol = 5.0e-6 # per orbital
delta = 0.01
step = 0.1
load_frecord = False

# vcor fit
imp_fit = False
emb_fit_iter = 200 # embedding fitting
full_fit_iter = 0
ytol = 1e-8
gtol = 1e-4 
CG_check = False

# vcor initialization
vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites)
z_mat = np.zeros((2, nscsites, nscsites))
vcor.assign(z_mat)

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

log.result("kmf electronic energy: %20.12f", (kmf.energy_tot()-kmf.energy_nuc())/ncell_sc)

### ************************************************************
### Pre-processing, LO and subspace partition
### ************************************************************

log.section("\nPre-process, orbital localization and subspace partition\n")
# IAO
S_ao_ao = kmf.get_ovlp()
C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao, full_return=True)

assert(nval == C_ao_iao_val.shape[-1])
C_ao_mo = np.asarray(kmf.mo_coeff)

# use IAO
C_ao_lo = C_ao_iao
Lat.set_Ham(kmf, gdf, C_ao_lo, eri_symmetry=4)

### ************************************************************
### DMET procedure
### ************************************************************

# DMET main loop
E_old = 0.0
conv = False
history = dmet.IterHistory()
dVcor_per_ele = None
if load_frecord:
    dmet.SolveImpHam_with_fitting.load("./frecord")

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)
    
    log.section("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    log.result("Mu (guess) = %20.12f", Mu)
    rho, Mu, res = dmet.RHartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True)
    Lat.update_Ham(rho*2.0, rdm1_lo_k=res["rho_k"]*2.0)

    log.section("\nconstructing impurity problem\n")
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching=True, int_bath=int_bath, add_vcor=add_vcor,\
            max_memory=max_memory)
    ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
    basis_k = Lat.R2k_basis(basis)
    
    log.section("\nsolving impurity problem\n")
    restart = False
    solver_args = {"restart": restart, "nelec": (Lat.ncore+Lat.nval)*2, \
            "dm0": dmet.foldRho_k(res["rho_k"], basis_k)*2.0}

    rhoEmb, EnergyEmb, ImpHam, dmu = \
        dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
        solver_args=solver_args, thrnelec=nelec_tol, \
        delta=delta, step=step)
    dmet.SolveImpHam_with_fitting.save("./frecord")
    last_dmu += dmu
    rhoImp, EnergyImp, nelecImp = \
        dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
        lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, \
        solver=solver, solver_args=solver_args)
    E_DMET_per_cell = EnergyImp*nscsites / ncell_sc
    log.result("last_dmu = %20.12f", last_dmu)
    log.result("E(DMET) = %20.12f", E_DMET_per_cell)
    
    # DUMP results:
    dump_res_iter = np.array([Mu, last_dmu, vcor.param, rhoEmb, basis, rhoImp, C_ao_lo], dtype=object)
    np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter)
    
    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, \
            vcor, beta, Filling, MaxIter1=emb_fit_iter, MaxIter2=full_fit_iter, method='CG', \
            imp_fit=imp_fit, ytol=ytol, gtol=gtol, CG_check=CG_check)

    if iter >= trace_start:
        # to avoid spiral increase of vcor and mu
        log.result("Keep trace of vcor unchanged")
        ddiagV = np.average(np.diagonal(\
                (vcor_new.get()-vcor.get())[:2], 0, 1, 2))
        vcor_new = dmet.addDiag(vcor_new, -ddiagV)

    dVcor_per_ele = np.max(np.abs(vcor_new.param - vcor.param))
    dE = EnergyImp - E_old
    E_old = EnergyImp 
    
    if iter >= diis_start:
        pvcor = adiis.update(vcor_new.param)
        dc.nDim = adiis.get_num_vec()
    else:
        pvcor = vcor_new.param
    
    dVcor_per_ele = np.max(np.abs(pvcor - vcor.param))
    vcor.update(pvcor)
    log.result("Trace of vcor: %20.12f ", \
            np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2)))
    
    history.update(E_DMET_per_cell, err, nelecImp, dVcor_per_ele, dc)
    history.write_table()
    
    # ZHC NOTE convergence criterion
    if dVcor_per_ele < u_tol and abs(dE) < E_tol and iter > iter_tol:
        conv = True
        break

if conv:
    log.result("DMET converge.")
else:
    log.result("DMET does not converge.")

### ************************************************************
### compare with KCCSD
### ************************************************************

log.section("Reference Energy")
mycc = cc.KCCSD(kmf)
mycc.kernel()
log.result("KRCCSD energy (per unit cell)")
log.result("%20.12f", mycc.e_tot - cell.energy_nuc())

