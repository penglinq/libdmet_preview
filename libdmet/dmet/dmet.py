import os, sys
import numpy as np
import scipy.linalg as la

from pyscf import lib
from pyscf.pbc.lib import chkfile
from pyscf.pbc import scf, gto, df, cc, tools

from libdmet.system import lattice
from libdmet.basis_transform import make_basis
from libdmet.lo.iao import reference_mol
import libdmet.dmet.Hubbard as Hubbard
from libdmet.solver import impurity_solver

from libdmet.utils import logger as log

log.verbose = "DEBUG2"
np.set_printoptions(4, linewidth=1000, suppress=True)

class DMET():
    def __init__(self, kmf, lat, solver=None):
        # system
        self._scf = kmf
        self.lat = lat
        self.filling = lat.cell.nelectron / (lat.nscsites*2.0)
        self.restricted = True
        self.bogoliubov = False
        self.int_bath = True
        self.add_vcor = False
        self.mu = 0
        self.last_dmu = 0.0
        self.beta = np.inf
        self.minao = 'minao'
        self.max_memory = lat.cell.max_memory
        self.C_ao_lo = None
        self.update_ham = True
        
        # DMET SCF control
        self.MaxIter = 100
        self.u_tol = 5.0e-5
        self.E_tol = 5.0e-6 # energy diff per orbital
        self.iter_tol = 4
        
        # DIIS
        self.adiis = lib.diis.DIIS()
        self.adiis.space = 4
        self.diis_start = 4
        self.trace_start = 3
        self.dc = None
        
        # solver and mu fit
        self.ncas = lat.nscsites + lat.nval
        self.nelecas = (lat.ncore + lat.nval)*2
        self.solver = solver
        self.nelec_tol = 5.0e-6 # per orbital
        self.delta = 0.01
        self.step = 0.1
        self.load_frecord = False
        self.restart = False
        
        # vcor fit
        self.imp_fit = False
        self.emb_fit_iter = 200 # embedding fitting
        self.full_fit_iter = 0
        self.ytol = 1e-8
        self.gtol = 1e-4 
        self.CG_check = False
        self.vcor = None
        
    def kernel(self, kmf=None, Lat=None, solver=None):
        # vcor initialization
        if Lat is None:
            Lat = self.lat
        self.vcor = Hubbard.VcorLocal(self.restricted, self.bogoliubov, Lat.nscsites)
        z_mat = np.zeros((2, Lat.nscsites, Lat.nscsites))
        self.vcor.assign(z_mat)

        self.dc = Hubbard.FDiisContext(self.adiis.space)
        return kernel(self, kmf, Lat, solver)

    def make_rdm1():
        return

    def make_rdm2():
        return

    def energy():
        """Return erergy per cell"""
        return



def kernel(mydmet, kmf=None, Lat=None, solver=None): 
    '''
        Lat could be either a lattice object or a molecule object. 
        For a molecule object, Lat.is_mol == True.
        solver can be an instance of any subclass of SolverMixin. 
        'FCI' or 'CCSD' will initialize a default instance of the corresponding solver class. 
    '''
    if kmf is None:
        kmf = mydmet._scf
    if Lat is None:
        Lat = mydmet.lat
    Mu = mydmet.mu  ### TODO switch to lowercase
    Filling = mydmet.filling ### TODO switch to lowercase
    vcor = mydmet.vcor

    if solver is None:
        solver = mydmet.solver
    if solver == "FCI": # Use pyscf FCI with libDMET wrapper
        solver = impurity_solver.FCI(restricted=mydmet.restricted, tol=1e-11)
    elif solver == "CCSD": # Use pyscf CCSD with libDMET wrapper
        solver = impurity_solver.CCSD(restricted=mydmet.restricted, tol=1e-9, \
                tol_normt=1e-6, max_memory=mydmet.max_memory)
    
    ### ************************************************************
    ### Pre-processing, LO and subspace partition
    ### ************************************************************
    
    log.section("\nPre-process, orbital localization and subspace partition\n")
    # Localize orbital
    S_ao_ao = kmf.get_ovlp()
    if mydmet.C_ao_lo is None:
        # By default use IAO to localize orbitals
        C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=mydmet.minao, full_return=True)
        assert(Lat.nval == C_ao_iao_val.shape[-1])
        assert(Lat.nvirt == C_ao_iao_virt.shape[-1])
        C_ao_lo = C_ao_iao
    else:
        # Pass in predetermined LO, e.g. Wannier orbitals
        C_ao_lo = mydmet.C_ao_lo
    C_ao_mo = np.asarray(kmf.mo_coeff)
    Lat.set_Ham(kmf, kmf.with_df, C_ao_lo, eri_symmetry=4)
    
    ### ************************************************************
    ### DMET procedure
    ### ************************************************************
    
    # DMET main loop
    E_old = 0.0
    last_dmu = 0.0
    conv = False
    history = Hubbard.IterHistory()
    dVcor_per_ele = None
    if mydmet.load_frecord:
        Hubbard.SolveImpHam_with_fitting.load("./frecord")
    
    for iter in range(mydmet.MaxIter):
        log.section("\nDMET Iteration %d\n", iter)
        
        log.section("\nsolving mean-field problem\n")
        log.result("Vcor =\n%s", vcor.get())
        log.result("Mu (guess) = %20.12f", Mu)
        rho, Mu, res = Hubbard.RHartreeFock(Lat, vcor, Filling, Mu, beta=mydmet.beta, ires=True)
        if mydmet.update_ham:
            Lat.update_Ham(rho*2.0, rdm1_lo_k=res["rho_k"]*2.0) 
    
        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = Hubbard.ConstructImpHam(Lat, rho, vcor, matching=True, int_bath=mydmet.int_bath,\
                add_vcor=mydmet.add_vcor,\
                max_memory=mydmet.max_memory)
        ImpHam = Hubbard.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)
        
        log.section("\nsolving impurity problem\n")
        solver_args = {"restart": mydmet.restart, "nelec": mydmet.nelecas, \
                "dm0": Hubbard.foldRho_k(res["rho_k"], basis_k)*2.0} 
        #solver.update(solver_args)
    
        rhoEmb, EnergyEmb, ImpHam, dmu = \
            Hubbard.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            solver_args=solver_args, thrnelec=mydmet.nelec_tol, \
            delta=mydmet.delta, step=mydmet.step) # TODO remove solver_args 
        Hubbard.SolveImpHam_with_fitting.save("./frecord")
        last_dmu += dmu
        rhoImp, EnergyImp, nelecImp = \
            Hubbard.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
            lattice=Lat, last_dmu=last_dmu, int_bath=mydmet.int_bath, \
            solver=solver, solver_args=solver_args)
        E_DMET_per_cell = EnergyImp*Lat.nscsites / Lat.ncell_sc
        log.result("last_dmu = %20.12f", last_dmu)
        log.result("E(DMET) = %20.12f", E_DMET_per_cell)
        
        # DUMP results:
        dump_res_iter = np.array([Mu, last_dmu, vcor.param, rhoEmb, basis, rhoImp, C_ao_lo], dtype=object)
        np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter)
        
        log.section("\nfitting correlation potential\n")
        vcor_new, err = Hubbard.FitVcor(rhoEmb, Lat, basis, \
                vcor, mydmet.beta, Filling, MaxIter1=mydmet.emb_fit_iter, MaxIter2=mydmet.full_fit_iter, method='CG', \
                imp_fit=mydmet.imp_fit, ytol=mydmet.ytol, gtol=mydmet.gtol, CG_check=mydmet.CG_check) 
    
        if iter >= mydmet.trace_start:
            # to avoid spiral increase of vcor and mu
            log.result("Keep trace of vcor unchanged")
            ddiagV = np.average(np.diagonal(\
                    (vcor_new.get()-vcor.get())[:2], 0, 1, 2))
            vcor_new = Hubbard.addDiag(vcor_new, -ddiagV)
    
        dVcor_per_ele = np.max(np.abs(vcor_new.param - vcor.param))
        dE = EnergyImp - E_old
        E_old = EnergyImp 
        
        if iter >= mydmet.diis_start:
            pvcor = mydmet.adiis.update(vcor_new.param)
            mydmet.dc.nDim = mydmet.adiis.get_num_vec()
        else:
            pvcor = vcor_new.param
        
        dVcor_per_ele = np.max(np.abs(pvcor - vcor.param))
        vcor.update(pvcor)
        log.result("Trace of vcor: %20.12f ", \
                np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2)))
        
        history.update(E_DMET_per_cell, err, nelecImp, dVcor_per_ele, mydmet.dc)
        history.write_table()
        
        # ZHC NOTE convergence criterion
        if dVcor_per_ele < mydmet.u_tol and abs(dE) < mydmet.E_tol and iter > mydmet.iter_tol:
            conv = True
            break
    
    if conv:
        log.result("DMET converge.")
    else:
        log.result("DMET does not converge.")
    mydmet.mu = Mu
    return E_DMET_per_cell,  conv


if __name__ == "__main__":
    '''
    Example of DMET for boron nitride with gth-dzv basis.
    '''
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
    cell = tools.pbc.super_cell(cell, cell_mesh)
    natom_sc = cell.natm
    
    kmesh = [3, 3, 1]
    Lat = lattice.Lattice(cell, kmesh)
    Lat.ncell_sc = np.prod(cell_mesh)
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
    solver = impurity_solver.CCSD(restricted=True, tol=cc_etol, \
        tol_normt=cc_ttol, max_memory=max_memory) ## add direct ref to solver
    mydmet = DMET(kmf, Lat, solver)
    mydmet.minao = 'gth-szv'
    mydmet.max_memory = max_memory
    e_dmet, conv = mydmet.kernel() # e_dmet = energy per cell



