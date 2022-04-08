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
        self.nscsites = Lat.nscsites
        self.mu = 0
        self.last_dmu = 0.0
        self.beta = np.inf
        self.minao = 'minao'
        
        # DMET SCF control
        self.MaxIter = 100
        self.u_tol = 5.0e-5
        self.E_tol = 5.0e-6 # energy diff per orbital
        self.iter_tol = 4
        
        # DIIS
        self.adiis = lib.diis.DIIS()
        self.adiis.space = 4
        self.diis_start = 4
        self.dc = dmet.FDiisContext(adiis.space)
        self.trace_start = 3
        
        # solver and mu fit
        self.ncas = nscsites + nval
        self.nelecas = (Lat.ncore+Lat.nval)*2
        self.cc_etol = lat.cell.natm * 1e-9
        self.cc_ttol = lat.cell.natm * 1e-6
        self.solver = solver
        self.nelec_tol = 5.0e-6 # per orbital
        self.delta = 0.01
        self.step = 0.1
        self.load_frecord = False
        
        # vcor fit
        self.imp_fit = False
        self.emb_fit_iter = 200 # embedding fitting
        self.full_fit_iter = 0
        self.ytol = 1e-8
        self.gtol = 1e-4 
        self.CG_check = False
        
        # vcor initialization
        self.vcor = dmet.VcorLocal(self.restricted, self.bogoliubov, self.nscsites)
        self.z_mat = np.zeros((2, self.nscsites, self.nscsites))
        self.vcor.assign(z_mat)
        
    def kernel(self, kmf=None, Lat=None, solver=None):
        return kernel(self, kmf, Lat, solver)

    def construct_imp_ham():
        return 

    def solveImpHam_with_fitting():
        return

    def transform_results():
        return

    def fit_vcor():
        return
        
    def make_rdm1():
        return

    def make_rdm2():
        return

    def energy():
        """Return erergy per cell"""
        return



def kernel(mydmet, kmf=None, Lat=None, solver=None): 
    if kmf is None:
        kmf = mydmet._scf
    if Lat is None:
        Lat = mydmet.lat
    gdf = kmf.with_df
    Mu = mydmet.mu  ### TODO switch to lowercase
    Filling = mydmet.filling ### TODO switch to lowercase
    restricted = mydmet.restricted
    minao = mydmet.minao
    MaxIter = mydmet.MaxIter

    if solver is None:
        solver = mydmet.solver
    if solver == "FCI":
        solver = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-11)
    elif solver == "CCSD":
        self.cisolver = dmet.impurity_solver.CCSD(restricted=True, tol=mydmet.cc_etol, \
                tol_normt=mydmet.cc_ttol, max_memory=mydmet.max_memory)
    
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
        Lat.update_Ham(rho*2.0, rdm1_lo_k=res["rho_k"]*2.0)   # TODO add an option
    
        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching=True, int_bath=int_bath, add_vcor=add_vcor,\
                max_memory=max_memory)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)
        
        log.section("\nsolving impurity problem\n")
        restart = False
        solver_args = {"restart": restart, "nelec": (Lat.ncore+Lat.nval)*2, \
                "dm0": dmet.foldRho_k(res["rho_k"], basis_k)*2.0}  # TODO User input
    
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
    return E_DMET_per_cell,  conv
