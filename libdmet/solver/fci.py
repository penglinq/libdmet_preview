#! /usr/bin/env python

"""
FCI impurity solver.

Author:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import numpy as np
import scipy.linalg as la
from libdmet.utils import logger as log
from pyscf.fci import direct_spin1, direct_uhf, cistring
import pyscf.lib.logger as pyscflogger
from libdmet.solver import scf
from libdmet.solver.scf import ao2mo_Ham, restore_Ham
from libdmet.basis_transform.make_basis import transform_rdm1_to_ao_mol, transform_rdm2_to_ao_mol 
from libdmet.utils.misc import mdot
from libdmet.solver.mixin import SolverMixin

class FCI(SolverMixin): # DMETsolverMixin
    def __init__(self, nproc=1, nnode=1, TmpDir="./tmp", SharedDir=None, \
            restricted=False, Sz=0, bcs=False, tol=1e-10, max_cycle=200, \
            max_memory=40000, compact_rdm2=False, scf_newton=True, ghf=False):
        """
        FCI solver.
        """
        self.restricted = restricted
        self.Sz = Sz
        self.bcs = bcs
        self.ghf = ghf
        self.conv_tol = tol
        self.max_memory = max_memory
        
        if (not self.restricted) and (not self.ghf):
            self.cisolver = direct_uhf.FCI()
        else:
            self.cisolver = direct_spin1.FCI()
        
        self.cisolver.max_memory = self.max_memory
        self.cisolver.max_cycle = max_cycle
        self.cisolver.conv_tol = self.conv_tol
        if log.Level[log.verbose] <= log.Level["INFO"]:
            self.cisolver.verbose = 4
        else:
            self.cisolver.verbose = 5
        self.scfsolver = scf.SCF(newton_ah=scf_newton)
        
        self.fcivec = None
        self.onepdm = None
        self.twopdm = None
        self.compact_rdm2 = compact_rdm2 # consider symm of rdm2
        self.optimized = False
    
    def run(self, Ham, nelec=None, guess=None, calc_rdm2=False, \
            pspace_size=800, Mu=None, **kwargs):
        """
        Main function of the solver.
        """
        log.info("FCI solver Run")
        spin = Ham.H1["cd"].shape[0]
        if spin > 1:
            assert not self.restricted
        if nelec is None:
            if self.bcs:
                nelec = Ham.norb * 2
            else:
                raise ValueError
        nelec_a = (nelec + self.Sz) // 2
        nelec_b = (nelec - self.Sz) // 2
        assert (nelec_a >= 0) and (nelec_b >=0) and (nelec_a + nelec_b == nelec)
        self.nelec = (nelec_a, nelec_b)

        # first do a mean-field calculation
        log.debug(1, "FCI solver: mean-field")
        dm0 = kwargs.get("dm0", None)
        scf_max_cycle = kwargs.get("scf_max_cycle", 200)

        self.scfsolver.set_system(nelec, self.Sz, False, self.restricted, \
                max_memory=self.max_memory)
        self.scfsolver.set_integral(Ham)
        E_HF, rhoHF = self.scfsolver.HF(tol=min(1e-9, self.conv_tol*0.1), \
                MaxIter=scf_max_cycle, InitGuess=dm0, Mu=Mu)

        log.debug(1, "FCI solver: mean-field converged: %s", self.scfsolver.mf.converged)
        log.debug(2, "FCI solver: mean-field rdm1: \n%s", self.scfsolver.mf.make_rdm1())

        Ham = ao2mo_Ham(Ham, self.scfsolver.mf.mo_coeff)
        if Ham.restricted: # RHF-FCI
            h1 = Ham.H1["cd"][0]
            h2 = Ham.H2["ccdd"][0]
        else: # UHF-FCI
            h1 = Ham.H1["cd"].copy()
            if Mu is not None:
                Mu_mat = np.eye(h1.shape[-1])
                nao = Mu_mat.shape[-1] // 2 
                Mu_mat[range(nao), range(nao)] = -Mu
                Mu_mat[range(nao, nao*2), range(nao, nao*2)] = Mu
                mo_coeff = self.scfsolver.mf.mo_coeff[0]
                Mu_mat = mdot(mo_coeff.conj().T, Mu_mat, mo_coeff)
                h1[0] += Mu_mat
                
            h2 = Ham.H2["ccdd"][[0, 2, 1]]
        
        # guess
        if guess == "random":
            na = cistring.num_strings(Ham.norb, nelec_a)
            nb = cistring.num_strings(Ham.norb, nelec_b)
            ci0 = np.random.random((na, nb)) - 0.5
            ci0 /= la.norm(ci0)
        else:
            ci0 = None
        
        # run
        E, self.fcivec = self.cisolver.kernel(h1, h2, Ham.norb, self.nelec,\
                ci0=ci0, ecore=Ham.H0, pspace_size=pspace_size)
        self.make_rdm1(Ham)
        if Mu is not None:
            E -= np.einsum('pq, qp', Mu_mat, self.onepdm_mo[0])
        if calc_rdm2:
            self.make_rdm2(Ham)
        
        self.optimized = True
        self.E = E
        log.info("FCI solver converged: %s", self.cisolver.converged)
        log.info("FCI total energy: %s", self.E)
        return self.onepdm, E
    
    def run_dmet_ham(self, Ham, last_aabb=True, **kwargs):
        """
        Run scaled DMET Hamiltonian.
        """
        log.info("FCI solver Run DMET Hamiltonian.")
        log.debug(0, "ao2mo for DMET Hamiltonian.")
        Ham = ao2mo_Ham(Ham, self.scfsolver.mf.mo_coeff, compact=True, in_place=True)
        Ham = restore_Ham(Ham, 1, in_place=True)
        
        # calculate rdm2 in aa, bb, ab order
        self.make_rdm2(Ham)
        if Ham.restricted:
            h1 = Ham.H1["cd"]
            h2 = Ham.H2["ccdd"]
            r1 = self.onepdm_mo
            r2 = self.twopdm_mo
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape
            
            E1 = np.einsum('pq, qp', h1[0], r1[0]) * 2.0
            E2 = np.einsum('pqrs, pqrs', h2[0], r2[0]) * 0.5
        else:
            h1 = Ham.H1["cd"]
            h2 = Ham.H2["ccdd"]
            r1 = self.onepdm_mo
            r2 = self.twopdm_mo
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape
            
            E1 = np.einsum('spq, sqp', h1, r1)
            
            E2_aa = 0.5 * np.einsum('pqrs, pqrs', h2[0], r2[0])
            E2_bb = 0.5 * np.einsum('pqrs, pqrs', h2[1], r2[1])
            E2_ab = np.einsum('pqrs, pqrs', h2[2], r2[2])
            E2 = E2_aa + E2_bb + E2_ab
        
        E = E1 + E2
        E += Ham.H0
        log.debug(0, "run DMET Hamiltonian:\nE0 = %20.12f, E1 = %20.12f, " 
                "E2 = %20.12f, E = %20.12f", Ham.H0, E1, E2, E)
        return E
    
    def make_rdm1(self, Ham):
        log.debug(1, "FCI solver: solve rdm1")
        if Ham.restricted:
            onepdm = self.cisolver.make_rdm1(self.fcivec, Ham.norb, self.nelec)
            self.onepdm_mo = (onepdm * 0.5)[np.newaxis]
        else:
            onepdm = self.cisolver.make_rdm1s(self.fcivec, Ham.norb, self.nelec)
            self.onepdm_mo = np.asarray(onepdm)
        
        # rotate back to the AO basis
        log.debug(1, "FCI solver: rotate rdm1 to AO")
        self.onepdm = transform_rdm1_to_ao_mol(self.onepdm_mo, \
                self.scfsolver.mf.mo_coeff)
    
    def make_rdm2(self, Ham, ao_repr=False):
        log.debug(1, "FCI solver: solve rdm2")
        if Ham.restricted:
            self.twopdm_mo = self.cisolver.make_rdm2(self.fcivec, Ham.norb, \
                    self.nelec)[np.newaxis]
        else:
            self.twopdm_mo = np.asarray(self.cisolver.make_rdm12s(self.fcivec, \
                    Ham.norb, self.nelec)[1])
        
        if ao_repr:
            log.debug(1, "FCI solver: rotate rdm2 to AO")
            self.twopdm = transform_rdm2_to_ao_mol(self.twopdm_mo, \
                    self.scfsolver.mf.mo_coeff)
        else:
            self.twopdm = None
        
        if not Ham.restricted and not self.ghf:
            self.twopdm_mo = self.twopdm_mo[[0, 2, 1]]
            if self.twopdm is not None:
                self.twopdm = self.twopdm[[0, 2, 1]]

    def onepdm(self):
        log.debug(1, "Compute 1pdm")
        return self.onepdm

    def twopdm(self):
        log.debug(1, "Compute 2pdm")
        return self.twopdm

    def cleanup(self):
        pass

class FCI_AO(SolverMixin):
    def __init__(self, nproc=1, nnode=1, TmpDir="./tmp", SharedDir=None, \
            restricted=False, Sz=0, bcs=False, ghf=False, tol=1e-10, max_cycle=200):
        """
        FCI solver in AO basis.
        """
        log.eassert(nnode == 1 or SharedDir is not None, \
                "Running on multiple nodes (nnod = %d), must specify shared directory", \
                nnode)
        if not restricted:
            self.cisolver = direct_uhf.FCI()
        else:
            self.cisolver = direct_spin1.FCI()
        self.restricted = restricted
        self.cisolver.max_cycle = max_cycle
        self.cisolver.conv_tol = tol
        self.cisolver.verbose = 5
        self.bcs = bcs
        self.ghf = ghf
        self.fcivec = None
        self.onepdm = None
        self.twopdm = None

    def run(self, Ham, nelec=None, guess=None, last_aabb=True, **kwargs):
        """
        Main function of the solver.
        """
        if last_aabb:
            order = [0, 2, 1]
        else:
            order = [0, 1, 2]

        if nelec is None:
            if self.bcs:
                nelec = Ham.norb * 2
            else:
                nelec = Ham.norb
        assert (Ham.restricted == self.restricted)
        if Ham.restricted:
            h1 = Ham.H1["cd"][0]
            h2 = Ham.H2["ccdd"][0]
        else:
            h1 = Ham.H1["cd"]
            h2 = Ham.H2["ccdd"][order]
        
        # guess
        if guess == "random":
            na = cistring.num_strings(Ham.norb, nelec//2)
            nb = cistring.num_strings(Ham.norb, nelec//2)
            ci0 = np.random.random((na, nb)) - 0.5
            ci0 /= la.norm(ci0)
        else:
            ci0 = None
        # run
        E, self.fcivec = self.cisolver.kernel(h1, h2, Ham.norb, nelec, ci0=ci0)
        log.info("FCI solver converged: %s", self.cisolver.converged)
        if Ham.restricted:
            onepdm, twopdm = self.cisolver.make_rdm12(self.fcivec, Ham.norb, nelec)
            self.onepdm = (onepdm * 0.5)[np.newaxis]
            self.twopdm = twopdm[np.newaxis]
        else:
            onepdm, twopdm = self.cisolver.make_rdm12s(self.fcivec, Ham.norb, nelec)
            self.onepdm = np.asarray(onepdm)
            self.twopdm = np.asarray(twopdm)[order]
        E += Ham.H0
        return self.onepdm, E
    
    def run_dmet_ham(self, Ham, last_aabb=True, **kwargs):
        """
        Run scaled DMET Hamiltonian.
        """
        Ham = restore_Ham(Ham, 1, in_place=True)
        if last_aabb: # last is aabb, move it to middle
            order = [0, 2, 1]
        else:
            order = [0, 1, 2]
        
        if self.ghf:
            h1 = Ham.H1["cd"][0]
            h2 = Ham.H2["ccdd"][0]
            r1 = self.onepdm
            r2 = self.twopdm
            E = np.tensordot(h1, r1, axes=((0,1), (1,0))) + \
                    np.tensordot(h2, r2, axes=((0,1,2,3), (0,1,2,3)))*0.5
        elif Ham.restricted:
            h1 = Ham.H1["cd"]
            h2 = Ham.H2["ccdd"]
            r1 = self.onepdm
            r2 = self.twopdm
            E = np.tensordot(h1[0], r1[0], axes=((0,1), (1,0)))*2.0 + \
                    np.tensordot(h2[0], r2[0], axes=((0,1,2,3), (0,1,2,3)))*0.5
        else:
            h1 = Ham.H1["cd"]
            h2 = Ham.H2["ccdd"][order]
            r1 = self.onepdm
            r2 = self.twopdm[order]
            E1 = np.tensordot(h1, r1, axes=((0,1,2), (0,2,1)))
            E2_a = 0.5 * np.tensordot(r2[0], h2[0], axes=((0,1,2,3), (0,1,2,3)))
            E2_ab = np.tensordot(r2[1], h2[1], axes=((0,1,2,3), (0,1,2,3)))
            E2_b = 0.5 * np.tensordot(r2[2], h2[2], axes=((0,1,2,3), (0,1,2,3)))
            E = E1 + E2_a + E2_b + E2_ab 
        E += Ham.H0
        return E

    def onepdm(self):
        log.debug(1, "Compute 1pdm")
        return self.onepdm

    def twopdm(self):
        log.debug(1, "Compute 2pdm")
        return self.twopdm

    def cleanup(self):
        pass
