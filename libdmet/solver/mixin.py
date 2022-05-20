#! /usr/bin/env python

"""
Wrapper to all impurity solvers.

Author:
    Linqing Peng
"""

from libdmet.solver import block
from libdmet.solver.afqmc import AFQMC
from libdmet.system import integral
from libdmet.utils import logger as log
import numpy as np
import scipy.linalg as la
from pyscf import __config__

__all__ = ["AFQMC", "Block", "StackBlock", "Block2", "DmrgCI", "CASSCF", "BCSDmrgCI",
           "FCI", "FCI_AO", "CCSD", "SCFSolver"]
try:
    from libdmet.solver.shci import SHCI
    __all__.append("SHCI")
except ImportError:
    log.info("ImportError in SHCI solver, settings.py should be set in pyscf")

class DMETSolverMixin(object):
    tmpdir = getattr(__config__, 'TMPDIR', '.')
    verbose = getattr(__config__, 'VERBOSE', 3)
    max_memory = getattr(__config__, 'MAX_MEMORY', 4000)
    # suggest optional functions

    def __init__(self):
        self.conv_tol = 1e-9  # Add a general warning TODO
        self.max_cycle = 100
        self.converged = None
        self.onepdm = None
        self.onepdm_mo = None
        self.twopdm = None
        self.twopdm_mo = None
    
    @property 
    def e_tot(self):
        raise NotImplementedError

    def make_rdm1(self, ao_repr=False):
        '''
            Need to provide ao_repr 
        '''
        raise NotImplementedError

    def make_rdm2(self, ao_repr=False):
        raise NotImplementedError

    def run(self, Ham, nelec=None, calc_rdm2=False, ):
        raise NotImplementedError

    def run_dmet_ham(self, Ham, **kwargs):
        """
        Run scaled DMET Hamiltonian.
        Assume Ham in MO and symmetry=1
        Need to be modified to accommodate different Ham.
        """
        log.info("Solver Mixin Run DMET Hamiltonian.")
        log.debug(0, "ao2mo for DMET Hamiltonian.")
        
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

    def dump_flags(self, verbose=None):
        raise NotImplementedError

