#!/usr/bin/env python

"""
Make embedding basis, Hamiltonian for Slater determinant.

Author:
    Zhi-Hao Cui
    Bo-Xiao Zheng
"""

import copy
import h5py
import numpy as np
import scipy.linalg as la
import itertools as it
from math import sqrt
from tempfile import NamedTemporaryFile
from pyscf import ao2mo
from pyscf import lib

from libdmet.utils import logger as log
from libdmet.system import integral
from libdmet import settings
from libdmet.routine import localizer
from libdmet.routine import ftsystem
from libdmet.routine import mfd
from libdmet.routine.fit import minimize
from libdmet.routine import pbc_helper as pbc_hp
from libdmet.solver.scf import _get_jk, _get_veff, restore_Ham
from libdmet.basis_transform import make_basis
from libdmet.basis_transform.eri_transform import get_emb_eri, get_unit_eri
from libdmet.lo.lowdin import vec_lowdin
from libdmet.routine.slater_helper import *
from libdmet.utils.misc import *

def MatSqrt(M):
    """
    M = UA^{1/2}
    """
    log.eassert(max_abs(M - M.T.conj()) < 1e-10, "matrix must be symmetric")
    ew, ev = la.eigh(M)
    if ew[0] < 0:
        ew += 1e-11
    log.eassert((ew >= 0).all(), "matrix must be positive definite")
    log.check(ew[0] > 1e-10, "small eigenvalue for rho_imp,"
        "cut-off is recommended\nthe first 5 eigenvalues are %s", ew[:5])
    ewsq = np.sqrt(ew).real
    return np.dot(ev, np.diag(ewsq))

def normalizeBasis(b):
    # simple array
    ovlp = np.dot(b.T, b)
    log.debug(1, "basis overlap is\n%s", ovlp)
    n = np.diag(1./np.sqrt(np.diag(ovlp)).real)
    return np.dot(b, n)

def normalizeBasis1(b):
    # array in blocks
    ovlp = np.tensordot(b, b, axes = ((0,1), (0,1)))
    log.debug(0, "basis norm is\n%s", np.diag(ovlp))
    norms = np.diag(1./np.sqrt(np.diag(ovlp)))
    return np.tensordot(b, norms, axes = (2,0))

def orthonormalizeBasis(b):
    nbasis = b.shape[-1]
    ovlp = np.tensordot(b, b, axes = ((0,1), (0,1)))
    log.debug(1, "basis overlap is\n%s", ovlp)
    if np.allclose(ovlp - np.diag(np.diag(ovlp)), 0.):
        return normalizeBasis1(b)
    else:
        ew, ev = la.eigh(ovlp)
        ew = ew[::-1]
        ev = ev[:, ::-1]
        b = np.tensordot(b, ev, axes = (2, 0))
        b = np.tensordot(b, np.diag(ew**(-0.5)), axes = (2, 0))
        return b

def getNonDiagBlocks(mat):
    log.eassert(max_abs(mat - mat.T) < 1e-10, "Input matrix is not symmetric")
    nonzero = np.asarray(np.nonzero(abs(mat - np.diag(np.diag(mat))) > 1e-12)).T
    nonzero = nonzero[:nonzero.shape[0] // 2].tolist()
    blocks = []
    for pair in nonzero:
        found = False
        for b in blocks:
            if pair[0] in b:
                b.add(pair[1])
                found = True
            elif pair[1] in b:
                b.add(pair[0])
                found = True
        if not found:
            blocks.append(set(pair))
    return [sorted(list(b)) for b in blocks]

def get_emb_basis(lattice, rho=None, local=True, **kwargs):
    """
    Get embedding basis for slater determinant.
    """
    if rho is None:
        rho = lattice.rdm1_lo_R
    if local:
        return __embBasis_proj(lattice, rho, **kwargs)
    else:
        return __embBasis_phsymm(lattice, rho, *kwargs)

embBasis = get_emb_basis

def __embBasis_proj(lattice, rdm1, **kwargs):
    """
    Get embedding basis, C_lo_eo, using SVD projection.
    
    Args:
        lattice: lattice object.
        rdm1: density matrix, shape ((spin,) ncells, nlo, nlo)

    Kwargs:
        imp_idx: impurity indices.
        val_idx: valence indices.
        valence_bath: whether to use valence bath or full bath.
        orth: True, orthogonalize bath by projecting out the virtual orbitals.
              if orth == False and valence_bath, the basis can be non-orthonal.
        tol_bath: 1e-9, tolerance for discarding small singular values.
        nbath: number of bath.
    
    Returns:
        basis: C_lo_eo, shape ((spin,), ncells, nlo, neo)
    """
    imp_idx      = kwargs.get("imp_idx", lattice.imp_idx)
    val_idx      = kwargs.get("val_idx", lattice.val_idx)
    valence_bath = kwargs.get("valence_bath", True)
    orth         = kwargs.get("orth", True)
    tol_bath     = kwargs.get("tol_bath", 1e-9)
    nbath        = kwargs.get("nbath", None)

    ncells = lattice.ncells
    nlo    = lattice.nscsites
    # imp_idx for bath construction
    imp_idx_bath = val_idx if valence_bath else imp_idx
    env_idx = []
    # boolean mask of virtual in the env_idx
    virt_mask = [] 
    for i in range(ncells * nlo):
        if not i in imp_idx_bath:
            env_idx.append(i)
            virt_mask.append(i in imp_idx)
    nimp = len(imp_idx)
    log.debug(0, "imp_idx for bath : %-15s [val  : %s] ", \
            format_idx(imp_idx_bath), format_idx(val_idx))
    log.debug(0, "env_idx for bath : %-15s [virt : %s] ", \
            format_idx(env_idx), format_idx(np.array(env_idx)[virt_mask]))

    rdm1 = np.asarray(rdm1)
    if rdm1.ndim == 3:
        rdm1 = rdm1[np.newaxis]
    assert rdm1.shape[-3:] == (ncells, nlo, nlo)
    spin = rdm1.shape[0]
    
    if np.max(imp_idx_bath) >= nlo - 1: 
        # some imp_idx_bath is outside of first cell
        rdm1_env_imp = lattice.expand(rdm1)\
                                [:, env_idx][:, :, imp_idx_bath]
        nbath_final = len(imp_idx_bath)
    else:
        rdm1_env_imp = rdm1.reshape(spin, ncells * nlo, nlo)\
                                [:, env_idx][:, :, imp_idx_bath]
        nbath_final = nlo
    basis = np.zeros((spin, ncells * nlo, nimp * 2))
    
    for s in range(spin):
        # SVD
        u, sigma, vt = la.svd(rdm1_env_imp[s], full_matrices=False)
        if nbath is None:
            nbath_s = (sigma >= tol_bath).sum()
        else:
            nbath_s = nbath
        B = u[:, :nbath_s]

        # check zero singular values
        with np.printoptions(suppress=False):
            log.debug(1, "Singular values:\n%s", sigma[:nbath_s])
            log.debug(1, "Singular values discarded:\n%s", sigma[nbath_s:])
        nzero = np.sum(np.abs(sigma[:nbath_s]) < tol_bath)
        log.debug(0, "Zero singular values number: %s", nzero)
        if nzero > 0:
            log.warn("Zero singular value exists, \n"
                     "this may cause numerical instability.")
         
        if nbath_s > 0:
            # project out the local virtual component
            # this is actually only used when valence_bath is true.
            if orth:
                B[virt_mask] = 0.0
                B = vec_lowdin(B, np.eye(B.shape[0]))
            
            # localization of bath
            loc_method = kwargs.get("localize_bath", None)
            if loc_method is not None:
                if not lattice.is_model:
                    log.warn("Only model is currently supported" \
                             "for localization of bath.")
                B = localizer.localize_bath(B, method=loc_method)
        
        basis[s, imp_idx, :nimp] = np.eye(nimp)
        basis[s, env_idx, nimp:nimp + nbath_s] = B
        nbath_final = min(nbath_final, nbath_s)
    
    log.debug(0, "nimp : %d", nimp)
    log.debug(0, "nbath: %d", nbath_final)
    basis = basis[:, :, :nimp + nbath_final]\
            .reshape(spin, ncells, nlo, nimp + nbath_final)
    return basis

def get_emb_Ham(lattice, basis, vcor, local=True, **kwargs):
    """
    Get embedding Hamiltonian.

    Args:
        lattice: lattice object.
        basis: C_lo_eo, (spin, ncells, nlo, neo).
        vcor: vcor object.
        local: indicate basis is local, deprecated.
    
    Kwargs:
        incore  : [True] using incore routine for ERI transform.
        H2_given: [None] manully specify a np.array form of H2.
        H2_fname: [None] load H2 from a .h5 file with name of H2_fname.
    
    Returns:
        ImpHam: a integral object if incore == True, 
                else a integral.H2 will be a H5File.
        H1e: deprecated.
    """
    basis  = np.asarray(basis)
    spin   = basis.shape[0]
    nbasis = basis.shape[-1]
    
    # First transform two-body, 
    # since we need ERI to calculate JK_emb for interacting bath.
    log.info("Two-body part")
    H2_given = kwargs.get("H2_given", None)
    if H2_given is None:
        H2_fname = kwargs.get("H2_fname", None)
        if H2_fname is None:
            H2 = __embHam2e(lattice, basis, vcor, local, **kwargs)
        else:
            log.debug(1, "Load H2 from %s", H2_fname)
            with h5py.File(H2_fname, 'r') as feri:
                H2 = np.asarray(feri['emb_eri'])
    else:
        log.debug(1, "Using specified H2 array.")
        H2 = H2_given
    
    log.info("One-body part")
    H1, ovlp_emb = __embHam1e(lattice, basis, vcor, H2, **kwargs)
    H0 = lattice.getH0()
    if isinstance(H2, np.ndarray):
        H2 = {"ccdd": H2}

    ImpHam = integral.Integral(nbasis, spin == 1, False, H0, {"cd": H1}, 
                               H2, ovlp=ovlp_emb)
    return ImpHam, None

embHam = get_emb_Ham

def __embHam2e(lattice, basis, vcor, local, int_bath=True, last_aabb=True, **kwargs):
    """
    Internal routine to construct H2_emb.
    """
    nscsites = lattice.nscsites
    nbasis = basis.shape[-1]
    spin = basis.shape[0] 
    eri_symmetry = lattice.eri_symmetry 
    max_memory = kwargs.get("max_memory", None)

    if lattice.is_model:
        LatH2 = lattice.getH2(compact=False, kspace=False)
        if settings.save_mem:
            if local:
                return {"ccdd": Lat_H2[np.newaxis]}
            else:
                log.warning("Basis nonlocal, ignoring memory saving option")
                settings.save_mem = False

        if kwargs.get("mmap", False):
            log.debug(0, "Use memory map for 2-electron integral")
            H2 = np.memmap(NamedTemporaryFile(dir=TmpDir), \
                dtype=float, mode='w+', shape=(spin*(spin+1)//2, nbasis, nbasis, nbasis, nbasis))

        if local:
            #for s in range(spin):
            #    log.eassert(max_abs(basis[s, 0, :, :nscsites] - np.eye(nscsites)) < 1e-10, \
            #        "the embedding basis is not local")

            if lattice.H2_format == 'local':
                if int_bath:
                    H2 = transform_eri_local(basis, lattice, LatH2) 
                else:
                    H2 = unit2emb(np.asarray((LatH2,) * (spin*(spin+1)//2)), nbasis)
            elif lattice.H2_format == "nearest":
                if int_bath:
                    raise NotImplementedError
                else:
                    H2 = np.zeros((spin*(spin+1)//2, nbasis, nbasis, nbasis, nbasis))
                    for i in range(H2.shape[0]):
                        H2[i, :nscsites, :nscsites, :nscsites, :nscsites] = LatH2[0]
            elif lattice.H2_format == "full":
                if int_bath:
                    raise NotImplementedError
                else:
                    H2 = np.zeros((spin*(spin+1)//2, nbasis, nbasis, nbasis, nbasis))
                    for i in range(H2.shape[0]):
                        H2[i, :nscsites, :nscsites, :nscsites, :nscsites] = LatH2[0, 0, 0]
            elif lattice.H2_format == "spin local":
                if int_bath:
                    raise NotImplementedError 
                else:
                    H2 = np.zeros((spin*(spin+1)//2, nbasis, nbasis, nbasis, \
                            nbasis))
                    for i in range(H2.shape[0]):
                        H2[i, :nscsites, :nscsites, :nscsites, :nscsites] = \
                                LatH2[i]
            else:
                raise ValueError
        else:
            log.eassert(lattice.H2_format == "local", \
                    "non local bath currently only support local lattice ERI.")
            H2 = np.zeros((spin*(spin+1)//2, nbasis, nbasis, nbasis, nbasis))
            H2[0] = transform_4idx(Lat_H2, basis[0, 0], basis[0, 0], \
                    basis[0, 0], basis[0, 0])
            H2[1] = transform_4idx(Lat_H2, basis[1, 0], basis[1, 0], \
                    basis[1, 0], basis[1, 0])
            H2[2] = transform_4idx(Lat_H2, basis[0, 0], basis[0, 0], \
                    basis[1, 0], basis[1, 0])
    elif getattr(lattice, "is_mol", None) and lattice.is_mol: 
        raise NotImplementedError
    else: # ab initio system
        cell = lattice.cell
        mydf = lattice.df
        C_ao_lo = lattice.C_ao_lo
        swap_idx = kwargs.get("swap_idx", None)
        kscaled_center = kwargs.get("kscaled_center", None)
        t_reversal_symm = kwargs.get("t_reversal_symm", True)
        incore = kwargs.get("incore", True)
        fout = kwargs.get("fout", "H2.h5")
        
        if int_bath: # interacting bath
            H2 = get_emb_eri(cell, mydf, C_ao_lo=C_ao_lo, basis=basis, 
                             kscaled_center=kscaled_center, 
                             symmetry=eri_symmetry, max_memory=max_memory, 
                             swap_idx=swap_idx, 
                             t_reversal_symm=t_reversal_symm, incore=incore,
                             fout=fout)
            if last_aabb and isinstance(H2, np.ndarray) and H2.shape[0] == 3:
                H2 = H2[[0, 2, 1]]
        else: # non-interacting bath
            H2 = get_unit_eri(cell, mydf, C_ao_lo=C_ao_lo, 
                              kscaled_center=kscaled_center, 
                              symmetry=eri_symmetry, max_memory=max_memory, 
                              swap_idx=swap_idx, 
                              t_reversal_symm=t_reversal_symm, incore=incore,
                              fout=fout)
            if last_aabb and isinstance(H2, np.ndarray) and H2.shape[0] == 3:
                H2 = H2[[0, 2, 1]]
            H2 = unit2emb(H2, nbasis)
    
    if isinstance(H2, np.ndarray):
        log.info("H2 memory allocated size = %d MB", H2.size * 8. / 1024 / 1024)
    return H2

def get_veff(rdm1, eri, hyb=1.0):
    """
    Get effetive potential for the embedding Hamiltonian.
    
    Args:
        rdm1: rdm1, if restricted, it is spin traced.
        eri:  eri, can have spin dimension 1 or 3.
        hyb: coefficient before the K matrix.
    """
    rdm1 = np.asarray(rdm1)
    if rdm1.ndim == 2:
        rdm1 = rdm1[None]
    spin = rdm1.shape[0]
    if hyb == 1.0: # HF
        veff = _get_veff(rdm1, eri)
    elif hyb == 0.0: # pure DFT, J only
        vj = _get_jk(rdm1, eri, with_j=True, with_k=False)[0]
        if spin == 1:
            veff = vj
        else:
            veff = vj[0] + vj[1]
    else: # hybrid DFT
        vj, vk = _get_jk(rdm1, eri, with_j=True, with_k=True)
        if spin == 1:
            veff = vj - vk * (hyb * 0.5)
        else:
            veff = vj[0] + vj[1] - vk * hyb
    return veff

def __embHam1e(lattice, basis, vcor, H2_emb, int_bath=True, add_vcor=False, **kwargs):
    """
    Internal routine to construct H1_emb.
    """
    ncells = lattice.ncells
    nscsites = lattice.nscsites
    spin = basis.shape[0]
    nbasis = basis.shape[-1]
    basis_k = lattice.R2k_basis(basis) 
    hcore_k = lattice.getH1(kspace=True)
    fock_k  = lattice.getFock(kspace=True)
    ovlp_k  = lattice.get_ovlp(kspace=True)
    JK_imp = lattice.get_JK_imp()
    if not isinstance(H2_emb, np.ndarray):
        H2_emb = np.asarray(H2_emb["ccdd"])

    log.debug(1, "transform hcore")
    hcore_emb = transform_h1(hcore_k, basis_k) 
    log.debug(1, "transform ovlp")
    ovlp_emb = transform_h1(ovlp_k, basis_k)
    if ovlp_emb.ndim == 3 and ovlp_emb.shape[0] == 1:
        ovlp_emb = ovlp_emb[0]

    dft = kwargs.get("dft", False)
    vxc_dc = kwargs.get("vxc_dc", False)
    vhf_for_energy = kwargs.get("vhf_for_energy", False)
    if dft:
        hyb = pbc_hp.get_hybrid_param(lattice.kmf)[-1]
        vxc_emb = transform_h1(lattice.vxc_lo_k, basis_k)
    else:
        hyb = 1.0
        vxc_loc = vxc_emb = None

    if int_bath:
        rdm1_emb = foldRho_k(lattice.rdm1_lo_k, basis_k)
        if dft:
            if vxc_dc:
                vxc_loc = get_vxc_loc(lattice, rdm1_emb, lattice.C_ao_lo,
                        basis_k)
            else:
                vxc_loc = vxc_emb
            # J + scaled K + vxc
            veff_emb = transform_h1(lattice.veff_lo_k, basis_k)
            
            H1 = hcore_emb + veff_emb 
            if vxc_loc is not None:
                log.debug(1, "remove the contribution of vxc")
                H1 -= vxc_loc
            
            # after subtract vxc_emb, H1 has hcore, veff_loc, veff_core
            # now subtract veff_loc (i.e. local J + scaled K)
            log.debug(1, "Construct JK_emb")
            veff_loc = get_veff(rdm1_emb, H2_emb, hyb=hyb)
            H1 -= veff_loc
            
            log.debug(1, "Construct JK_core")
            if vhf_for_energy:
                # the JK core used for energy evaluation is HF one
                vhf_emb = transform_h1(lattice.vhf_lo_k, basis_k)
                vhf_loc = get_veff(rdm1_emb, H2_emb, hyb=1.0)
                vhf_emb -= vhf_loc
                lattice.JK_core = vhf_emb
            else:
                # only J and scaled K are used for energy
                lattice.JK_core = H1 - hcore_emb
        else:
            log.debug(1, "transform fock")
            if not lattice.is_model:
                fock_k = lattice.hcore_lo_k + lattice.vhf_lo_k
            H1 = transform_h1(fock_k, basis_k)
            
            log.debug(1, "Construct JK_emb")
            JK_emb = get_veff(rdm1_emb, H2_emb)
            
            # Subtract JK_emb
            # i.e. rho_kl[2(ij||kl)-(il||jk)] 
            # where i,j,k,l are all embedding basis.
            H1 -= JK_emb
            
            log.debug(1, "Construct JK_core")
            JK_core = H1 - hcore_emb
             
            # save JK_core for energy evaluation
            lattice.JK_core = JK_core
    else:
        # NIB
        add_vcor = True
        if lattice.use_hcore_as_emb_ham:
            log.debug(1, "Use hcore as embedding Hamiltonian.")
            H1 = hcore_emb
            # NIB and use hcore, do not include JK_core in energy.
            lattice.JK_core = None
        else: # NIB but use fock as embedding hamiltonian.
            log.debug(1, "transform fock")
            H1 = transform_h1(fock_k, basis_k)
            
            log.debug(1, "Construct JK_emb")
            # subtract impurity Fock if necessary
            # i.e. rho_kl[2(ij||kl)-(il||jk)] where i,j,k,l are all impurity
            if JK_imp is not None:
                log.debug(1, "used predefined impurity JK")
                if JK_imp.ndim == 2:
                    JK_emb = np.asarray([transform_imp(basis[s], lattice, \
                            JK_imp) for s in range(spin)])
                else:
                    JK_emb = np.asarray([transform_imp(basis[s], lattice, \
                            JK_imp[s]) for s in range(spin)])
            else:
                rdm1_emb = foldRho_k(lattice.rdm1_lo_k, basis_k)
                JK_emb = get_veff(rdm1_emb, H2_emb)
            H1 -= JK_emb
            
            log.debug(1, "Construct JK_core")
            JK_core = H1 - hcore_emb
            lattice.JK_core = JK_core
    
    if add_vcor:
        log.eassert(vcor.islocal(), \
                "nonlocal correlation potential cannot be treated in this routine")
        for s in range(spin):
            # ZHC FIXME vcor should be included in the interacting bath?
            # then add Vcor only in environment
            # need to subtract impurity contribution
            log.debug(1, "transform Vcor, spin: %s", s)
            H1[s] += transform_local(basis[s], lattice, vcor.get()[s])
            if not "fitting" in kwargs or not kwargs["fitting"]:
                # for fitting purpose, we need H1 with vcor on impurity
                H1[s] -= transform_imp(basis[s], lattice, vcor.get()[s])
    return H1, ovlp_emb

def transform_h1(H1_k, basis_k):
    spin = basis_k.shape[0]
    nbasis = basis_k.shape[-1]
    H1_k = add_spin_dim(H1_k, spin, non_spin_dim=3)
    H1 = np.empty((spin, nbasis, nbasis)) 
    for s in range(spin):
        H1[s] = transform_trans_inv_k(basis_k[s], H1_k[s])
    return H1

def foldRho(rho, lattice, basis):
    """
    Transform density matrix to embedding space.
    """
    log.warn("foldRho is slow. You can try foldRho_k(rho_k, basis_k).")
    spin = rho.shape[0]
    nbasis = basis.shape[-1]
    rdm1_emb = np.empty((spin, nbasis, nbasis)) 
    for s in range(spin):
        rdm1_emb[s] = transform_trans_inv(basis[s], lattice, rho[s])
    return rdm1_emb

# k version of foldRho (rho_k, basis_k)
foldRho_k = transform_h1

def get_vxc_loc(lattice, rdm1_emb, C_ao_lo, C_lo_eo=None):
    """
    Get vxc for the embedding density.
    First transform rdm1 in the embedding space to the supercell lattice AO basis.
    
    Args:
        lattice: lattice
        rdm1_emb: rdm1 in the embedding space, ((spin,), nkpts, neo, neo)
        C_ao_lo: C_ao_lo, ((spin,), nkpts, nao, nlo), in k space
        C_lo_eo: C_lo_eo, ((spin,), nkpts, nlo, neo), in k space, 
                 if None, will use C_ao_lo as C_ao_eo
        
    Returns:
        vxc_loc: (spin, neo, neo)
    """
    from libdmet.basis_transform import make_basis
    log.info("get_vxc_loc: construct rdm1_AO")
    C_ao_lo = np.asarray(C_ao_lo)
    rdm1_emb = np.asarray(rdm1_emb)
    if rdm1_emb.ndim == 2:
        rdm1_emb = rdm1_emb[None]
    spin = rdm1_emb.shape[0]
    if C_lo_eo is None:
        C_ao_eo = C_ao_lo
    else:
        C_lo_eo = np.asarray(C_lo_eo)
        C_ao_eo = make_basis.multiply_basis(C_ao_lo, C_lo_eo)
    C_ao_eo = add_spin_dim(C_ao_eo, spin)
    spin, nkpts, nao, neo = C_ao_eo.shape
    
    C_AO_eo = lattice.k2R_basis(C_ao_eo).reshape(spin, nkpts*nao, neo)
    rdm1_AO = np.empty((spin, nkpts*nao, nkpts*nao), \
            dtype=np.result_type(C_AO_eo.dtype, rdm1_emb.dtype))
    for s in range(spin):
        rdm1_AO[s] = mdot(C_AO_eo[s], rdm1_emb[s], C_AO_eo[s].conj().T)
    log.info("get_vxc_loc: construct vxc_AO")
    vxc_AO = pbc_hp.get_vxc(lattice.kmf_sc, dm=rdm1_AO[:, None])
    log.info("get_vxc_loc: construct vxc_loc")
    vxc_loc = np.empty((spin, neo, neo))
    for s in range(spin):
        vxc_loc[s] = mdot(C_AO_eo[s].conj().T, vxc_AO[s], C_AO_eo[s])
    return vxc_loc

def addDiag(v, val, idx_range=None):
    """
    Add elements to the diagonal elements of v.

    Args:
        v: vcor.
        val: can be a value, or a list of values (per spin sector).
        idx_range: indices for adding.

    Returns:
        v: updated vcor.
    """
    rep = v.get()
    spin = rep.shape[0]
    if not isinstance(val, Iterable):
        val = [val for s in range(spin)]
    if idx_range is None:
        idx_range = getattr(v, "idx_range", range(rep.shape[-1]))
    # for BCS type vcor, the third is not diagonal block.
    for s in range(min(spin, 2)):
        rep[s, idx_range, idx_range] += val[s]
    v.assign(rep)
    return v

def vcor_diag_average(v, idx_range=None):
    """
    Compute the average value of diagonal element of vcor.

    Args:
        v: vcor
        idx_range: indices for averaging.

    Returns:
        ave: shape (spin,), the average of diagonals at each spin sector.
    """
    rep = v.get()
    if idx_range is None:
        idx_range = getattr(v, "idx_range", range(rep.shape[-1]))
    return np.average(rep[:, idx_range, idx_range], axis=1)

def make_vcor_trace_unchanged(v_new, v_old, idx_range=None):
    """
    Make trace of v_new the same as v_old.

    Args:
        v_new: new vcor
        v_old: old vcor
        idx_range: indices for considering.

    Returns:
        v_new: updated v_new.
    """
    v_mat_old = v_old.get()
    v_mat_new = v_new.get()
    spin = v_mat_new.shape[0]
    nao  = v_mat_new.shape[-1]
    if idx_range is None:
        idx_range = getattr(v_new, "idx_range", range(nao))
    dv_ave = np.average((v_mat_new - v_mat_old)[:, idx_range, idx_range], \
            axis=1)
    addDiag(v_new, -dv_ave, idx_range=idx_range)
    return v_new

def test_grad(vcor, errfunc, gradfunc, dx=1e-5):
    """
    Test analytic gradient and compare with numerical one.
    """
    if isinstance(vcor, np.ndarray):
        param0 = vcor
    else:
        param0 = vcor.param.copy()
    grad_num = np.zeros_like(param0)
    grad_ana = gradfunc(param0)
    f0 = errfunc(param0)
    for i in range(len(grad_num)):
        param_tmp = param0.copy()
        param_tmp[i] += dx
        grad_num[i] = (errfunc(param_tmp) - f0) / dx
    
    log.info("Test gradients in fitting, finite difference dx = %s", dx)
    log.info("Analytical gradient:\n%s", grad_ana)
    log.info("Numerical gradient:\n%s",  grad_num)
    log.info("grad_ana / grad_num:\n%s", grad_ana / grad_num)
    log.info("Norm difference: %s", la.norm(grad_ana - grad_num))

def get_dV_dparam(vcor, basis, basis_k, lattice, P_act=None, compact=True):
    """
    Get dV / dparam.
    
    Args:
        compact: if true dv is the tril part, dv_tril / dparam
    
    Returns:
        dv_dparam.
    """
    spin, nkpts, nlo, nbasis = basis.shape
    if compact:
        nbasis_pair = nbasis * (nbasis + 1) // 2
        tril_idx = np.tril_indices(nbasis)
        dV_dparam = np.empty((vcor.length(), spin, nbasis_pair))
    else:
        dV_dparam = np.empty((vcor.length(), spin, nbasis, nbasis))
    
    if vcor.is_local():
        if P_act is None: 
            for s in range(spin):
                for ip in range(vcor.length()):
                    if compact:
                        dV_dparam[ip, s] = transform_local_sparseH(basis[s], \
                                lattice, vcor.gradient()[ip, s])[tril_idx]
                    else:
                        dV_dparam[ip, s] = transform_local_sparseH(basis[s], \
                                lattice, vcor.gradient()[ip, s])
        else:
            from libdmet.basis_transform import make_basis
            C_lo_eo = make_basis.multiply_basis(P_act, basis_k) 
            grad_k = np.empty((nkpts, nlo, nlo), dtype=np.complex128)
            for s in range(spin):
                for ip in range(vcor.length()):
                    grad_k[:] = vcor.gradient()[ip, s]
                    if compact:
                        dV_dparam[ip, s] = transform_trans_inv_k(C_lo_eo[s], \
                                grad_k)[tril_idx]
                    else:
                        dV_dparam[ip, s] = transform_trans_inv_k(C_lo_eo[s], \
                                grad_k)
    else:
        if vcor.grad_k is None:
            vcor.gradient()
        for s in range(spin):
            for ip in range(vcor.length()):
                if compact:
                    dV_dparam[ip, s] = transform_trans_inv_k(basis_k[s], \
                            vcor.grad_k[ip, s])[tril_idx]
                else:
                    dV_dparam[ip, s] = transform_trans_inv_k(basis_k[s], \
                            vcor.grad_k[ip, s])
    
    # release the grad memory.
    vcor.grad   = None
    vcor.grad_k = None
    return dV_dparam

def FitVcorEmb(rho, lattice, basis, vcor, beta, MaxIter=300, 
               imp_fit=False, imp_idx=None, det=False, det_idx=None, 
               CG_check=False, BFGS=False, diff_criterion=None, **kwargs):
    """
    Fitting the correlation potential in the embedding space.

    Args:
        rho: target rdm1.
        lattice: lattice object.
        basis: embedding basis, C_lo_eo in R space.
        vcor: vcor object.
        beta: inverse temperature.
        MaxIter: max cycle.
        imp_fit: only fit the impurity part of rho.
        imp_idx: manually specify the impurity indices to fit.
        det: only fit the diagonal part of rho.
        det_idx: manually specify the diagonal indices to fit.
        CG_check: use scipy CG   to check after convergence.
        BFGS:     use scipy BFGS to check after convergence.
        diff_criterion: criterion for accept the scipy's result.
    
    Kwargs:
        ytol: tolerance for dy.
        gtol: tolerance for norm of gradient.
        xtol: tolerance for dx.
        mu0       [None]  mu0 for chemical potential.
        fix_mu    [False] fix chemical potential.
        num_grad  [False] use numerical gradient.
        remove_diag_grad  [False] remove the diagonal component of vcor.
        test_grad [False] test gradient and compared with numerical.
        idem_fit  [False] use idempotent part of rho to fit.
        P_act     [None]  add active space projector for veff.
        C_act     [None]  add active space projector for rho.
        return_drho_dparam [False] return drho_dparam.
        vcor_mat [None] if provide, will add to the embH1.

    Returns:
        vcor: vcor after optimization.
        err_begin: cost function value before optimization.
        err_end:   cost function value after optimization.
    """
    param_begin = vcor.param.copy()
    spin = basis.shape[0]
    nbasis = basis.shape[-1]
    nbasis_pair = nbasis * (nbasis + 1) // 2
    basis_k = lattice.R2k_basis(basis)
    nscsites = lattice.nscsites
    if spin == 1:
        nelec = lattice.ncore + lattice.nval
    else:
        nelec = [lattice.ncore + lattice.nval, lattice.ncore + lattice.nval]
    mu0                = kwargs.get("mu0", None)
    fix_mu             = kwargs.get("fix_mu", False)
    num_grad           = kwargs.get("num_grad", False)
    remove_diag_grad   = kwargs.get("remove_diag_grad", False)
    idem_fit           = kwargs.get("idem_fit", False)
    P_act              = kwargs.get("P_act", None)
    C_act              = kwargs.get("C_act", None)
    use_drho_dparam    = kwargs.get("use_drho_dparam", False)
    return_drho_dparam = kwargs.get("return_drho_dparam", False)

    # idem_fit
    if idem_fit:
        log.info("idempotent fitting? %s", idem_fit)
        rho = get_rdm1_idem(rho, nelec, beta)

    if lattice.use_hcore_as_emb_ham:
        fock_k = lattice.getH1(kspace=True)
    else:
        fock_k = lattice.getFock(kspace=True)
    fock_k = np.array(fock_k, copy=True)
    if fock_k.ndim == 3:
        fock_k = fock_k[np.newaxis]
    ovlp_k = lattice.get_ovlp(kspace=True)

    # initialize imp_idx and det_idx:
    if imp_fit:
        imp_idx = list(range(lattice.nimp))
        det_idx = []
    elif det:
        imp_idx = []
        det_idx = list(range(lattice.nimp))
    elif imp_idx is None:
        if det_idx is None: # imp + bath fitting
            imp_idx = list(range(nbasis))
            det_idx = []
        else:
            imp_idx = []
    elif det_idx is None:
        det_idx = []
    imp_idx = list(imp_idx)
    det_idx = list(det_idx)

    log.info("impurity fitting? %s", imp_fit)
    log.debug(1, "imp_idx: %s", format_idx(imp_idx))
    log.info("det (diagonal fitting)? %s", det)
    log.debug(1, "det_idx: %s", format_idx(det_idx))
    
    fit_idx = imp_idx + det_idx
    nimp, nidx = len(imp_idx), len(fit_idx)
    imp_mesh = np.ix_(imp_idx, imp_idx)
    det_mesh = (det_idx, det_idx)
    imp_fill_mesh = (slice(nimp), slice(nimp))
    det_fill_mesh = (range(nimp, nidx), range(nimp, nidx))
    if len(np.unique(fit_idx)) != nidx:
        log.warn("fit_idx has repeated indices: %s", format_idx(fit_idx))
    
    if P_act is not None:
        log.info("active space fitting? True.")
        P_act = get_active_projector_full(P_act, lattice.ovlp_lo_k)

    # pre-allocate the objects for efficiency:
    rho_target = np.zeros((spin, nidx, nidx))
    rho1 = np.zeros_like(rho_target)
    ew = np.empty((spin, nbasis))
    ev = np.empty((spin, nbasis, nbasis))

    # fock in the embedding space
    vcor_mat = kwargs.get("vcor_mat", None)
    if vcor_mat is not None:
        for s in range(spin):
            fock_k[s] += vcor_mat[s]
    embH1 = transform_h1(fock_k, basis_k)

    ovlp_emb = transform_h1(ovlp_k, basis_k)
    # dV / dparam
    dV_dparam = get_dV_dparam(vcor, basis, basis_k, lattice, P_act=P_act, \
            compact=True)
    diag_idx = (np.arange(nbasis), np.arange(nbasis))
    tril_idx = np.tril_indices(nbasis)
    
    # rho_target
    for s in range(spin):
        rho_target[s][imp_fill_mesh] = rho[s][imp_mesh]
        rho_target[s][det_fill_mesh] = rho[s][det_mesh]
    
    def Vemb_param(param):
        """
        Give param return corresponding Vemb.
        """
        if dV_dparam.ndim == 3:
            tmp = np.tensordot(param, dV_dparam, axes=(0, 0))
            v_emb = np.zeros((spin, nbasis, nbasis))
            for s in range(spin):
                # only the lower part is needed
                v_emb[s][tril_idx] = tmp[s] 
        else:
            v_emb = np.tensordot(param, dV_dparam, axes=(0, 0))
        return v_emb
    
    def errfunc(param):
        embHeff = embH1 + Vemb_param(param)
        for s in range(spin):
            ew[s], ev[s] = la.eigh(embHeff[s], ovlp_emb[s])
        if not fix_mu:
            if spin == 1:
                ew_sorted = np.sort(ew, axis=None, kind='mergesort')
                mu = 0.5 * (ew_sorted[nelec-1] + ew_sorted[nelec])
            else:
                ew_sorted = [np.sort(ew[s], axis=None, kind='mergesort') \
                             for s in range(spin)]
                mu = [0.5 * (ew_sorted[0][nelec[0]-1] + ew_sorted[0][nelec[0]]), \
                      0.5 * (ew_sorted[1][nelec[1]-1] + ew_sorted[1][nelec[1]])]
        else:
            mu = mu0
        ewocc, mu, _ = mfd.assignocc(ew, nelec, beta, mu, fix_mu=fix_mu)
        for s in range(spin):
            tmp = np.dot(ev[s]*ewocc[s], ev[s].T)
            rho1[s][imp_fill_mesh] = tmp[imp_mesh]
            rho1[s][det_fill_mesh] = tmp[det_mesh]
        
        if C_act is None:
            drho = rho1 - rho_target
        else:
            drho = np.empty((spin, C_act.shape[-1], C_act.shape[-1]))
            for s in range(spin):
                drho[s] = mdot(C_act[s].T, (rho1[s] - rho_target[s]), C_act[s])
            
        return la.norm(drho) / sqrt(spin)
    
    def gradfunc(param):
        embHeff = embH1 + Vemb_param(param)
        for s in range(spin):
            ew[s], ev[s] = la.eigh(embHeff[s], ovlp_emb[s])
        if not fix_mu:
            if spin == 1:
                ew_sorted = np.sort(ew, axis=None, kind='mergesort')
                mu = 0.5 * (ew_sorted[nelec-1] + ew_sorted[nelec])
            else:
                ew_sorted = [np.sort(ew[s], axis=None, kind='mergesort') \
                             for s in range(spin)]
                mu = [0.5 * (ew_sorted[0][nelec[0]-1] + ew_sorted[0][nelec[0]]), \
                      0.5 * (ew_sorted[1][nelec[1]-1] + ew_sorted[1][nelec[1]])]
        else:
            mu = mu0
        ewocc, mu, _ = mfd.assignocc(ew, nelec, beta, mu, fix_mu=fix_mu)
         
        for s in range(spin):
            tmp = np.dot(ev[s]*ewocc[s], ev[s].T)
            rho1[s][imp_fill_mesh] = tmp[imp_mesh]
            rho1[s][det_fill_mesh] = tmp[det_mesh]
        
        if C_act is None:
            drho = rho1 - rho_target
            val = la.norm(drho)
        else:
            drho = np.empty((spin, C_act.shape[-1], C_act.shape[-1]))
            for s in range(spin):
                drho[s] = mdot(C_act[s].T, (rho1[s] - rho_target[s]), C_act[s])
            val = la.norm(drho)
            
            # drho need C^d C for gradient:
            drho_grad = np.empty_like(rho1)
            for s in range(spin):
                drho_grad[s] = mdot(C_act[s], drho[s], C_act[s].T)
            drho = drho_grad

        occ = int(np.round(np.sum(ewocc) / spin))
        ewocc, ewvirt = ew[:, :occ], ew[:, occ:]
        evocc, evvirt = ev[:, :, :occ], ev[:, :, occ:]
        
        if dV_dparam.ndim == 3:
            dw_dV = np.empty((spin, nbasis_pair))
            for s in range(spin):
                e_mn = 1.0 / (-ewvirt[s].reshape((-1,1)) + ewocc[s])
                temp_mn = mdot(evvirt[s, fit_idx].T, drho[s], \
                        evocc[s, fit_idx]) * e_mn / (val * sqrt(spin))
                dw_dV_full  = mdot(evvirt[s], temp_mn, evocc[s].T)
                dw_dV_full  = dw_dV_full + dw_dV_full.T
                dw_dV_full *= 2.0
                dw_dV_full[diag_idx] *= 0.5
                dw_dV[s] = dw_dV_full[tril_idx]
            res = np.tensordot(dV_dparam, dw_dV, axes=((1, 2), (0, 1)))
        else:
            dw_dV = np.empty((spin, nbasis, nbasis))
            for s in range(spin):
                e_mn = 1.0 / (-ewvirt[s].reshape((-1,1)) + ewocc[s])
                temp_mn = mdot(evvirt[s, fit_idx].T, drho[s], \
                        evocc[s, fit_idx]) * e_mn / (val * sqrt(spin))
                dw_dV[s] = mdot(evvirt[s], temp_mn, evocc[s].T)
                dw_dV[s] += dw_dV[s].T
            res = np.tensordot(dV_dparam, dw_dV, axes=((1, 2, 3), (0, 1, 2)))
        
        # project out the diagonal component
        if remove_diag_grad:
            for s in range(spin):
                res[vcor.diag_indices()[s]] -= \
                        np.average(res[vcor.diag_indices()[s]])
        return res

    def gradfunc_ft(param):
        embHeff = embH1 + Vemb_param(param)
        for s in range(spin):
            ew[s], ev[s] = la.eigh(embHeff[s], ovlp_emb[s])
        if not fix_mu:
            if spin == 1:
                ew_sorted = np.sort(ew, axis=None, kind='mergesort')
                mu = 0.5 * (ew_sorted[nelec-1] + ew_sorted[nelec])
            else:
                ew_sorted = [np.sort(ew[s], axis=None, kind='mergesort') \
                             for s in range(spin)]
                mu = [0.5 * (ew_sorted[0][nelec[0]-1] + ew_sorted[0][nelec[0]]), \
                      0.5 * (ew_sorted[1][nelec[1]-1] + ew_sorted[1][nelec[1]])]
        else:
            mu = mu0
        ewocc, mu, _ = mfd.assignocc(ew, nelec, beta, mu, fix_mu=fix_mu)
        for s in range(spin):
            tmp = np.dot(ev[s]*ewocc[s], ev[s].T)
            rho1[s][imp_fill_mesh] = tmp[imp_mesh]
            rho1[s][det_fill_mesh] = tmp[det_mesh]
        
        if C_act is None:
            drho = rho1 - rho_target
            val = la.norm(drho)
        else:
            drho = np.empty((spin, C_act.shape[-1], C_act.shape[-1]))
            for s in range(spin):
                drho[s] = mdot(C_act[s].T, (rho1[s] - rho_target[s]), C_act[s])
            val = la.norm(drho)
            
            # drho need C^d C for gradient:
            drho_grad = np.empty_like(rho1)
            for s in range(spin):
                drho_grad[s] = mdot(C_act[s], drho[s], C_act[s].T)
            drho = drho_grad

        dw_dv = ftsystem.get_dw_dv(ew, ev, drho, mu, beta, \
                fix_mu=fix_mu, fit_idx=fit_idx, compact=(dV_dparam.ndim == 3))
        dw_dparam = dV_dparam.reshape(dV_dparam.shape[0], -1).dot(dw_dv.ravel()) \
                / (2.0 * val * sqrt(spin))
        
        # project out the diagonal component
        if remove_diag_grad:
            for s in range(spin):
                dw_dparam[vcor.diag_indices()[s]] -= \
                        np.average(dw_dparam[vcor.diag_indices()[s]])
        return dw_dparam

    err_begin = errfunc(vcor.param)
    if beta == np.inf:
        log.info("Using analytic gradient for 0 T")
    else:
        log.info("Using analytic gradient for finite T, beta = %s", beta)
        gradfunc = gradfunc_ft

    if kwargs.get("test_grad", False):
        """
        This block is for testing gradient.
        """
        np.random.seed(10086)
        param_rand = np.random.random(vcor.param.shape)
        test_grad(param_rand.copy(), errfunc, gradfunc, dx=1e-4)
        test_grad(param_rand.copy(), errfunc, gradfunc, dx=1e-5)
    
    if use_drho_dparam or return_drho_dparam:
        """
        This block is for testing drho_dparam.
        """
        assert beta < np.inf
        embHeff = embH1 + Vemb_param(vcor.param)
        for s in range(spin):
            ew[s], ev[s] = la.eigh(embHeff[s], ovlp_emb[s])
        if not fix_mu:
            if spin == 1:
                ew_sorted = np.sort(ew, axis=None, kind='mergesort')
                mu = 0.5 * (ew_sorted[nelec-1] + ew_sorted[nelec])
            else:
                ew_sorted = [np.sort(ew[s], axis=None, kind='mergesort') \
                             for s in range(spin)]
                mu = [0.5 * (ew_sorted[0][nelec[0]-1] + ew_sorted[0][nelec[0]]), \
                      0.5 * (ew_sorted[1][nelec[1]-1] + ew_sorted[1][nelec[1]])]
        else:
            mu = mu0
        ewocc, mu, _ = mfd.assignocc(ew, nelec, beta, mu, fix_mu=fix_mu)
        if not isinstance(mu, Iterable):
            mu = [mu]
        
        log.info("compute drho_dv")
        # dV_dparam (nparam, nv)
        # drho_dv (nv, nrho)
        drho_dv = np.empty((spin, dV_dparam.shape[-1], nbasis_pair))
        for s in range(spin):
            drho_dv[s] = ftsystem.get_rho_grad(ew[s], ev[s], mu[s], beta, \
                    fix_mu=fix_mu, compact=True)
        drho_dv = np.asarray(drho_dv)
        
        log.info("compute drho_dparam")
        drho_dparam = np.einsum('psV, sVr -> spr', dV_dparam, drho_dv, \
                optimize=True)

        log.info("compute drho_dparam norm")
        log.info("norm: %s", la.norm(drho_dparam, axis=1))
        if return_drho_dparam:
            return drho_dparam
        else:
            drho_dv = None
        
        #us, sigmas, vts = [], [], []
        #tol = 1e-8
        #nnonzero = 0
        #for s in range(spin):
        #    u, sigma, vt = la.svd(drho_dparam[s], full_matrices=False)
        #    us.append(u)
        #    sigmas.append(sigma)
        #    vts.append(vt)
        #    nnonzero = max(np.sum(sigma > tol), nnonzero)
        #C_trunc = np.asarray(vts)[:, :nnonzero].transpose(0, 2, 1)
        #
        #rho_ind = np.empty((spin, nbasis_pair))
        #for s in range(spin):
        #    rho_ind[s] = rho[s][np.tril_indices(nbasis)]
        #print (rho_ind)
        #print (np.einsum('sr, srx -> sx', rho_ind, C_trunc))

    if num_grad: 
        log.warn("You are using numerical gradient...")
        gradfunc = None

    param, err_end, pattern, gnorm_res = minimize(errfunc, vcor.param, MaxIter, gradfunc, **kwargs)
    vcor.update(param)
    
    log.info("Minimizer converge pattern: %d ", pattern)
    log.info("Current function value: %15.8f", err_end)
    log.info("Norm of gradients: %s", gnorm_res)
    log.info("Norm diff of x: %15.8f", max_abs(param - param_begin))
    
    # Check with scipy minimizer
    if CG_check and (pattern == 0 or gnorm_res > 1.0e-4):
        log.info("Check with optimizer in Scipy...")
        from scipy import optimize as opt
        param_new = param.copy()
        gtol = max(5.0e-5, gnorm_res*0.1)
        gtol = min(gtol, 1.0e-2)
        if BFGS:
            log.info("BFGS used in check")
            method = 'BFGS'
        else:
            log.info("CG used in check")
            method = 'CG'
        
        min_result = opt.minimize(errfunc, param_new, method=method, jac=gradfunc ,\
                options={'maxiter': min(len(param_new)*10, MaxIter), \
                'disp': True, 'gtol': gtol})
        param_new_2 = min_result.x
        log.info("CG Final Diff: %s", min_result.fun) 
        log.info("Converged: %s", min_result.status)
        log.info("Jacobian: %s", max_abs(min_result.jac))
        if(not min_result.success):
            log.warn("Minimization unsuccessful. Message:\n%s", min_result.message)
    
        gnorm_new = max_abs(min_result.jac)
        diff_CG_old = max_abs(param_new_2 - param_new)
        log.info("max diff in x between %s and old: %s", method, diff_CG_old)
        if diff_criterion is None:
            if pattern == 0:
                diff_criterion = 2.0
            else:
                diff_criterion = 1.0
        if (gnorm_new < gnorm_res * 0.9) and (min_result.fun < err_end) \
                and (diff_CG_old < diff_criterion):
            log.info("New result used")
            vcor.update(param_new_2)
            err_end = min_result.fun
        else:
            log.info("Old result used")
            vcor.update(param_new)
    else:
        log.info("Old result used")
    
    if log.Level[log.verbose] > log.Level["DEBUG0"]:
        errfunc(vcor.param)
        log.debug(1, "rdm1 target:\n%s", rho_target)
        log.debug(1, "rdm1 fitted:\n%s", rho1)
    return vcor, err_begin, err_end

def FitVcorFull(rho, lattice, basis, vcor, beta, filling, MaxIter=20, imp_fit=False, \
        CG_check=False, BFGS=False, diff_criterion=None, scf=False, **kwargs):
    param_begin = vcor.param.copy()
    spin = basis.shape[0]
    nbasis = basis.shape[-1]
    nscsites = lattice.nscsites
    basis_k = lattice.R2k_basis(basis)
    rho1 = np.empty((spin, nbasis, nbasis))

    if imp_fit:
        fit_mesh_spin = np.ix_(range(spin), range(lattice.nimp), range(lattice.nimp))
    else:
        fit_mesh_spin = np.ix_(range(spin), range(nbasis), range(nbasis))
    
    def errfunc(param):
        vcor.update(param)
        verbose = log.verbose
        log.verbose = "RESULT"
        res = mfd.HF(lattice, vcor, filling, spin==1, mu0=None, \
                beta=beta, scf=scf, ires=True)[-1]
        log.verbose = verbose
        rho1 = foldRho_k(res["rho_k"], basis_k)
        return la.norm((rho - rho1)[fit_mesh_spin]) / sqrt(spin)
    
    err_begin = errfunc(vcor.param)
    param, err_end, pattern, gnorm_res = minimize(errfunc, vcor.param, MaxIter, **kwargs)
    vcor.update(param)
    
    log.info("Minimizer converge pattern: %d ", pattern)
    log.info("Current function value: %15.8f", err_end)
    log.info("Norm of gradients: %15.8f", gnorm_res)
    log.info("Norm diff of x: %15.8f", max_abs(param- param_begin))
    
    if CG_check and (pattern == 0 or gnorm_res > 1.0e-4):
        log.info("Check with optimizer in Scipy...")
        from scipy import optimize as opt
        param_new = param.copy()
        gtol = max(5.0e-5, gnorm_res*0.1)
        gtol = min(gtol, 1.0e-2)
        gradfunc = None

        if BFGS:
            log.info("BFGS used in check")
            method = 'BFGS'
        else:
            log.info("CG used in check")
            method = 'CG'
        min_result = opt.minimize(errfunc, param_new, method = method, jac = gradfunc ,\
                options={'maxiter': min(len(param_new)*100, MaxIter), 'disp': True, 'gtol': gtol})
        param_new_2 = min_result.x
    
        log.info("CG Final Diff: %s", min_result.fun) 
        log.info("Converged: %s", min_result.status)
        log.info("Jacobian: %s", max_abs(min_result.jac))
        if(not min_result.success):
            log.warn("Minimization unsuccessful. Message: %s", min_result.message)
    
        gnorm_new = max_abs(min_result.jac)
        diff_CG_old = max_abs(param_new_2 - param_new)
        log.info("max diff in x between New and Old: %s", diff_CG_old)
        if diff_criterion is None:
            if pattern == 0:
                diff_criterion = 2.0
            else:
                diff_criterion = 1.0
        if (min_result.fun < err_end) \
                and diff_CG_old < diff_criterion:
            log.info("New result used")
            vcor.update(param_new_2)
            err_end = min_result.fun
        else:
            log.info("Old result used")
            vcor.update(param_new)
    else:
        log.info("Old result used")

    return vcor, err_begin, err_end

def FitVcorTwoStep(rho, lattice, basis, vcor, beta, filling, \
        MaxIter1=300, MaxIter2=0, **kwargs):
    """
    Main wrapper for correlation potential fitting.
    """
    vcor_new = copy.deepcopy(vcor)
    log.result("Using two-step vcor fitting")
    if MaxIter1 > 0:
        log.info("Impurity model stage  max %d steps", MaxIter1)
        
        if kwargs.get("return_drho_dparam", False):
            return FitVcorEmb(rho, lattice, basis, vcor_new, beta, \
                    MaxIter=MaxIter1, **kwargs)
        
        vcor_new, err_begin, err_end = FitVcorEmb(rho, lattice, basis, vcor_new, beta, \
            MaxIter=MaxIter1, **kwargs)
        log.result("residue (begin) = %20.12f", err_begin)
        log.info("residue (end)   = %20.12f", err_end)
    if MaxIter2 > 0:
        log.info("Full lattice stage  max %d steps", MaxIter2)
        vcor_new, _, err_end = FitVcorFull(rho, lattice, basis, vcor_new, beta, \
            filling, MaxIter = MaxIter2, **kwargs)
    log.result("residue (end)   = %20.12f", err_end)
    return vcor_new, err_end

def get_H1_scaled(H1, imp_idx, env_idx=None):
    """
    Scale H1 by the number of imp indices.
    """
    assert H1.ndim == 3
    nbasis = H1.shape[-1]
    if env_idx is None:
        env_idx = np.asarray([idx for idx in range(nbasis) \
                              if idx not in imp_idx], dtype=int) 
    imp_env = np.ix_(imp_idx, env_idx)
    env_imp = np.ix_(env_idx, imp_idx)
    env_env = np.ix_(env_idx, env_idx)
    for s in range(H1.shape[0]):
        H1[s][imp_env] *= 0.5
        H1[s][env_imp] *= 0.5
        H1[s][env_env]  = 0.0
    return H1

def get_H2_scaled(H2, imp_idx, env_idx=None):
    """
    Scale H2 by the number of imp indices, 1-fold or 4-fold symmetry.
    """
    if H2.ndim == 3:
        nbasis_pair = H2.shape[-1]
        nbasis = int(np.sqrt(nbasis_pair * 2))
        if env_idx is None:
            env_idx = np.asarray([idx for idx in range(nbasis) \
                                  if idx not in imp_idx], dtype=int) 
        
        tril_idx = np.tril_indices(nbasis)
        mask = np.isin(tril_idx, imp_idx)
        zero = np.logical_not(np.logical_or(*mask))
        half = np.logical_xor(*mask)
        one  = np.logical_and(*mask)
        mask_list = (zero, half, one)
        for s in range(H2.shape[0]):
            for i, mask_i in enumerate(mask_list):
                for j, mask_j in enumerate(mask_list):
                    if i + j == 4:
                        continue
                    elif i + j == 0:
                        mesh = np.ix_(mask_i, mask_j)
                        H2[s][mesh] = 0.0
                    else:
                        mesh = np.ix_(mask_i, mask_j)
                        H2[s][mesh] *= ((i + j) * 0.25)
    elif H2.ndim == 5:
        nbasis = H2.shape[-1]
        if env_idx is None:
            env_idx = np.asarray([idx for idx in range(nbasis) \
                                  if idx not in imp_idx], dtype=int) 
        mask_list = (env_idx, imp_idx)
        for s in range(H2.shape[0]):
            for i, mi in enumerate(mask_list):
                for j, mj in enumerate(mask_list):
                    for k, mk in enumerate(mask_list):
                        for l, ml in enumerate(mask_list):
                            mesh = np.ix_(mi, mj, mk, ml)
                            factor = (i + j + k + l) * 0.25
                            H2[s][mesh] *= factor
    else:
        raise ValueError("Unknown H2 shape to scale: %s" % (str(H2.shape)))
    return H2

def transformResults(rhoEmb, E, basis, ImpHam, H1e=None, **kwargs):
    """
    Transform results.
    Get density matrix, dmet energy (of non-interacting bath) 
    and electron number.
    """
    spin = rhoEmb.shape[0]
    nscsites = basis.shape[2]
    nbasis = basis.shape[-1]
    
    # ZHC NOTE here I assume we use the local basis,
    # i.e. the first nimp indices are impurity, 
    # the remaining nbasis-nimp is bath.
    # this is useful for case that impurity is a subset or 
    # even beyond the first cell.
    if "lattice" in kwargs:
        imp_idx = np.asarray(kwargs.get("imp_idx", range(kwargs["lattice"].nimp)))
    else:
        imp_idx = np.asarray(kwargs.get("imp_idx", np.arange(nscsites)))
    if any(imp_idx >= nscsites):
        log.warn("imp_idx is out of the first cell... imp_idx:\n%s", imp_idx)
    nelec = 0.0
    for s in range(spin):
        nelec += np.sum(rhoEmb[s, imp_idx, imp_idx])
    nelec *= (2.0 / spin)
    rhoImp = rhoEmb[np.ix_(range(spin), imp_idx, imp_idx)]

    if E is not None:
        lattice = kwargs["lattice"]
        last_dmu = kwargs["last_dmu"]
        imp_idx = np.asarray(kwargs.get("imp_idx", list(range(lattice.nimp))))
        dmu_idx = kwargs.get("dmu_idx", None)
        if dmu_idx is None:
            dmu_idx = list(range(nscsites))
        env_idx = np.asarray([idx for idx in range(nbasis) 
                              if idx not in imp_idx], dtype=int) 

        E2 = E - np.sum(ImpHam.H1["cd"] * rhoEmb) * (2.0 / spin) - ImpHam.H0
        
        H1_scaled = np.array(ImpHam.H1["cd"], copy=True)
        dmu_mat = np.zeros((nscsites, nscsites))
        dmu_mat[dmu_idx, dmu_idx] = (-last_dmu)
        for s in range(spin):
            # remove the contribution of last_dmu
            H1_scaled[s] -= transform_imp(basis[s], lattice, dmu_mat)
            
            # remove the JK_core if possible
            if lattice.JK_core is not None:
                H1_scaled[s] -= 0.5 * lattice.JK_core[s]
        H1_scaled = get_H1_scaled(H1_scaled, imp_idx, env_idx) 

        E1 = np.sum(H1_scaled * rhoEmb) * (2.0 / spin)
        Efrag = E1 + E2 + lattice.getH0()
        log.debug(0, "E0 = %20.12f, E1 = %20.12f, " 
                "E2 = %20.12f, E = %20.12f", lattice.getH0(), E1, E2, Efrag)
    else:
        Efrag = None
    return rhoImp, Efrag, nelec

def get_veff_from_rdm1_emb(lattice, rdm1_emb, basis, kmf=None, C_ao_lo=None, 
                           return_update=False, sign=None):
    """
    First construct rdm1_glob and then compute the veff in LO basis.
    rdm1_emb is per spin.
    rdm1_glob is summed over spin if restricted.
    """
    if not isinstance(lattice, Iterable):
        lattice = [lattice]
        rdm1_emb = [rdm1_emb]
        basis = [basis]
    if kmf is None:
        kmf = lattice[0].kmf
    if C_ao_lo is None:
        C_ao_lo = lattice[0].C_ao_lo
    spin = basis[0].shape[-4]
    
    rdm1_glob = get_rho_glob_k(basis, lattice, rdm1_emb, sign=sign) * (2.0 / spin)
    np.save("rdm1_glob_lo_k.npy", rdm1_glob)
    rdm1_veff = make_basis.transform_rdm1_to_ao(rdm1_glob, C_ao_lo)
    #veff_ao = kmf.get_veff(dm_kpts=rdm1_veff)
    if getattr(lattice, "is_mol", None) and lattice.is_mol: 
        raise NotImplementedError
    else:
        vj, vk = kmf.get_jk(dm_kpts=rdm1_veff)
    if spin == 1:
        veff_ao = vj - (vk * 0.5)
    else:
        veff_ao = vj[0] + vj[1] - vk
    veff = make_basis.transform_h1_to_lo(veff_ao, C_ao_lo)
    if return_update:
        return veff, veff_ao, lattice[0].k2R(rdm1_glob)
    else:
        return veff

def get_veff_from_rdm1_emb_mpi(Lat, rdm1_emb, basis, comm, return_update=False):
    """
    Get veff from embedding rdm1 in LO basis, MPI version.
    Each process treat 1 impurity problem.
    """
    nfrag = comm.Get_size()
    rank  = comm.Get_rank()
    comm.Barrier()
    val_idx_col  = comm.gather(Lat.val_idx, root=0)
    virt_idx_col = comm.gather(Lat.virt_idx, root=0)
    core_idx_col = comm.gather(Lat.core_idx, root=0)
    basis_col    = comm.gather(basis, root=0)
    rdm1_emb_col = comm.gather(rdm1_emb, root=0)
    if return_update:
        if rank == 0:
            Lat_col = [copy.copy(Lat) for i in range(nfrag)]
            with lib.temporary_env(log, verbose="FATAL"):
                for i in range(nfrag):
                    Lat_col[i].set_val_virt_core(val_idx_col[i], virt_idx_col[i], core_idx_col[i])
            veff, veff_ao, rdm1_glob_R = get_veff_from_rdm1_emb(Lat_col, rdm1_emb_col, basis_col, return_update=True)
            rdm1_glob_R = np.asarray(rdm1_glob_R.real, order='C')
        else:
            spin, nkpts, nlo, _ = basis.shape
            nao = Lat.C_ao_lo.shape[-2]
            veff = np.zeros((spin, nkpts, nlo, nlo), dtype=np.complex128)
            veff_ao = np.zeros((spin, nkpts, nao, nao), dtype=np.complex128)
            rdm1_glob_R = np.zeros((spin, nkpts, nlo, nlo))
        comm.Barrier()
        comm.Bcast(veff, root=0)
        comm.Bcast(veff_ao, root=0)
        comm.Bcast(rdm1_glob_R, root=0)
        return veff, veff_ao, rdm1_glob_R
    else:
        if rank == 0:
            Lat_col = [copy.copy(Lat) for i in range(nfrag)]
            with lib.temporary_env(log, verbose="FATAL"):
                for i in range(nfrag):
                    Lat_col[i].set_val_virt_core(val_idx_col[i], virt_idx_col[i], core_idx_col[i])
                veff = get_veff_from_rdm1_emb(Lat_col, rdm1_emb_col, basis_col)
        else:
            spin, nkpts, nlo, _ = basis.shape
            veff = np.zeros((spin, nkpts, nlo, nlo), dtype=np.complex128)
        comm.Barrier()
        comm.Bcast(veff, root=0)
        return veff

def get_H_dmet(basis, lattice, ImpHam, last_dmu, imp_idx=None, dmu_idx=None, 
               add_vcor_to_E=False, vcor=None, compact=True, rdm1_emb=None, 
               veff=None, rebuild_veff=False, E1=None):
    """
    Get a DMET hamiltonian, which is scaled by number of impurity indices, 
    and can be directly used for evaluation of DMET energy.
    The impurity index can be specified by imp_idx
    This function does not change the aabb order for unrestricted Ham.
    
    rdm1_emb: embedding rdm1 (per spin).
    veff: if provide, use veff as JK_core.
    rebuild_veff: re-evaluate the JK_core from the global density matrix.
    E1: if provide, will use it as E1 (including hcore and J, K).
    """
    log.debug(0, "Construct Heff for DMET.")
    spin = basis.shape[0]
    nbasis = basis.shape[-1]

    if imp_idx is None:
        imp_idx = list(range(lattice.nimp))
    imp_idx = np.asarray(imp_idx)
    env_idx = np.asarray([idx for idx in range(nbasis) \
                          if idx not in imp_idx], dtype=int) 
    log.debug(1, "imp_idx: %s", format_idx(imp_idx))
    log.debug(1, "env_idx: %s", format_idx(env_idx))
    basis_k = lattice.R2k_basis(basis) 
    
    if E1 is None:
        hcore_k = lattice.getH1(kspace=True)
        H1_scaled = transform_h1(hcore_k, basis_k)
        
        # note the double counting from JK_core cf. HF energy
        if (veff is not None) or rebuild_veff:
            if veff is None:
                veff = get_veff_from_rdm1_emb(lattice, rdm1_emb, basis)
            JK_core  = transform_h1(veff, basis_k)
            veff_loc = get_veff(rdm1_emb * (2.0 / spin), ImpHam.H2["ccdd"], hyb=1.0)
            JK_core -= veff_loc
            if lattice.JK_core is not None:
                log.debug(1, "difference between JK_glob and JK_HF: %15.8g", 
                          max_abs(JK_core - lattice.JK_core))
        else:
            if lattice.JK_core is not None:
                JK_core = lattice.JK_core
            else:
                JK_core = [0.0 for s in range(spin)]

        for s in range(spin): 
            H1_scaled[s] += 0.5 * JK_core[s]
            if add_vcor_to_E:
                log.debug(0, "Add Vcor to energy expression")
                H1_scaled[s] += transform_local(basis[s], lattice, vcor.get()[s] * 0.5)
                H1_scaled[s] -= transform_imp(basis[s], lattice, vcor.get()[s] * 0.5)
        H1_scaled = get_H1_scaled(H1_scaled, imp_idx, env_idx)
        
        # H0 part
        H0 = lattice.getH0()
    else:
        # manually specify E1 (contribution from hcore and J, K)
        H1_scaled = (-1.0 / spin) * get_veff(rdm1_emb, ImpHam.H2["ccdd"], hyb=1.0)
        H1_scaled = get_H1_scaled(H1_scaled, imp_idx, env_idx)
        H0 = (E1 + lattice.getH0()).real

    # H2 part
    # restore 4-fold symmetry
    H2_scaled = np.empty((spin*(spin+1)//2, nbasis*(nbasis+1)//2, nbasis*(nbasis+1)//2))
    for s in range(spin*(spin+1)//2):
        H2_scaled[s] = ao2mo.restore(4, ImpHam.H2["ccdd"][s], nbasis)
    H2_scaled = get_H2_scaled(H2_scaled, imp_idx, env_idx)

    ImpHam_dmet = integral.Integral(nbasis, spin == 1, False, H0, 
                                    {"cd": H1_scaled}, {"ccdd": H2_scaled})
    if not compact:
        log.warn("Restoring 1-fold symmetry in dmet Hamiltonian...")
        ImpHam_dmet = restore_Ham(ImpHam_dmet, 1, in_place=True)
    return ImpHam_dmet

def get_E_dmet(basis, lattice, ImpHam, last_dmu, solver, solver_args={}, **kwargs):
    ImpHam_scaled = get_H_dmet(basis, lattice, ImpHam, last_dmu, **kwargs)
    E = solver.run_dmet_ham(ImpHam_scaled, **solver_args)
    return E

def get_E_dmet_HF(basis, lattice, ImpHam, last_dmu, solver, **kwargs):
    """
    Get a DMET energy for a given mean-field solver. 
    """
    log.debug(0, "Construct Heff for DMET (HF).")
    spin = basis.shape[0]
    nbasis = basis.shape[-1]
    nscsites = lattice.nscsites  
    nimp = lattice.nimp
    if hasattr(solver, "mf"):
        mf = solver.mf
    else:
        mf = solver.scfsolver.mf
    
    imp_idx = kwargs.get("imp_idx", list(range(lattice.nimp)))
    env_idx = np.asarray([idx for idx in range(nbasis) \
                          if idx not in imp_idx], dtype=int)
    log.debug(1, "imp_idx: %s", format_idx(imp_idx))
    log.debug(1, "env_idx: %s", format_idx(env_idx))

    dmu_mat = np.zeros((nscsites, nscsites))
    dmu_mat[imp_idx, imp_idx] = (-last_dmu)

    rdm1 = np.asarray(mf.make_rdm1())
    h1e  = np.asarray(mf.get_hcore())
    fock = np.asarray(mf.get_fock(h1e=h1e, dm=rdm1))
    if rdm1.ndim == 2:
        rdm1 = rdm1[np.newaxis]
        h1e = h1e[np.newaxis]
        fock = fock[np.newaxis]
    heff = (h1e + fock) * 0.5
    
    for s in range(spin):
        # remove the double counting from JK_core cf. HF energy
        if lattice.JK_core is not None:
            heff[s] -= 0.5 * lattice.JK_core[s]
        # remove mu
        heff[s] -= transform_imp(basis[s], lattice, dmu_mat)
    
    heff = get_H1_scaled(heff, imp_idx, env_idx)
    E = np.sum(heff * rdm1) + lattice.getH0()

    # compute JK_core's contribution for debug.
    JK_core_scaled = get_H1_scaled(lattice.JK_core * 0.5, imp_idx, env_idx)
    E_JK_core = np.sum(JK_core_scaled * rdm1)
    log.debug(0, "JK_core contribution in DMET HF energy: %20.10f", E_JK_core)
    return E

# ************************************************
# test and old functions
# ************************************************

def get_H1_dmet(lattice, basis, hcore_lo_k, fock_lo_k, JK_imp, imp_idx=None):
    """
    Get a DMET H1, which is scaled by number of impurity indices.
    This function only use hcore_lo_k, fock_lo_k, JK_imp as input.
    The impurity index can be specified by imp_idx.
    This function does not change the aabb order for unrestricted Ham.
    """
    log.debug(0, "Construct Heff_1 for DMET.")
    spin, nkpts, nlo, nbasis = basis.shape
    
    basis_k = lattice.R2k_basis(basis)
    hcore_emb = transform_h1(hcore_lo_k, basis_k)
    fock_emb = transform_h1(fock_lo_k, basis_k)
    JK_imp = add_spin_dim(JK_imp, spin, non_spin_dim=2)
    JK_emb = np.asarray([transform_imp(basis[s], lattice, JK_imp[s]) \
                         for s in range(spin)])
    JK_core = fock_emb - hcore_emb - JK_emb
    H1 = hcore_emb + JK_core * 0.5
    
    if imp_idx is None:
        imp_idx = list(range(lattice.nimp))
    imp_idx = np.asarray(imp_idx)
    env_idx = np.asarray([idx for idx in range(nbasis) \
                          if idx not in imp_idx], dtype=int) 
    log.debug(1, "imp_idx: %s", format_idx(imp_idx))
    log.debug(1, "env_idx: %s", format_idx(env_idx))
    
    H1_scaled = get_H1_scaled(H1, imp_idx, env_idx)
    return H1_scaled

def get_active_projector(act_idx, rdm1, ovlp, orth=True, tol=1e-9):
    """
    Get active projector.
    
    Args:
        act_idx: nact indices of selected LOs for active space.
        rdm1: rdm1, (spin, nkpts, nlo, nlo), 
              if restricted, it is actually spin traced.
        ovlp: (nkpts, nlo, nlo), overlap matrix.
        orth: whether apply Lowdin orthogonalization on the orbitals.
        tol: tolerance for droping singular values of overlap.

    Returns:
        P_act: active space projector, a list (spin, nkpts, nlo, nact),
               the last dimension may vary for different kpts.
        nocc: occupied orbitals at each spin, k, shape (spin, nkpts).
    """
    from libdmet.basis_transform import make_basis
    act_idx = np.asarray(act_idx)
    nact = len(act_idx)

    rdm1 = np.asarray(rdm1)
    if rdm1.ndim == 3:
        rdm1 = rdm1[None]
    spin, nkpts, nlo, _ = rdm1.shape
    if spin == 1:
        rdm1 = rdm1 * 0.5
    ovlp = np.asarray(ovlp)
    ovlp = add_spin_dim(ovlp, spin, non_spin_dim=3)
    # ZHC NOTE FIXME the convention of hole rdm1 need to be discussed.
    # here I use a convention consistent with supercell.
    # but the following way seems more intuitive from formula.
    #rdm1_h = ovlp - rdm1.transpose(0, 1, 3, 2)
    rdm1_h = ovlp - rdm1
    
    P_occ  = rdm1  [:, :, :, act_idx]
    P_virt = rdm1_h[:, :, :, act_idx]

    ovlp_occ  = make_basis.transform_h1_to_lo(ovlp, P_occ)
    ovlp_virt = make_basis.transform_h1_to_lo(ovlp, P_virt)

    P_non_singular = []
    nocc = np.zeros((spin, nkpts), dtype=int)
    for s in range(spin):
        P = []
        for k in range(nkpts):
            ew, ev = la.eigh(ovlp_occ[s][k])
            idx = (ew > tol)
            with np.printoptions(suppress=False):
                log.debug(0, "active projector: s %s k %s, ew_occ:\n%s", s, k, ew)
                log.debug(0, "active projector: select %d / %d", np.sum(idx), len(ew))
            X_occ = ev[:, idx]
            tmp_occ = np.dot(P_occ[s, k], X_occ)
            if orth:
                tmp_occ = vec_lowdin(tmp_occ, ovlp[s, k]) 
            nocc[s][k] = X_occ.shape[-1]
             
            ew, ev = la.eigh(ovlp_virt[s][k])
            idx = (ew > tol)
            with np.printoptions(suppress=False):
                log.debug(0, "active projector: s %s k %s, ew_virt:\n%s", s, k, ew)
                log.debug(0, "active projector: select %d / %d", np.sum(idx), len(ew))
            X_virt = ev[:, idx]
            tmp_virt = np.dot(P_virt[s, k], X_virt)
            if orth:
                tmp_virt = vec_lowdin(tmp_virt, ovlp[s, k])

            P.append(np.hstack((tmp_occ, tmp_virt)))
            log.debug(0, "active projector: P shape %s", P[-1].shape)

        P_non_singular.append(P)
    
    return P_non_singular, nocc

def get_active_projector_full(P_act, ovlp):
    """
    Get the projection operator from full LO to full LO.

    Args:
        P_act: a list (spin, nkpts, nlo, nact)
        ovlp: ndarray, ((spin,), nkpts, nlo, nlo)

    Returns:
        P_full: ndarray (spin, nkpts, nlo, nlo).
    """
    ovlp = np.asarray(ovlp)
    if ovlp.ndim == 3:
        ovlp = ovlp[None]
    spin, nkpts, nlo, _ = ovlp.shape
    assert len(P_act) == spin

    P_full = np.empty((spin, nkpts, nlo, nlo), dtype=ovlp.dtype)
    for s in range(spin):
        for k in range(nkpts):
            # ovlp_act = P.conj().T ovlp_lo P
            ovlp_act = mdot(P_act[s][k].conj().T, ovlp[s][k], P_act[s][k])
            # P_full = P ovlp_act P.conj().T
            P_full[s, k] = mdot(P_act[s][k], ovlp_act, P_act[s][k].conj().T)
    return P_full

def make_rdm1_P(fock_lo, ovlp_lo, vcor, P_act, nocc, project_back=True, \
        lattice=None, beta=np.inf):
    """
    Make rdm1 for active projected space.
    
    Args:

    Returns:
        rdm1_P: rdm1 after projection, from P(f+u)P.
    """
    fock_lo = np.asarray(fock_lo)
    spin, nkpts, nlo, _ = fock_lo.shape
    ovlp_lo = np.asarray(ovlp_lo)
    ovlp_lo = add_spin_dim(ovlp_lo, spin, non_spin_dim=3)
    vcor_lo = np.zeros_like(fock_lo)
    if vcor is not None:
        vcor_mat = vcor.get()
        for s in range(spin):
            vcor_lo[s] = vcor_mat[s]
    
    # diagonalize
    # ZHC TODO use lattice for time reversal symmetry.
    ew = []
    ev = []
    for s in range(spin):
        ew_s = []
        ev_s = []
        for k in range(nkpts):
            fock_P = mdot(P_act[s][k].conj().T, \
                    fock_lo[s, k] + vcor_lo[s, k], P_act[s][k])
            ovlp_P = mdot(P_act[s][k].conj().T, ovlp_lo[s, k], P_act[s][k])
            e, v = la.eigh(fock_P, ovlp_P)
            ew_s.append(e)
            ev_s.append(v)
        ew.append(ew_s) 
        ev.append(ev_s)

    # make rdm1
    if beta < np.inf:
        # ZHC TODO finite T case. 
        raise NotImplementedError
    else:
        rdm1_P = np.empty((spin, nkpts, nlo, nlo), dtype=np.complex128)
        for s in range(spin):
            for k in range(nkpts):
                gap = abs(ew[s][k][nocc[s, k]] - ew[s][k][nocc[s, k]-1])
                log.debug(2, "make_rdm1_P: gap %s", gap)
                if gap < 1e-6:
                    log.warn("make_rdm1_P: HOMO %s == LUMO %s", \
                            ew[s][k][nocc[s, k]-1], ew[s][k][nocc[s, k]])
                ev_occ = ev[s][k][:, :nocc[s, k]]
                rdm1 = np.dot(ev_occ, ev_occ.conj().T)
                if spin == 1:
                    rdm1 *= 2.0
                rdm1_P[s, k] = mdot(P_act[s][k], rdm1, P_act[s][k].conj().T)
    return rdm1_P

if __name__ == '__main__':
    np.set_printoptions(3, linewidth=1000)
