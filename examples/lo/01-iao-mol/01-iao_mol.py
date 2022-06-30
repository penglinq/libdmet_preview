#!/usr/bin/env python

'''
IAO orbitals
'''
import sys
sys.path.append('../../')

from functools import reduce
import numpy as np
import scipy.linalg as la
from pyscf import gto, scf, lo
from pyscf.tools import molden
from libdmet.lo import iao

def get_iao_virt(cell, C_ao_iao, S, minao='minao'): 
    """
    Get virtual orbitals from orthogonal IAO orbitals, C_ao_iao.
    Math: (1 - |IAO><IAO|) |B1> where B1 only choose the remaining virtual AO basis.
    """
    mol = cell
    pmol = lo.iao.reference_mol(mol, minao)

    B1_labels = mol.ao_labels()
    B2_labels = pmol.ao_labels()
    virt_idx = [idx for idx, label in enumerate(B1_labels) if (not label in B2_labels)]
    nB1 = mol.nao_nr()
    nB2 = pmol.nao_nr()
    nvirt = len(virt_idx)
    assert(nB2 + nvirt == nB1)
    
    CCdS = reduce(np.dot, (C_ao_iao, C_ao_iao.conj().T, S))
    C_virt = (np.eye(nB1) - CCdS)[:, virt_idx]
    return C_virt


x = .63
mol = gto.M(atom=[['C', (0, 0, 0)],
                  ['H', (x ,  x,  x)],
                  ['H', (-x, -x,  x)],
                  ['H', (-x,  x, -x)],
                  ['H', ( x, -x, -x)]],
            basis='321g')
mf = scf.RHF(mol).run()

mo_occ = mf.mo_coeff[:,mf.mo_occ>0]

S = mf.get_ovlp()

C_val = iao.iao(mol, mo_occ)
C_val = lo.vec_lowdin(C_val, S)

mo_occ = reduce(np.dot, (C_val.T, S, mo_occ))

dm = np.dot(mo_occ, mo_occ.T) * 2

molden.from_mo(mol, 'iao_val.molden', C_val)

C_virt = get_iao_virt(mol, C_val, S, minao='minao')
C_virt = lo.vec_lowdin(C_virt, S)

molden.from_mo(mol, 'iao_virt.molden', C_virt)

