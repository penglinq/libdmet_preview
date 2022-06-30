
Quickstart
**********

This quickstart provides a brief introduction to the use of libDMET in common quantum chemical simulations. These make reference to specific examples within the dedicated `examples <https://github.com/pyscf/pyscf/tree/master/examples>`_ directory. For brevity, and so as to not repeat a number of function calls, please note that the cells below often share objects in-between one another. The selection below is far from exhaustive: additional details on the various modules are presented in the accompanying user guide and within the examples directory.

.. _INPUT:

Input
=============
Lattice

Mean-field calculation: PySCF mean-field object

High-level solver: three options


.. _LOCAL:

Localization
=================

.. _LOC:

Default Intrinsic Atomic Orbitals (IAO)
--------------------------------------

libDMET by default uses Intrinsic Atomic Orbitals (IAO) from PySCF to localize atomic orbitals. (cf. `local_orb/03-split_localization.py <https://github.com/pyscf/pyscf/blob/master/examples/local_orb/03-split_localization.py>`_):

  >>> from pyscf import lo
  >>> occ_orbs = rhf_h2o.mo_coeff[:, rhf_h2o.mo_occ > 0.]
  >>> fb_h2o = lo.Boys(mol_h2o, occ_orbs, rhf_h2o) # Foster-Boys
  >>> loc_occ_orbs = fb.kernel()
  >>> virt_orbs = rhf_h2o.mo_coeff[:, rhf_h2o.mo_occ == 0.]
  >>> pm_h2o = lo.PM(mol_h2o, virt_orbs, rhf_h2o) # Pipek-Mezey
  >>> loc_virt_orbs = pm.kernel()
  
One can use other customized localized orbitals by directly passing in the basis transformation matrix C_ao_lo. 

Wannier orbitals can be computed as (cf. `local_orb/04-ibo_benzene_cubegen.py <https://github.com/pyscf/pyscf/blob/master/examples/local_orb/04-ibo_benzene_cubegen.py>`_):

  >>> iao = lo.wannier(mol, occ_orbs)
  >>> iao = lo.vec_lowdin(iao, rhf_h2o.get_ovlp())
  >>> ibo = lo.ibo.ibo(mol, occ_orbs, iaos=iao)

High-level calculations
===========================

.. _HL:

Common solvers with default
---------------------------


Build-in solvers
--------------------------


Interface with new solver
--------------------------
