
Quickstart
**********

This quickstart provides a brief introduction to the use of libDMET in common periodic embedding calculations from a PySCF style user interface. Completed DMET examples with detailed guidance can be found within the dedicated `run <https://github.com/penglinq/libdmet_preview/tree/cookiecutter/examples/run>`_ directory. For more advanced usage, please see the detailed user documentation. 
Running a general DMET calculation can be divided into the following steps:

1. Define a lattice object that includes a unit cell and kmesh and that defines valence, virtual, and core orbitals.
2. Perform a low-level calculation, e.g. HF here, on the same cell and k-points.
3. Select a high-level solver and, along with the lattice and low-level calculation, instantiate a DMET object.
4. Use .kernel() to run DMET. 
5. Obtain properties e.g. density matrices and energies can be obtained by the corresponing method following PySCF's standard, e.g. .make_rdm1(), .energy_cell().

The following sections provide more detailed explanation on each step.


.. _INPUT:

Input: Lattice
=============
For periodic embedding calculations, the system to study is defined using an instance of the Lattice class in libDMET.
The Lattice is instantiated from a PySCF cell object that defines the unit cell and a kmesh that defines the supercell. 

  >>> cell = gto.Cell()
  >>> cell.build(unit = 'angstrom',
  >>>      a = [[2.50, 0.0, 0.0], [-1.25, 2.1650635094610964, 0.0], [0.0, 0.0, 20]],
  >>>      atom = 'B 0.0 0.0 0.0; N 1.25 0.721687836487032 0.0',
  >>>      dimension = 3,
  >>>      pseudo = 'gth-pade',
  >>>      basis='gth-dzv',
  >>>      precision = 1e-12)
  >>> kmesh = [3, 3, 1]
  >>> lat = lattice.Lattice(cell, kmesh) 


Input: Low-level calculation
=============
A low-level calculation obtains an approximated wave function of the whole lattice using an economic solver such as kHF, kDFT. Only mean-field methods in PySCF are supported by the current bath construction methods. 

  >>> gdf = df.GDF(cell, kpts)
  >>> gdf._cderi_to_save = gdf_fname
  >>> gdf.build()
  >>> kmf = scf.KRHF(cell, kpts, exxdiv=None)
  >>> kmf.with_df = gdf
  >>> kmf.with_df._cderi = gdf_fname
  >>> kmf.kernel()

Input: High-level solver
=============
The high-level solver solves an embedding Hamiltonian accurately in a small embedded space derived from Schmidt decomposition. Solving the embedding problem is normally the computational time bottleneck, so the high-level solver should be chosen based on both the accuracy and the amount of computational resource. Common solvers include FCI for small embedded space and CCSD which can treat hundreds of orbitals. 
Here, the solver could be an object of the built-in solver under the sovler folder, an object of a child of the dmetsolvermixin class or a string of one of the following: "ccsd" or "fci". Customization of solver should be passed through solver's attributes.

  >>> from libdmet.solver import impurity_solver
  >>> solver = impurity_solver.CCSD()
  >>> solver.restricted = True
  >>> solver.conv_tol = cell.natm * 1e-9
  >>> solver.conv_tol_normt = cell.natm * 1e-6
  >>> solver.max_memory = max_memory 

Run DMET calculation
====================
Similar to a solver call in PySCF, a DMET calculation can be performed by first instantiating a dmet object with
previously defined low-level calculation, lattice, and high-level solver and then running with .kernel(). 

  >>> mydmet = dmet.dmet(kmf, lat, solver)
  >>> e_dmet, conv = mydmet.kernel() # e_dmet = energy per cell

.. _LOCAL:

Localization
=================
By default DMET uses Intrinsic Atomic Orbital (IAO) to localize atomic orbitals. One can specify the minimal basis for IAO construction by setting mydmet.minao. For localization scheme other than IAO, input a predetermined set of localized orbitals (LO) by setting mydmet.C_ao_lo to be the basis transformation matrix from AO to LO.

.. _LOC:

Built-in localization
--------------------------------------

libDMET by default uses Intrinsic Atomic Orbitals (IAO) from PySCF to localize the core and valence atomic orbitals. (cf. `local_orb/03-split_localization.py <https://github.com/pyscf/pyscf/blob/master/examples/local_orb/03-split_localization.py>`_) Projected Atomic Orbitals (PAO) are used to generate localized virtual orbitals.
Another commonly used localization scheme is the Maximal Localized Wannier functions. One can generate Wannier functions using built-in interface as below and directly passing in the basis transformation matrix through attribute C_ao_lo. 

Wannier orbitals can be computed as (cf. `local_orb/04-ibo_benzene_cubegen.py <https://github.com/pyscf/pyscf/blob/master/examples/local_orb/04-ibo_benzene_cubegen.py>`_):

  >>> iao = lo.wannier(mol, occ_orbs)
  >>> iao = lo.vec_lowdin(iao, rhf_h2o.get_ovlp())
  >>> ibo = lo.ibo.ibo(mol, occ_orbs, iaos=iao)


