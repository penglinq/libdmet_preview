#! /usr/bin/env python

def test_poscar():
    """
    Test pdos routine and plot.
    """
    import numpy as np
    from pyscf.pbc import gto
    from libdmet.utils.iotools import read_poscar, write_poscar
    from libdmet.utils.misc import max_abs

    cell = gto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''
    cell.atom = ''' H  5.0       5.0      0.75
                    H  5.0       5.0      2.25
                    C  10.0      5.0      0.0
                    He 10.0      5.0      2.25
                    C  10.0      5.0      5.0 '''
    cell.basis = 'minao'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')
    
    write_poscar(cell, species=False)
    write_poscar(cell, cart=True)
    cell2 = read_poscar()
    cell2.basis = 'minao'
    cell2.build()

    # lattice_vectors, atom should be the same
    assert max_abs(cell.lattice_vectors() - cell2.lattice_vectors()) < 1e-12
    labels1, coords1 = list(zip(*cell._atom))
    labels2, coords2 = list(zip(*cell2._atom))
    coords1 = np.asarray(coords1)
    coords2 = np.asarray(coords2)
    assert sorted(labels1) == sorted(labels2)
    assert max_abs(np.sort(coords1, axis=None) - np.sort(coords2, axis=None)) < 1e-12

def test_struct_dump():
    import numpy as np
    from libdmet.utils.iotools import struct_dump, sc2POSCAR
    from libdmet.system.lattice import Square3BandSymm
    ImpSize = [2, 2, 1]
    cellsize = np.eye(3) * 2.0
    atoms = [(np.array([0.0, 0.0, 0.0]), "Cu"), 
             (np.array([1.0, 0.0, 0.0]),  "O"),
             (np.array([0.0, 1.0, 0.0]),  "O")]
    
    struct_dump(cellsize, ImpSize, atoms, fmt="POSCAR", frac=False, center=None, \
        filename=None, elements=None)
    
    struct_dump(cellsize, ImpSize, atoms, fmt="XYZ", frac=False, center=None, \
        filename="CuO2.xyz", elements=None)
    
    latt = Square3BandSymm(1, 1, 1, 1)
    sc2POSCAR(latt.supercell, fout="test.vasp")

def test_change_cell_shape():
    import os
    import numpy as np
    from libdmet.utils.iotools import change_cell_shape
    from libdmet.utils import misc
    from libdmet.utils.misc import max_abs
    pos_file = os.path.dirname(os.path.realpath(__file__)) + "/HBCO.pos"

    cell = misc.read_poscar(fname=pos_file)
    vec_new = np.array([[7.7428, 0.0000, 0.0000],
                        [3.8714, 3.8714, 0.0000],
                        [0.0000, 0.0000, 9.5023]])
    #vec_new = np.array([[7.7428, 0.0000, 0.0000],
    #                    [0.0000, 7.7428, 0.0000],
    #                    [0.0000, 0.0000, 9.5023]])
    cell_new = change_cell_shape(cell, vec_new, search_range=[2, 2, 2], \
            origin=[-7.7428*0.25, 0.0, 0.0])
    misc.write_poscar(cell_new, cart=False)

if __name__ == "__main__":
    test_change_cell_shape()
    test_struct_dump()
    test_poscar()
