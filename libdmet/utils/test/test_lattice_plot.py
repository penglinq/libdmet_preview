#! /usr/bin/env python

def test_lattice_plot():
    import numpy as np
    from libdmet.utils import lattice_plot
    from libdmet.system import lattice
    import matplotlib
    matplotlib.use('Agg')
    
    Lat = lattice.Square3BandSymm(1, 1, 1, 1)

    latt_plt = lattice_plot.LatticePlot(Lat)
    latt_plt.plot_lattice(noframe=True)
    latt_plt.plot_atoms(rad_list=[1.0, 0.2, 0.2] * 4, \
            color_dic={'Cu': 'gold', 'O': 'C3'})
    latt_plt.plot_spins(m_list=[0.3, 0.0, 0.0, -0.25, -0.0, -0.0001, \
            0.3, 0.0, 0.0, -0.35, 0.0, 0.0])
    latt_plt.plot_text([0.0, 0.0], "test")
    latt_plt.plot_d_orb([4.0, 4.0], direct='down')
    latt_plt.plot_d_orb([0.0, 4.0], direct='up')
    latt_plt.plot_p_orb([3.0, 4.0], direct='right', phase=["+", "-"])
    latt_plt.plot_p_orb([2.0, 4.0], direct='down', phase=["-", "+"])
    latt_plt.plot_bond([1.0, 1.0], [2.0, 1.0], val=-0.01)
    latt_plt.plot_bond([1.0, 1.0], [1.0, 3.0], val=+0.02)
    latt_plt.savefig("latt.png")

def test_plot_3band_order():
    import os
    import numpy as np
    from collections import OrderedDict
    
    from libdmet.system import lattice
    from libdmet.utils import lattice_plot
    from libdmet.utils import get_order_param as order
    import matplotlib
    matplotlib.use('Agg')
    
    GRho_file = os.path.dirname(os.path.realpath(__file__)) + "/GRho_3band"
    GRho = np.load(GRho_file)
    res = order.get_3band_order(GRho)

    latt_plt = lattice_plot.plot_3band_order(res, pairing='Cu-Cu')
    latt_plt = lattice_plot.plot_3band_order(res, pairing='O-O')
    latt_plt = lattice_plot.plot_3band_order(res, pairing='Cu-O')
    
    #latt_plt.show()
    latt_plt.savefig("pairing.png")
    
if __name__ == "__main__":
    test_plot_3band_order()
    test_lattice_plot()
