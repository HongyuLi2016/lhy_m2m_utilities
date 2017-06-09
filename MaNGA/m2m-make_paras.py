#!/usr/bin/env python
import numpy as np
import os
import pickle
from m2m_manga_utils import write_pymge
# from m2m_manga_utils import write_interp_table
import util_config as uc
from optparse import OptionParser


class make_paras:

    def __init__(self, gname, mname=None, information=False):
        if mname is None:
            mname = gname
        self.model = mname
        # read in configuration file
        self.xmodel_name, self.xconfig = uc.get_config(
            '{}/{}.cfg'.format(gname, self.model), self.model)
        self.size = self.xconfig.getfloat('sec:Model', 'size')
        self.inc_deg = self.xconfig.getfloat('sec:Model', 'inclination')
        self.G = self.xconfig.getfloat('sec:Potential', 'grav_constant')
        self.ml = self.xconfig.getfloat('sec:rt_Luminous_matter',
                                        'mass_to_light')
        self.interp_folder = self.xconfig.get(
            'sec:rt_Luminous_matter', 'interp_folder')
        mge = np.load('{}/auxiliary_data/mge.npy'.format(gname))
        pa, eps = mge[1], mge[2]
        with open('{}/auxiliary_data/JAM_pars.dat'.format(gname)) as f:
            rst = pickle.load(f)
        self.sol = rst['lum2d']    # L_sun/pc^2 arcsec q
        self.pot = rst['dhmge2d']  # L_sun/pc^2 pc q

        self.dist = rst['dist']
        self.Re_arcsec = rst['Re_arcsec']
        # convert units
        pc = self.dist * np.pi / 0.648
        self.pot[:, 1] /= pc  # convert pc to arcsec for dark halo
        self.sol[:, 0] = 2.0 * np.pi * self.sol[:, 0] * \
            (self.sol[:, 1] * pc)**2 * self.sol[:, 2] / 1e10
        self.sol[:, 1] /= self.Re_arcsec
        self.pot[:, 0] = 2.0 * np.pi * self.pot[:, 0] * \
            (self.pot[:, 1] * pc)**2 * self.pot[:, 2] / 1e10 / self.ml
        self.pot[:, 1] /= self.Re_arcsec
        self.L_tot = np.sum(self.sol[:, 0])
        os.system('mkdir -p {}/{}'.format(gname, self.interp_folder))
        write_pymge(self.sol, fname='m2m_mge_lum'.format(self.model),
                    outpath='{}/{}'.format(gname, self.interp_folder))
        write_pymge(self.pot, fname='m2m_mge_dm'.format(self.model),
                    outpath='{}/{}'.format(gname, self.interp_folder))
        if information:
            with open('{}/auxiliary_data/information.dat'
                      .format(gname), 'w') as ff:
                print >>ff, '%.2f' % pa
                print >>ff, '%.2f' % eps
                print >>ff, '%.2f' % self.dist
                print >>ff, '%.2f' % self.Re_arcsec

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-g', action='store', type='string',
                      dest='gname', default=None, help='galaxy name')
    parser.add_option('-m', action='store', type='string',
                      dest='mname', default=None, help='model name')
    (options, args) = parser.parse_args()
    gname = options.gname
    if options.mname is None:
        mname = gname
    else:
        mname = options.mname
    # lhy = make_paras(gname,information=False)
    lhy = make_paras(gname, mname=mname, information=True)
