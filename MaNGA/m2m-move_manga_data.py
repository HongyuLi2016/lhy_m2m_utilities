#!/usr/bin/env python
import numpy as np
import os
flist = np.genfromtxt('glist', dtype='S30')
data_dir = '/home/lhy/manga/mpl5/all'
rst_dir = '/home/lhy/manga/mpl5/all/run2'
for gname in flist:
    ID = gname.split('_')[1]
    os.system('mkdir -p {}'.format(gname))
    os.system('mkdir -p {}/auxiliary_data'.format(gname))
    os.system('cp {}/{}/manga_{}.fits {}/auxiliary_data/IFU.fits'
              .format(data_dir, gname, ID, gname))
    os.system('cp {}/{}/mge.npy {}/auxiliary_data/'
              .format(data_dir, gname, gname))
    os.system('cp {}/{}/rst.dat {}/auxiliary_data/JAM_pars.dat'.format(rst_dir,gname,gname))
    os.system('cp {}/{}/rst.txt {}/auxiliary_data/JAM_pars.txt'.format(rst_dir,gname,gname))
    # os.system('cp {}/{}/rst/save.npy {}/auxiliary_data/JAM_profile.npy'.format(JAM_dir,gname,gname))
    os.system('cp {}/{}/information.fits {}/auxiliary_data/'
              .format(data_dir, gname, gname))
