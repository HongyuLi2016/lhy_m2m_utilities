#!/usr/bin/env python
import numpy as np
from read_rc import read_rc
import os
flist = read_rc('glist')
JAM_dir = '/home/lhy/worksapce/manga/MPL4/mcmc/upload_021_2/2016-06-27'
for gname in flist:
  os.system('mkdir -p {}'.format(gname))
  os.system('mkdir -p {}/auxiliary_data'.format(gname))
  os.system('cp /home/lhy/worksapce/manga/MPL4/DAP_MPL4/STON-021_m2m/manga_{}.fits {}/auxiliary_data/IFU.fits'.format(gname,gname))
  os.system('cp /home/lhy/worksapce/manga/MPL4/mge/{}/mge.npy {}/auxiliary_data/'.format(gname,gname))
  os.system('cp {}/{}/rst/rst.npy {}/auxiliary_data/JAM_pars.npy'.format(JAM_dir,gname,gname))
  os.system('cp {}/{}/rst/save.npy {}/auxiliary_data/JAM_profile.npy'.format(JAM_dir,gname,gname))
  os.system('cp /home/lhy/worksapce/manga/MPL4/mge/{}/information.fits {}/auxiliary_data/'.format(gname,gname))
