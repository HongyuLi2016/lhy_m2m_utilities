#!/usr/bin/env python
import numpy as np
from read_rc import read_rc
import os
flist = read_rc('glist')
out_name = 'm2mrun'
for gname in flist:
  with open('{}/M2M.sh'.format(gname),'w') as ff:
    print >>ff, '#!/bin/bash'
    print >>ff, 'mpiexec -np 8 execm2m.py  -o{} {} >std{}'.format(out_name,gname,out_name)
    print >>ff, 'cp {}.cfg {}'.format(gname,out_name)
    print >>ff, 'eor_mean_chi2.py {} >> std{}'.format(out_name,out_name)
    print >>ff, 'eor_entropy.py {} >> std{}'.format(out_name,out_name)
    print >>ff, 'eor_obs.py {} > {}/observables/repro_analysis'.format(out_name,out_name)
    print >>ff, 'eor_energy.py {} > {}/particles/energy_analysis'.format(out_name,out_name)
    print >>ff, 'eor_weights.py {} > {}/particles/weights_analysis'.format(out_name,out_name)
    print >>ff, 'mv std{} {}'.format(out_name,out_name)
  os.system('chmod +x {}/M2M.sh'.format(gname))
