#!/usr/bin/env python
import numpy as np
from read_rc import read_rc
import os
flist = read_rc('glist')
for gname in flist:
  with open('{}/create_all.sh'.format(gname),'w') as ff:
    print >>ff, '#!/bin/bash'
    print >>ff, 'date > {}/stdcreate.log'.format(gname)
    print >>ff, 'echo create .cfg file >> {}/stdcreate.log'.format(gname)
    print >>ff, 'make_cfg.py -g {} >> {}/stdcreate.log'.format(gname,gname)
    print >>ff, 'echo create mge parameter files >> {}/stdcreate.log'.format(gname)
    print >>ff, 'make_paras.py -g {} >> {}/stdcreate.log'.format(gname,gname)
    print >>ff, 'echo create data and bin files >> {}/stdcreate.log'.format(gname)
    print >>ff, 'make_data.py -g {} >> {}/stdcreate.log'.format(gname,gname)
  os.system('chmod +x {}/create_all.sh'.format(gname))
