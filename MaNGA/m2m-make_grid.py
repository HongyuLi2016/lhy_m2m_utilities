#!/usr/bin/env python
import numpy as np
import os
from optparse import OptionParser
from m2m_grid_utils import create_grid_cfg

if __name__=='__main__':
  parser = OptionParser()
  parser.add_option('-g', action='store',type='string' ,dest='gname',default=None,help='galaxy name')
  parser.add_option('-m', action='store',type='string' ,dest='model',default=None,help='model name')
  (options, args) = parser.parse_args()
  inc = np.linspace(53.0,90.0,8)
  ml = np.linspace(8.5,11.5,10)
  lhy = create_grid_cfg(options.gname,model=options.model)
  lhy.create_i_ml(inc,ml)

