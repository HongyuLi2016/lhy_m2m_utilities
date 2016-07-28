#!/usr/bin/env python
import numpy as np
import os
import glob
from optparse import OptionParser

if __name__ == '__main__':
  parser = OptionParser()
  parser.add_option('-g', action='store',type='string' ,dest='gname',default=None,help='galaxy name')
  (options, args) = parser.parse_args()
  if options.gname is None:
    print 'galaxy name must be provided!'
    exit(1)
  pbs_files = glob.glob('%s/pbsscript/*.pbs'%options.gname)
  if len(pbs_files) == 0:
    print 'no .pbs in %s/pbsscript folder'%options.gname
  for npbs in pbs_files:
    os.system('qsub %s'%npbs)
    os.system('sleep 2.0')

