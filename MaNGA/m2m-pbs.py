#!/usr/bin/env python
# import numpy as np
import os
import glob
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    (options, args) = parser.parse_args()
    if len(args) != 1:
        print 'galaxy name must be provided!'
        exit(1)
    cfg_files = glob.glob('%s/*.cfg' % args[0])
    if len(cfg_files) == 0:
        print 'no .cfg if %s folder' % args[0]
    os.system('mkdir -p %s/pbsscript' % args[0])
    path = os.getcwd()
    for ncfg in cfg_files:
        model_name = ncfg.split('/')[1][0:-4]
        with open('%s/pbsscript/%s.pbs' % (args[0], model_name),
                  'w') as ff:
            print >>ff, '#!/bin/bash'
            print >>ff, '#PBS -N %s' % model_name
            print >>ff, '#PBS -o %s/%s/rst_%s/' % (path, args[0], model_name)
            print >>ff, '#PBS -e %s/%s/rst_%s/' % (path, args[0], model_name)
            print >>ff, '#PBS -l select=1:ncpus=32:mpiprocs=32:mem=32gb'
            print >>ff, '#PBS -V'
            print >>ff, '#PBS -j oe'
            # print >>ff, 'nproc=`cat $PBS_NODEFILE | wc -l`'
            # print >>ff, 'cat $PBS_NODEFILE > NODEFILE'
            print >>ff, 'export OMP_NUM_THREADS=1'
            print >>ff, 'cd %s' % (path)
            print >>ff, 'm2m-prepare.sh %s %s' % (args[0], model_name)
            print >>ff, 'cd %s/%s' % (path, args[0])
            print >>ff, 'm2m-run.sh %s 32' % model_name
            print >>ff, 'cd ..'
            print >>ff, 'm2m-analysis.py %s -m %s' % (args[0], model_name)
