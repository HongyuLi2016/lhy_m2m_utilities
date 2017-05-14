#!/usr/bin/env python
import numpy as np
import os
import glob
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-g', action='store', type='string', dest='gname',
                      default=None, help='galaxy name')
    (options, args) = parser.parse_args()
    if options.gname is None:
        print 'galaxy name must be provided!'
        exit(1)
    cfg_files = glob.glob('%s/*.cfg' % options.gname)
    if len(cfg_files) == 0:
        print 'no .cfg if %s folder' % options.gname
    os.system('mkdir -p %s/pbsscript' % options.gname)
    path = os.getcwd()
    for ncfg in cfg_files:
        model_name = ncfg.split('/')[1][0:-4]
        with open('%s/pbsscript/%s.pbs' % (options.gname, model_name), 'w') as ff:
            print >>ff, '#!/bin/bash'
            print >>ff, '#PBS -N %s' % model_name
            print >>ff, '#PBS -o %s/%s/rst_%s/' % (
                path, options.gname, model_name)
            print >>ff, '#PBS -e %s/%s/rst_%s/' % (
                path, options.gname, model_name)
            print >>ff, '#PBS -l nodes=1:ppn=32'
            print >>ff, 'nproc=`cat $PBS_NODEFILE | wc -l`'
            print >>ff, 'cat $PBS_NODEFILE > NODEFILE'
            print >>ff, 'cd %s' % (path)
            print >>ff, '/home/lhy/bin/prepare.sh %s %s' % (
                options.gname, model_name)
            print >>ff, 'cd %s/%s' % (path, options.gname)
            print >>ff, '/home/lhy/bin/M2M.sh %s 32' % model_name
