#!/usr/bin/env python
# import numpy as np
import os
import glob
from optparse import OptionParser

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-a', action='store_true', dest='add',
                      default=False, help='add models')
    (options, args) = parser.parse_args()
    if len(args) != 1:
        print 'galaxy name must be provided!'
        exit(1)
    pbs_files = sorted(glob.glob('%s/pbsscript/*.pbs' % args[0]))
    if len(pbs_files) == 0:
        print 'no .pbs in %s/pbsscript folder' % args[0]
    for npbs in pbs_files:
        if options.add:
            model_name = npbs.split('/')[-1][0:-4]
            if not os.path.exists(
                    '{}/rst_{}'.format(args[0], model_name)):
                print '%s do not exist, qsub again' % model_name
                os.system('qsub %s' % npbs)
                os.system('sleep 0.2')
        else:
            os.system('qsub %s' % npbs)
            os.system('sleep 0.2')
