#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: analysis.py
# Author: Hongyu Li <lhy88562189@gmail.com>
# Date: 08.06.2017
# Last Modified: 08.06.2017
# ============================================================================
#  DESCRIPTION: ---
#      OPTIONS: ---
# REQUIREMENTS: ---
#         BUGS: ---
#        NOTES: ---
# ORGANIZATION:
#      VERSION: 0.0
# ============================================================================
import m2m_analysis_util as au
from optparse import OptionParser
import os


parser = OptionParser()
parser.add_option('-m', action='store', type='string', dest='model',
                  default=None, help='model name')
(options, args) = parser.parse_args()
if len(args) != 1:
    print 'Error - please provide a folder name'
    exit(1)
if options.model is None:
    model = args[0]
else:
    model = options.model
os.system('mkdir -p {}/rst_{}/analysis'.format(args[0], model))

modelRst = au.M2Mrst(args[0], model)
vel = modelRst.kin_data['IFU_vel']
disp = modelRst.kin_data['IFU_disp']
Z = modelRst.spec_data['IFU_Z']
scale = modelRst.vel2kms

fig, axes = au.plot_map(vel['xbin'], vel['ybin'], vel['obs']*scale,
                        vel['model']*scale, goodbins=vel['inUse'],
                        title='Velocity')
fig.savefig('{}/rst_{}/analysis/vel.png'.format(args[0], model), dpi=150)

fig, axes = au.plot_map(disp['xbin'], disp['ybin'], disp['obs']**0.5*scale,
                        disp['model']**0.5*scale, goodbins=disp['inUse'],
                        title='Dispersion')
fig.savefig('{}/rst_{}/analysis/disp.png'.format(args[0], model), dpi=150)


fig, axes = au.plot_map(Z['xbin'], Z['ybin'], Z['obs'],
                        Z['model']**0.5*scale, goodbins=Z['inUse'],
                        title='Z')
fig.savefig('{}/rst_{}/analysis/Z.png'.format(args[0], model), dpi=150)
