#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: m2m_analysis_util.py
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
import util_config as uc
import numpy as np
from JAM.utils import util_fig
from JAM.utils import velocity_plot
import matplotlib.pyplot as plt
from matplotlib import colors
util_fig.ticks_font1.set_size(10)


def readObs(gname, rstFolder, key):
    data = np.genfromtxt(
        '{}/{}/observables/{}mfile1'.format(gname, rstFolder, key),
        dtype=[('binID', 'i8'),
               ('obs', 'f8'),
               ('obs_err', 'f8'),
               ('inUse', 'i8'),
               ('modelRaw', 'f8'),
               ('smoothDelta', 'f8'),
               ('xbin', 'f8'),
               ('ybin', 'f8')])
    rst = {}
    rst['binID'] = data['binID']
    rst['obs'] = data['obs']
    rst['obs_err'] = data['obs_err']
    rst['inUse'] = data['inUse'].astype(bool)
    rst['modelRaw'] = data['modelRaw']
    rst['smoothDelta'] = data['smoothDelta']
    rst['xbin'] = data['xbin']
    rst['ybin'] = data['ybin']
    rst['model'] = data['obs'] + data['obs_err'] * data['smoothDelta']
    return rst


def plot_map(xbin, ybin, obs, model, goodbins=None, symmetry=False,
             title='', barlabel='$\mathbf{km/s}$'):
    if goodbins is None:
        goodbins = np.ones_like(xbin, dtype=bool)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
    fig.subplots_adjust(left=0.08, bottom=0.1, right=0.92, top=0.98,
                        wspace=0.6, hspace=0.2)
    if symmetry:
        vmax = np.percentile(abs(obs[goodbins]), 98.0)
        norm = colors.Normalize(vmin=-vmax, vmax=vmax)
    else:
        vmin, vmax = np.percentile(obs[goodbins], [2.0, 98.0])
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    velocity_plot(xbin, ybin, obs, ax=axes[0], text='Obs',
                  norm=norm, barlabelSize=10, barlabel=barlabel)
    velocity_plot(xbin, ybin, model, ax=axes[1], text='Model',
                  norm=norm, barlabelSize=10, barlabel=barlabel)
    vmax = np.percentile(abs(model[goodbins]-obs[goodbins]), 98.0)
    norm_res = colors.Normalize(vmin=-vmax, vmax=vmax)
    velocity_plot(xbin, ybin, (model-obs), ax=axes[2], text='Residual',
                  norm=norm_res, barlabelSize=10, barlabel=barlabel)
    axes[0].set_ylabel('$\mathbf{R/R_e}$', fontproperties=util_fig.label_font)
    axes[1].set_title(title, fontproperties=util_fig.label_font)
    return fig, axes


class M2Mrst:

    def __init__(self, gname, model, rstFolder=None):
        if rstFolder is None:
            rstFolder = 'rst_{}'.format(model)
        self.gname = gname
        self.rstFolder = rstFolder
        self.model = model
        self.xmodel_name, self.xconfig = \
            uc.get_config('{}/{}/{}.cfg'.format(gname, rstFolder, model),
                          model)
        # read units
        pc_km = 3.0856775975e13
        as_pc = np.pi / 0.648
        tenmegayear = 3600.0 * 24.0 * 365 * 1e7
        try:
            info = np.genfromtxt('{}/auxiliary_data/information.dat'
                                 .format(gname))
            self.eps = info[1]
            self.dist = info[2]
            self.Re_arcsec = info[3]
            self.Re_kpc = self.Re_arcsec * self.dist * as_pc * 1e-3
            # vel in [Re/10Mys] * self.vel2kms = vel in [km/s]
            self.vel2kms = 1.0 / (tenmegayear * 1e-3 / pc_km / self.Re_kpc)
        except:
            print('Warning - No infomation.dat file is provided'
                  'in auxiliary_data')
            self.eps = 1.0
            self.dist = np.nan
            self.Re_arcsec = 1.0
            self.Re_kpc = 1.0
            self.vel2kms = 1.0

        # read luminosity results
        nobs_lum = self.xconfig.getint('sec:Lum_constraints', 'num_obs')
        list_lum = [self.xconfig.get('sec:Lum_constraints',
                                     'obsx{}'.format(i)).split(':')[1]
                    for i in range(nobs_lum)]
        self.lum_data = {}
        for lum in list_lum:
            self.lum_data[lum] = readObs(gname, rstFolder, lum)

        # read kinematic results
        nobs_kin = self.xconfig.getint('sec:Kin_constraints', 'num_obs')
        list_kin = [self.xconfig.get('sec:Kin_constraints',
                                     'obsx{}'.format(i)).split(':')[1]
                    for i in range(nobs_kin)]
        self.kin_data = {}
        for kin in list_kin:
            self.kin_data[kin] = readObs(gname, rstFolder, kin)

        # read spectral line results
        nobs_spec = self.xconfig.getint('sec:Spectral_lines', 'num_obs')
        list_spec = [self.xconfig.get('sec:sec:Spectral_lines',
                                      'obsx{}'.format(i)).split(':')[1]
                     for i in range(nobs_spec)]
        self.spec_data = {}
        for spec in list_spec:
            self.spec_data[spec] = readObs(gname, rstFolder, spec)

    def coordinates(self):
        gname = self.gname
        rstFolder = self.rstFolder
        # read particle data
        # x, y, z, vx, vy, vz, lum_weights, inuse
        self.data = np.genfromtxt('{}/{}/particles/coordinates'
                                  .format(gname, rstFolder), skip_header=1,
                                  dtype=['f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                         'f8', 'i8'])
        # initial lum_weights, eor lum_weights, deviation?
        self.lumconv = np.genfromtxt('{}/{}/particles/lumconv'
                                     .format(gname, rstFolder), skip_header=1,
                                     dtype=['f8', 'f8', 'f8'])
        # initial energy, eor energy, inues
        self.energy = np.genfromtxt('{}/{}/particles/energy'
                                    .format(gname, rstFolder), skip_header=1,
                                    dtype=['f8', 'f8', 'i8'])
