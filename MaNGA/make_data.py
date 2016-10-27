#!/usr/bin/env python
import numpy as np
import os
from manga_util import write_mge
from manga_util import symmetrize_velfield
import util_config as uc
from create_data_bin import create
import pyfits
from optparse import OptionParser


if __name__ == '__main__':
  parser = OptionParser()
  parser.add_option('-g', action='store',type='string' ,dest='gname',default=None,help='galaxy name')
  parser.add_option('-m', action='store',type='string' ,dest='mname',default=None,help='model name')
  parser.add_option('-s', action='store_false',dest='symmetrize',default=True,help='model name')
  (options, args) = parser.parse_args()
  gname = options.gname
  if options.mname is None:
    mname = gname
  else:
    mname = options.mname

  lhy = create(mname, folder = gname)
  with open('{}/auxiliary_data/information.dat'.format(gname),'r') as ff: 
    ang = float(ff.readline()) 
    eps = float(ff.readline())
    dist = float(ff.readline())
    Re_arcsec = float(ff.readline())
  mge = np.genfromtxt('{}/MGE{}/m2m_mge_lum'.format(gname,mname),skip_header=1)
  hdulist = pyfits.open('{}/auxiliary_data/IFU.fits'.format(gname))
  data1 = hdulist[1].data
  data2 = hdulist[2].data
  x0 = data1['xbin']
  y0 = data1['ybin']   
  tem_rebin_x = data2['rebin_x']
  tem_rebin_y = data2['rebin_y']

  pa=np.radians(ang - 90.0)
  xbin = np.cos(pa)*x0-np.sin(pa)*y0
  ybin = np.sin(pa)*x0+np.cos(pa)*y0
  rebin_x = np.cos(pa) * tem_rebin_x - np.sin(pa) * tem_rebin_y
  rebin_y = np.sin(pa) * tem_rebin_x + np.cos(pa) * tem_rebin_y

  r = (xbin**2 + ybin**2)**0.5
  ii = r < 3.0
  v0 = data1['v0']
  vel = v0 - v0[ii].mean()
  # rotate the data until velocity have positive value for x < 0 
  iii = xbin < 0.0
  rotate = vel[iii].mean()
  if rotate < 0.0:
    pa=np.radians(ang - 90.0 + 180.0)
    xbin = np.cos(pa)*x0-np.sin(pa)*y0
    ybin = np.sin(pa)*x0+np.cos(pa)*y0
    rebin_x = np.cos(pa) * tem_rebin_x - np.sin(pa) * tem_rebin_y
    rebin_y = np.sin(pa) * tem_rebin_x + np.cos(pa) * tem_rebin_y
  v0_err = data1['v0_err'].clip(10.0,200.0)
  #v0_err = np.zeros_like(vel) + 15.0
  vd = data1['vd']
  vd_err = data1['vd_err'].clip(10.0,200.0)
  #vd_err = (vd.copy() * 0.05).clip(6.0)
  h3 = data1['h3']
  h3_err = data1['h3_err']
  h4 = data1['h4']
  h4_err = data1['h4_err']


  mask_filename = '{}/auxiliary_data/IFU_mask.fits'.format(gname)
  if os.path.exists(mask_filename):
    mask = pyfits.open(mask_filename)[0].data
  else:
    mask = np.zeros_like(xbin, dtype=int)
  goodbins = (mask==0).astype(int)

  #mge[:,0] = mge[:,0] / (2.0 * np.pi * mge[:,1]**2 * mge[:,2]) # do not use this again, it has been included in the create_data_bin.py file
  lhy.surface_brightness(mge)
  inc_deg = lhy.xconfig.getfloat('sec:Model','inclination')
  lhy.luminosity_density(mge,inc_deg)
  lhy.IFU(xbin,ybin,vel=vel,vel_err=v0_err,disp=vd,disp_err=vd_err,h3=h3,h3_err=h3_err,\
          h4=h4,h4_err=h4_err,rebin_x=rebin_x,rebin_y=rebin_y,dist=dist,n_part=300000,\
          plot=False,Re=Re_arcsec,good=goodbins, symmetrize=options.symmetrize)
