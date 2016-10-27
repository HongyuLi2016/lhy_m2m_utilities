#!/usr/bin/env python
import numpy as np
from scipy import interpolate

def write_mge(size,inc_deg,G,sol,fname='mge_params',outpath='./'):
  '''
  write mge data into file for creating interpolation table using cmge
  sol unit: Luminosity 10^10 Lsun sigma in Re
  '''
  with open('{}/{}'.format(outpath,fname),'w') as ff:
    print >>ff, '{:e}'.format(size)
    print >>ff, '{:.2f}'.format(inc_deg)
    print >>ff, '{:e}'.format(G)
    print >>ff, '{:e} {:e}'.format(0.1,1e-4)
    print >>ff, '{}'.format(sol.shape[0])
    for i in range(sol.shape[0]):
      print >>ff, '{:e} {:e} {:e}'.format(sol[i,0],sol[i,1],sol[i,2])

def write_pymge(sol,fname='m2m_mge_lum',outpath='./'):
  '''
  write mge data into file for creating interpolation table using pymge
  sol unit: Luminosity 10^10 Lsun sigma in Re
  '''
  with open('{}/{}'.format(outpath,fname),'w') as ff:
    print >>ff, '{}'.format(sol.shape[0])
    for i in range(sol.shape[0]):
      print >>ff, '{:e} {:e} {:e}'.format(sol[i,0],sol[i,1],sol[i,2])



def write_interp_table(size,inc_deg,G,sol,L_tot,fname='mge_pf',num_R=1500,num_Z=1500,outpath='./'):
  '''
  write pot data into file, which will subsitute the file in mge folder and give the correct total luminosity
  sol unit: Luminosity 10^10 Lsun sigma in Re
  '''
  with open('{}/{}'.format(outpath,fname),'w') as ff:
    print >>ff, '{:e}'.format(size)
    print >>ff, '{:e}'.format(inc_deg)
    print >>ff, '{:e}'.format(G)
    print >>ff, '{:e}  {:e}'.format(0.1,1e-4)
    print >>ff, '{}'.format(sol.shape[0])
    for i in range(sol.shape[0]):
      print >>ff, '{:e} {:e} {:e}'.format(sol[i,0],sol[i,1],sol[i,2])
    print >>ff,'{:d} {:d}'.format(num_R,num_Z) 
    print >>ff,'{:e}'.format(L_tot)

def _rotate_points(x, y, ang):
    """
    Rotates points conter-clockwise by an angle ANG in degrees.
    Michele cappellari, Paranal, 10 November 2013
    
    """
    theta = np.radians(ang - 90.)
    xNew = x*np.cos(theta) - y*np.sin(theta)
    yNew = x*np.sin(theta) + y*np.cos(theta)
    return xNew, yNew
    
#----------------------------------------------------------------------
    
def symmetrize_velfield(xbin, ybin, velBin, sym=2, pa=90.):
    """
    This routine generates a bi-symmetric ('axisymmetric') 
    version of a given set of kinematical measurements.
    PA: is the angle in degrees, measured counter-clockwise,
      from the vertical axis (Y axis) to the galaxy major axis.
    SYM: by-simmetry: is 1 for (V,h3,h5) and 2 for (sigma,h4,h6)
    """        
    xbin = np.asarray(xbin)
    ybin = np.asarray(ybin)
    velBin = np.asarray(velBin)
    x, y = _rotate_points(xbin, ybin, -pa)  # Negative PA for counter-clockwise
    
    xyIn = np.column_stack([x, y])
    xout = np.hstack([x,-x, x,-x])
    yout = np.hstack([y, y,-y,-y])
    xyOut = np.column_stack([xout, yout])
    velOut = interpolate.griddata(xyIn, velBin, xyOut)
    velOut = velOut.reshape(4, xbin.size)
    
    if sym == 1:
        velOut[[1,3],:] *= -1.
    velSym = np.nanmean(velOut, axis=0)
    return velSym.copy()

