#!/usr/bin/env python
import numpy as np
import os
from manga_util import write_mge
import util_config as uc
from optparse import OptionParser

class make_paras:
  def __init__(self,gname,mname=None,information=False):
    if mname is None:
      mname = gname
    self.model = mname
    #read in configuration file
    self.xmodel_name, self.xconfig = uc.get_config('{}/{}.cfg'.format(gname, self.model), self.model)    
    self.size = float(self.xconfig.get('sec:Model', 'size'))
    self.inc_deg = float(self.xconfig.get('sec:Model', 'inclination'))
    self.G = float(self.xconfig.get('sec:Potential', 'grav_constant'))
    self.outfolder = 'MGE{}'.format(self.model)
    rst = np.load('{}/auxiliary_data/JAM_pars.npy'.format(gname))
    sol, pa, eps = np.load('{}/auxiliary_data/mge.npy'.format(gname))
    self.sol = rst[0]
    self.pot = np.array([rst[10],rst[11],rst[12]]).T
    self.dist = rst[6]
    self.Re_arcsec = rst[7]  
    # convert units 
    pc = self.dist * np.pi / 0.648
    self.sol[:,0] = 2.0 * np.pi * self.sol[:,0] * (self.sol[:,1] * pc)**2 * self.sol[:,2] / 1e10 
    self.sol[:,1] /= self.Re_arcsec
    self.pot[:,0] = 2.0 * np.pi * self.pot[:,0] * (self.pot[:,1] * pc)**2 * self.pot[:,2] / 1e10  
    self.pot[:,1] /= self.Re_arcsec
    write_mge(self.size,self.inc_deg,self.G,self.sol,fname='mge_params_{}'.format(self.model),outpath='{}/auxiliary_data'.format(gname))
    write_mge(self.size,self.inc_deg,self.G,self.pot,fname='pot_params_{}'.format(self.model),outpath='{}/auxiliary_data'.format(gname))
    if information:
      with open('{}/auxiliary_data/information.dat'.format(gname),'w') as ff:
        print >>ff, '%.2f'%pa
        print >>ff, '%.2f'%eps
        print >>ff, '%.2f'%self.dist
        print >>ff, '%.2f'%self.Re_arcsec

if __name__ == '__main__':
  parser = OptionParser()
  parser.add_option('-g', action='store',type='string' ,dest='gname',default=None,help='galaxy name')
  parser.add_option('-m', action='store',type='string' ,dest='mname',default=None,help='model name')
  (options, args) = parser.parse_args()
  gname = options.gname
  if options.mname is None:
    mname = gname
  else:
    mname = options.mname
  #lhy = make_paras(gname,information=False)
  lhy = make_paras(gname,mname=mname,information=True)
