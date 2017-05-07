#!/usr/bin/env python
import numpy as np
import os
from optparse import OptionParser
import pyfits
import pickle
cdir = os.getcwd()

duration = 150
size = 5.0
n_part = 100000
integration_time_step = 0.01
epsilon = 0.0002
#recalculate_weights_frequency = 10

scale_million = 1e10
pc_km = 3.0856775975e13
tenmegayear = 3600.0 * 24.0 * 365 * 1e7
msun = 1.9884e30 * scale_million
gravconstant = 6.6742e-11

if __name__=='__main__':
  parser = OptionParser()
  parser.add_option('-g', action='store',type='string' ,dest='gname',default=None,help='galaxy name')
  parser.add_option('-m', action='store',type='string' ,dest='mname',default=None,help='model name')
  (options, args) = parser.parse_args()
  gname = options.gname
  if options.mname is None:
    mname = gname
  else:
    mname = options.mname
  os.chdir('{}/{}'.format(cdir,gname))
  os.system('define_M2M_model.py {}'.format(mname))
  zz = pyfits.open('auxiliary_data/information.fits')[1].data['redshift'][0]
  with open('auxiliary_data/JAM_pars.dat') as f:
      rst = pickle.load(f)
  sol = rst['lum2d']
  ml = rst['bestPars'][2]
  inc_deg = np.degrees(np.arccos(rst['bestPars'][0]))
  dist = rst['dist']
  Re_arcsec = rst['Re_arcsec']
  size = (sol[:,1].max()/Re_arcsec*3.0).clip(5,None)
  Re_kpc = Re_arcsec * dist * np.pi / 0.648 * 1e-3
  revised_gravconstant = gravconstant * msun * tenmegayear * tenmegayear / (pc_km * pc_km * pc_km * 1e18 * Re_kpc * Re_kpc * Re_kpc)
  # model parameters
  os.system('update_model.py {} -t{} -s{} -i{} -uyes'.format(mname,duration,size,inc_deg))
  os.system('define_M2M_particles.py {} -n{} -pfrom_elz -vfrom_elz -fP{}_{:.3f}'.format(mname,n_part,mname,ml))
  os.system('update_lm.py {} -m{:.3f} -nMGE -fMGE{}'.format(mname, ml, mname))
  os.system('update_grav_constant.py {} -g{:e}'.format(mname, revised_gravconstant))
  os.system('update_orbit_int.py {} -t{:f}'.format(mname, integration_time_step))
  os.system('update_weight_adapt.py {} -e{:f} -p"(0,-1)"'.format(mname,epsilon))
  os.system('update_potential.py {}'.format(mname))
  # bin scheme
  os.system('define_M2M_scheme.py {} -nld -taxisym -i"(16,32)" -s"(3.0)" -f{}bins'.format(mname,mname))
  os.system('define_M2M_scheme.py {} -nsb -tpolar -i"(16,16)" -s"(3.0)" -f{}bins'.format(mname,mname))
  os.system('define_M2M_scheme.py {} -nIFU -ttree -f{}bins'.format(mname,mname))

  os.system('define_M2M_observ.py {} -nld_data -tld -bld -cMaNGA -r"(0,-1,1e-6,1.0)" -f{}data'.format(mname,mname,mname))
  os.system('define_M2M_observ.py {} -nsb_data -tsb -bsb -cMaNGA -r"(0,-1,1e-6,1.0)" -f{}data'.format(mname,mname,mname))
  os.system('define_M2M_observ.py {} -nIFU_vel  -tlosvelocity -bIFU -cMaNGA -r"(0,-1,1e-6,1.0)" -f{}data'.format(mname,mname))
  os.system('define_M2M_observ.py {} -nIFU_disp -tlosveldisp  -bIFU -cMaNGA -r"(0,-1,1e-6,1.0)" -f{}data'.format(mname,mname))
  os.system('update_observ.py {} -nld_data -uno -r"(0,-1,2e-3,1.0)"'.format(mname))
  os.system('update_observ.py {} -nsb_data -uno -r"(0,-1,2e-3,1.0)"'.format(mname))
  os.system('update_observ.py {} -nIFU_vel -uno -r"(0,-1,2e-3,1.0)"'.format(mname))
  os.system('update_observ.py {} -nIFU_disp -uno -r"(0,-1,2e-3,1.0)"'.format(mname))
  #os.system('define_M2M_observ.py {} -nIFU_h3 -th3  -bIFU -cMaNGA -r"(0,-1,1e-6,1.0)" -f{}data'.format(mname,mname))
  #os.system('define_M2M_observ.py {} -nIFU_h4 -th4  -bIFU -cMaNGA -r"(0,-1,1e-6,1.0)" -f{}data'.format(mname,mname))
  #os.system('define_ghlosvd_group.py {} -mIFU_vel -nIFU_losvd  -sIFU_disp -g"(IFU_h3,IFU_h4)"'.format(mname))
  #os.system('define_ghlosvd_group.py {}'.format(mname))

  #os.system(''.format(mname,))
  #os.system(''.format(mname,))
