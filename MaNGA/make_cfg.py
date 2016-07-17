#!/usr/bin/env python
import numpy as np
import os
from optparse import OptionParser
cdir = os.getcwd()

duration = 100
size = 5.0
n_part = 100000
integration_time_step = 0.01
epsilon = 0.005
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
  rst = np.load('auxiliary_data/JAM_pars.npy')
  ml = rst[1]
  inc_deg = rst[2]
  dist = rst[6]
  Re_arcsec = rst[7]
  Re_kpc = Re_arcsec * dist * np.pi / 0.648 * 1e-3
  revised_gravconstant = gravconstant * msun * tenmegayear * tenmegayear / (pc_km * pc_km * pc_km * 1e18 * Re_kpc * Re_kpc * Re_kpc)
  # model parameters
  os.system('update_model.py {} -t{} -s{} -i{} -uyes'.format(mname,duration,size,inc_deg))
  os.system('define_M2M_particles.py {} -n{} -pfrom_elz -vfrom_elz -fP{}_{:.3f}'.format(mname,n_part,mname,ml))
  os.system('update_lm.py {} -m{:.3f} -nMGE -fMGE{}'.format(mname, ml, mname))
  os.system('update_grav_constant.py {} -g{:e}'.format(mname, revised_gravconstant[0]))
  os.system('update_orbit_int.py {} -t{:f}'.format(mname, integration_time_step))
  os.system('update_weight_adapt.py {} -e{:f} -p"(0,-1)"'.format(mname,epsilon))
  # bin scheme
  os.system('define_M2M_scheme.py {} -nld -taxisym -i"(16,32)" -s"(3.0)" -f{}bins'.format(mname,mname))
  os.system('define_M2M_scheme.py {} -nsb -tpolar -i"(16,16)" -s"(3.0)" -f{}bins'.format(mname,mname))
  os.system('define_M2M_scheme.py {} -nIFU -ttree -f{}bins'.format(mname,mname))

  os.system('define_M2M_observ.py {} -nld_data -tld -bld -cMaNGA -r"(0,-1,1e-6,1.0)" -f{}data'.format(mname,mname,mname))
  os.system('define_M2M_observ.py {} -nsb_data -tsb -bsb -cMaNGA -r"(0,-1,1e-6,1.0)" -f{}data'.format(mname,mname,mname))
  os.system('define_M2M_observ.py {} -nIFU_vel  -tlosvelocity -bIFU -cMaNGA -r"(0,-1,1e-6,1.0)" -f{}data'.format(mname,mname))
  os.system('define_M2M_observ.py {} -nIFU_disp -tlosveldisp  -bIFU -cMaNGA -r"(0,-1,1e-6,1.0)" -f{}data'.format(mname,mname))
  #os.system('define_M2M_observ.py {} -nIFU_h3 -tlosveldisp  -bIFU -cMaNGA -r"(0,-1,1e-6,1.0)" -f{}data'.format(mname,mname))
  #os.system(''.format(mname,))
  #os.system(''.format(mname,))
