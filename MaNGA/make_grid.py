#!/usr/bin/env python
import numpy as np
import os
import util_config as uc
from optparse import OptionParser

def make_grid(*p_list):
  ndim = len(p_list)
  #dim = [len(p_list[i]) for i in range(ndim)]
  #index is a ndim * n_gird array, index[0,:] index[1,:].... is the index for the number in each dimension, i.e. id for each parameter
  #index = np.indices(dim).reshape(ndim,-1)
  #grid = np.zeros_like(index)
  grid = np.array(np.meshgrid(*p_list)).reshape(ndim,-1)
  return grid

class create_grid_cfg:
  def __init__(self,gname,model=None):
    if model is None:
      model = gname
    self.gname = gname
    self.model = model
    if not os.path.exists('%s/%s.cfg'%(self.gname,self.model)):
      print 'Error - %s/%s.cfg does not exist'%(self.gname,self.model)
      exit(1)
    self.cdir = os.getcwd()
    os.chdir('{}/{}'.format(self.cdir,self.gname))

  def create_i_ml(self,inc,ml,MGE=False,ICS=False):
    grid = make_grid(inc,ml)
    print '%d models are going to be created!'%grid.shape[1]
    for i in range(grid.shape[1]):
      nmodel_name = '{}_{:.1f}_{:.2f}'.format(self.model,grid[0,i],grid[1,i])
      os.system('copy_model.py {} {}'.format(self.model,nmodel_name)) 
      os.system('update_model.py {} -i{:.1f}'.format(nmodel_name,grid[0,i]))
      os.system('update_lm.py {} -m{:.3f} -nMGE -fMGE{}'.format(nmodel_name,grid[1,i],nmodel_name))
      os.system('define_M2M_particles.py {} -fP{}'.format(nmodel_name,nmodel_name))

      os.system('define_M2M_scheme.py {} -nld -taxisym -i"(16,32)" -s"(3.0)" -f{}bins'.format(nmodel_name,nmodel_name))
      os.system('define_M2M_scheme.py {} -nsb -tpolar -i"(16,16)" -s"(3.0)" -f{}bins'.format(nmodel_name,nmodel_name))
      os.system('define_M2M_scheme.py {} -nIFU -ttree -f{}bins'.format(nmodel_name,nmodel_name))
    
      os.system('define_M2M_observ.py {} -nld_data -tld -bld -cMaNGA -r"(0,-1,1e-6,1.0)" -f{}data'.format(nmodel_name,nmodel_name))
      os.system('define_M2M_observ.py {} -nsb_data -tsb -bsb -cMaNGA -r"(0,-1,1e-6,1.0)" -f{}data'.format(nmodel_name,nmodel_name))
      #os.system('define_M2M_observ.py {} -nIFU_vel  -tlosvelocity -bIFU -cMaNGA -r"(0,-1,1e-6,1.0)" -f{}data'.format(nmodel_name,nmodel_name))
      os.system('define_M2M_observ.py {} -nIFU_disp -tlosveldisp  -bIFU -cMaNGA -r"(0,-1,1e-6,1.0)" -f{}data'.format(nmodel_name,nmodel_name))
      

    '''
    # create MGE tables and particle initial conditions
    os.chdir('{}'.format(self.cdir))
    if MGE:
      for i in range(len(inc)):
        print 'Creating MGE table for inclination = {:.1f}'.format(inc[i])
        nmodel_name = '{}_{:.1f}_{:.2f}'.format(self.model,inc[i],ml[0])
        os.system('make_paras.py -g{} -m{}'.format(self.gname,nmodel_name))
        xmodel_name, xconfig = uc.get_config('{}/{}.cfg'.format(self.gname,nmodel_name),nmodel_name)
        MGE_folder = xconfig.get('sec:ics_Luminous_matter','interp_folder')
        os.system('mkdir -p {}/{}'.format(self.gname,MGE_folder))
        os.system('ext3dmge  -o{}/{} -g{}/auxiliary_data/mge_params_{} > {}/{}/create_mge.log'.format(self.gname,\
                   MGE_folder,self.gname,nmodel_name,self.gname,MGE_folder))
    if ICS:  
      for i in range(len(ml)):
        print 'Creating isc for M/L = {:.2f}'.format(ml[i])
        nmodel_name = '{}_{:.1f}_{:.2f}'.format(self.model,inc[0],ml[i])
        os.system('create_ics.py {} -f {}'.format(nmodel_name,self.gname))
    '''
      
    
if __name__=='__main__':
  parser = OptionParser()
  parser.add_option('-g', action='store',type='string' ,dest='gname',default=None,help='galaxy name')
  parser.add_option('-m', action='store',type='string' ,dest='model',default=None,help='model name')
  (options, args) = parser.parse_args()
  inc = np.linspace(45.0,60.0,3)
  ml = np.linspace(3.0,6.0,3)
  lhy = create_grid_cfg(options.gname,model=options.model)
  lhy.create_i_ml(inc,ml)
