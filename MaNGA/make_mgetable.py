#!/usr/bin/env python
import numpy as np
import os
from manga_util import write_mge
import util_config as uc
from optparse import OptionParser
from read_rc import read_rc
class make_paras:
  def __init__(self,gname,mname=None,information=False):
    if mname is None:
      mname = gname
    #read in configuration file
    self.xmodel_name, self.xconfig = uc.get_config('{}/{}.cfg'.format(gname, mname), mname)    
    mge_folder = self.xconfig.get('sec:ics_Luminous_matter', 'interp_folder')
    with open('{}/create_MGE.sh'.format(gname),'w') as ff:
      print >>ff, 'mkdir -p {}/{}'.format(gname,mge_folder)
      print >>ff, 'ext3dmge  -o{}/{}  -g{}/auxiliary_data/mge_params_{}  >{}/{}/stdMGE'.format(gname,\
                  mge_folder,gname,gname,mname,mge_folder)
    os.system('chmod +x {}/create_MGE.sh'.format(gname))
    
if __name__ == '__main__':
  #parser = OptionParser()
  #parser.add_option('-n', action='store',type='string' ,dest='gname',default=None,help='modelname')
  #(options, args) = parser.parse_args()
  #gname = options.gname
  #lhy = make_paras(gname,information=False)
  flist = read_rc('glist')
  for gname in flist:
    lhy = make_paras(gname,mname=gname,information=True)
