#!/usr/bin/env python
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from util_extract import extract
from optparse import OptionParser
import os
import sys

if __name__ == '__main__':
  parser = OptionParser()
  parser.add_option('-g', action='store',type='string' ,dest='gname',default=None,help='galaxy name')
  (options, args) = parser.parse_args()
  gname = options.gname
  if gname is None:
    print 'Error - galaxy name must be provided!'
    sys.exit()
  cfg_list = glob.glob('{}/{}_*.cfg'.format(gname,gname))
  chi2_total = np.zeros(len(cfg_list)) 
  ml = np.zeros(len(cfg_list))
  inc = np.zeros(len(cfg_list))
  chi2_obs = []
  lambda_value = []
  for i in range(len(cfg_list)):
    cfg = cfg_list[i]
    model_name = cfg.split('/')[-1][:-4]
    rst_name = '{}/rst_{}'.format(gname,model_name)  
    lhy = extract(rst_name,model_name)
    chi2_total[i] = lhy.total_chi2()
    chi2_obs.append([lhy.chi2[obs] for obs in lhy.obs_list])
    lambda_value.append([lhy.lambda_value[obs] for obs in lhy.obs_list])
    ml[i] = float(model_name.split('_')[-1])
    inc[i] = float(model_name.split('_')[-2])
  
  chi2_obs = np.array(chi2_obs)
  lambda_value = np.array(lambda_value)
  os.system('mkdir -p {}/grid_rst'.format(gname))
  np.save('{}/grid_rst/grid.npy'.format(gname),[inc,ml])
  np.save('{}/grid_rst/chi2_total.npy'.format(gname),chi2_total)
  for i in range(len(lhy.obs_list)):
    chi2 = chi2_obs[:,i]
    lambda_obs = lambda_value[:,i]
    np.save('{}/grid_rst/chi2_{}.npy'.format(gname,lhy.obs_list[i]),chi2)
    np.save('{}/grid_rst/lambda_{}.npy'.format(gname,lhy.obs_list[i]),lambda_obs)





