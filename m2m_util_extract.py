#!/usr/bin/env python
import numpy as np
import glob
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import re
import util_config as uc

class extract:
  '''
  extract the model observeables, chi2, particles for further use
  '''
  def __init__(self,rname,mname,Particles=False):
    #initialise the class

    #check if the necessary file/folder exist
    if not os.path.isdir(rname):
      print 'Error - run folder {0} is not available'.format(rname)
      sys.exit()

    cfg_name = '{}/{}.cfg'.format(rname,mname) 

    if not os.path.exists(cfg_name):
      print 'Error - .cfg file {0} is not available'.format(cfg_name)
      sys.exit()

    self.obs_folder = '{}/observables'.format(rname)
    self.part_folder = '{}/particles'.format(rname)
    

    if not os.path.isdir(self.obs_folder):
      print 'Error - observale folder {0} is not available'.format(self.obs_folder)
      sys.exit()

    if not os.path.isdir(self.part_folder):
      print 'Error - particle folder {0} is not available'.format(self.part_folder)
      sys.exit()

    # restore .cfg file
    constraints_section = 'sec:Constraints'
    model_name, config = uc.get_config(cfg_name, mname)
    self.model_name = model_name
    self.config  = config
    lum_section = config.get(constraints_section, 'lum_constraints')
    kin_section = config.get(constraints_section, 'kin_constraints')
    spec_section = config.get(constraints_section, 'spectral_lines')
    top_level_list = [lum_section, kin_section, spec_section]
    # restore observables
    obs_list = [] 
    for section in top_level_list:
      obs = uc.get_section_list(0, config, section, 'num_obs', 'obsx')
      for obsname in obs:
        obs_list.append(obsname.split(':')[1])
    self.obs_list = obs_list
    self.obs_dict = {}
    for obs in obs_list:
      obs_data = self.restore_obs(obs)
      self.obs_dict[obs] = obs_data
    # restore particles if requared
    if Particles:
      part_filename = '{}/coordinates'.format(self.part_folder)
      data = np.genfromtxt(part_filename,dtype=['f8','f8','f8','f8','f8','f8','f8','i8'],skip_header=1,\
               names = ['x','y','z','vx','vy','vz','weight','inuse'])
      self.part_data = {}
      self.part_data['x'] = data['x']
      self.part_data['y'] = data['y']
      self.part_data['z'] = data['z']
      self.part_data['vx'] = data['vx']
      self.part_data['vy'] = data['vy']
      self.part_data['vz'] = data['vz']
      self.part_data['weight'] = data['weight']
      self.part_data['inuse'] = data['inuse']


    # restore lambda values
    temp = glob.glob('{}/std*'.format(rname))
    if len(temp)!=1:
      print 'Error - number of std output file must equal to 1'
      sys.exit()
    stdout_filename = temp[0]
    self.lambda_value = {}
    with open(stdout_filename,'r') as ff:
      while True:
        line = ff.readline()
        if line == '':
          print 'Error - end of std file, no lamada value'
          sys.exit()

        if re.search('Revised lambda parameters,',line) is not None:
          break
      for obs in obs_list:
        pat = re.compile('{} += +([+-]?\d+.\d*[eE][-+]?\d+)'.format(obs))
        lambda_value = float(pat.search(line).group(1))
        self.lambda_value[obs] = lambda_value

    # restore chi2 arrays
    chi2_filename = '{}/log_chi2'.format(rname)
    with open(chi2_filename,'r') as ff:
      chi2_file = ff.readlines()
    num_step = len(chi2_file)
    time = np.zeros(num_step)
    for i in range(num_step):
      time[i] = float(chi2_file[i].split(',')[0])
    self.chi2 = {'t': time}
    for obs in obs_list:
      value = np.zeros(num_step)
      pat = re.compile('{} += +([+-]?\d+.\d*[eE][-+]?\d+)'.format(obs))
      for i in range(num_step):
        value[i] = float(pat.search(chi2_file[i]).group(1))
      self.chi2['{}_array'.format(obs)] = value
      if time.max() < 10.0:
        single_value = value[-1]
      else:
        iii = time > time.max() - 5.0
        linefit=np.polyfit(time[iii],value[iii],1)
        yy = np.poly1d(linefit)
        single_value = yy(time.max())
        if abs(single_value - value[-1])/value[-1] > 0.25:
          print 'Warning - final chi2 value and linefit chi2 value do not agree well, chi2: {:.3f}  fit: {:.3f}'.format(single_value,value[-1])
          single_value = value[-1]
      self.chi2[obs] = single_value
      #plt.plot(time[iii],yy(time[iii]),'r',lw=3)
      #plt.plot(time,value)
      #plt.show()

  def total_chi2(self,obs_list=None):
    if obs_list is None:
      obs_list = self.obs_list
    chi2 = 0.0
    for obs in obs_list:
      lambda_value = self.lambda_value[obs]
      single_chi2 = self.chi2[obs]
      nbins_inuse = self.obs_dict[obs]['nbins_inuse']
      chi2 += lambda_value * single_chi2 * nbins_inuse
    self.total_chi2 = chi2
    return chi2
    # calcualte total chi2 given obs_list
    pass


  def restore_obs(self,obs):
    obs_data = {}
    obs_filename = '{}/{}mfile1'.format(self.obs_folder,obs)
    data = np.genfromtxt(obs_filename,dtype=['i8','f8','f8','f8','f8','f8','f8','f8'],\
           names = ['bin_id','dvalue','derror','inuse','mvalue','delta','xbin','ybin'])
    obs_data['obs_name'] = obs
    obs_data['bin_id'] = data['bin_id']
    obs_data['data'] = data['dvalue']
    obs_data['error'] = data['derror']
    obs_data['inuse'] = data['inuse'].astype(int)
    obs_data['model'] = data['mvalue']
    obs_data['smodel'] = data['dvalue'] + data['derror'] * data['delta']
    obs_data['xbin'] = data['xbin']
    obs_data['ybin'] = data['ybin']
    obs_data['nbins_inuse'] = (obs_data['inuse'] == 1).sum()
    obs_data['nbins'] = len(obs_data['inuse'])
    return obs_data


if __name__ == '__main__':
  lhy = extract('rst_1057_8588-6102','1057_8588-6102')
  obs1 = lhy.obs_dict[lhy.obs_list[0]]
  print lhy.obs_list
  print lhy.lambda_value
  lhy.total_chi2()
  print lhy.total_chi2
  #plt.plot(lhy.chi2['t'],lhy.chi2['ld_data'])
  #plt.plot(obs1['xbin'],obs1['ybin'],'.')
  #plt.show()
