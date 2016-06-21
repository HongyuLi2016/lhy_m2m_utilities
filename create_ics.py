#!/usr/bin/env python
'''
v0.0 Create initial conditions for particles, using ELZ method
'''
version='v0.0'
import numpy as np
import matplotlib.pyplot as plt
from fastmge import mge_system
import util_config as uc
from optparse import OptionParser
import os
import sys
import util_runtime as rt
from scipy.stats import uniform
from scipy.optimize import minimize,brentq
from time import time,localtime,strftime

date=strftime('%Y-%m-%d %X',localtime())

#check if the R vaule inside the energy lz sruface
def inside_zvc(R, z, energy, lz, Phi=None):
  return energy >= Phi([R, 0.0, z]) + 0.5 * lz * lz / (R * R)

#the function return the angular momentum given energy and R
def max_lz_fn(R,energy,Phi=None):
  patr = Phi( [ R, 0.0 , 0.0 ] )[0]
  if patr > energy:
    return 0.0
  else:
    return -np.sqrt(2.0 * R * R * ( energy - patr ) )

#find out the maximum lz given R and energy
def find_max_lz(energy, Renergy, Phi=None):
  res = minimize(max_lz_fn, 0.5*Renergy, method = 'SLSQP', bounds = [[0.0, Renergy]], args=(energy,Phi) )
  if not res.success:
    print 'Warning - minimize in find max lz failed for (energy, Renergy) = (%.4f, %.4f)'%(energy, Renergy)
  return -res.fun[0], res.x[0]

#the function used to find out the z value for a given lz, energy and R
def zvc_fn(z, energy, lz, R, Phi=None):    
  return Phi([R , 0.0, z]) + 0.5 * lz * lz /  (R * R) - energy
   
#the function used to find out the z value for a given lz, energy and R
def find_zvc_z(energy, lz, R, Phi=None, size = None):
  res = brentq(zvc_fn, 0.0, size, args=(energy, lz, R, Phi))
  #print res
  return res

class create_ics:
  #initialize class, read in cfg file
  def __init__(self,model_name,folder=None):
    if folder is None:
      folder=model_name
    input_cfg=model_name+'.cfg'
    self.folder=folder
    self.start_time=time()
    if not os.path.exists(folder):
      print 'Error - folder {0} does not exist'.format(folder)
      sys.exit()
    if not os.path.exists(folder+'/'+input_cfg):
      print 'Error - file {0} does not exist'.format(input_cfg)
      sys.exit()
    # restore .cfg file
    self.xmodel_name, self.xconfig = uc.get_config(folder+'/'+input_cfg, model_name)
    # read in particle section
    particle_section = self.xconfig.get(self.xmodel_name, 'particles')
    ics_potential = self.xconfig.get(particle_section, 'ics_potential')
    self.particle_folder = self.xconfig.get(particle_section, 'particle_folder')
    os.system('mkdir -p %s/%s'%(self.folder, self.particle_folder))
    self.num_particles = self.xconfig.getint(particle_section, 'num_particles')
    self.ics_position = self.xconfig.get(particle_section, 'ics_position')
    self.ics_velocity = self.xconfig.get(particle_section, 'ics_velocity')
    self.ics_weight = self.xconfig.get(particle_section, 'ics_weight')
    if self.ics_weight == '1/N':
      self.weight = np.zeros(self.num_particles)+1.0/self.num_particles
    else:
      print 'Error - the method must be 1/N'
    # read in ics_potential section
    self.grav_constant = self.xconfig.getfloat(ics_potential, 'grav_constant')
    ics_luminous_matter = self.xconfig.get(ics_potential, 'ics_luminous_matter')
    # read in ics_luminous_matter section
    if self.xconfig.getboolean(ics_luminous_matter, 'defined'):
      self.mass_to_light = self.xconfig.getfloat(ics_luminous_matter, 'mass_to_light')
      self.name = self.xconfig.get(ics_luminous_matter, 'name')
      self.parameters = self.xconfig.get(ics_luminous_matter, 'parameters')
      self.interp_folder = self.xconfig.get(ics_luminous_matter, 'interp_folder')
    else:
      print 'Error - luminous_matter for initial condition is not defined yet'
      sys.exit()

    # inatialize rt class for fastmge
    rt.global_items['mass2light'] = self.mass_to_light
    rt.global_items['model_name'] = self.xmodel_name
    rt.global_items['config']     = self.xconfig
    model_section = self.xconfig.get(self.xmodel_name, 'model')
    xmodel_duration = self.xconfig.getint(model_section, 'duration')
    rt.global_items['model_duration']     = xmodel_duration
    rt.global_items['maximum_time_steps'] = xmodel_duration
    rt.global_items['model_inclination'] = self.xconfig.getfloat(model_section, 'inclination') * np.pi / 180.0
    rt.global_items['max_radial_extent'] = self.xconfig.getfloat(model_section, 'size')
    self.size=self.xconfig.getfloat(model_section, 'size')
    xhmdtu_duration = self.xconfig.getboolean(model_section, 'hmdtu_duration')
    rt.global_items['hmdtu_duration'] = xhmdtu_duration
    rt.global_items['grav_constant'] = self.grav_constant
    '''
    Question: why id is used in MGEs?
    '''
    rt.global_items['my_id']     = 1
    if self.name=='MGE':
      try:
        self.mge=mge_system('%s/%s'%(self.folder,self.interp_folder))
      except:
        print 'Error - Load MGE interpolation table faild!'

    print 'Creat_ics %s run on %s'%(version, date)
    print '.cfg file name: %s'%input_cfg


  #wrapper for fast mge, given position x, return potential
  def Phi(self,x):
    '''
    calculate the potential at position x (Cartersian coordinate system)
    '''
    x=np.atleast_2d(x)
    rst= np.zeros(x.shape[0])
    for i in range(len(rst)):
      rst[i]=self.mge.phi(x[i,0],x[i,1],x[i,2])
    return rst

  def Elz(self,use_logz=True,LOGB=100,NCIRC=100,cvalue_limit=-1.0,num_particles=None,\
          plot=True):
    if num_particles is None:
      num_particles = self.num_particles
    origin=np.asarray([0.0,0.0,0.0]) 
    limit=np.asarray([self.size,0.0,0.0])
    energy_limit  = self.Phi(limit)[0]
    energy_origin = self.Phi(origin)[0]
    #print energy_limit,energy_origin
    energy_interval = energy_limit - energy_origin
    energy=np.zeros(num_particles)
    Renergy=np.zeros(num_particles)
    lz=np.zeros(num_particles)
    R=np.zeros(num_particles)
    z=np.zeros(num_particles)
    #uniform randam boundary for Renergy and cvalue
    loc_logb=np.log10(self.size/LOGB)
    scale_logb=np.log10(self.size)-np.log10(self.size/LOGB)
    circ_limit=1.0
    loc_circ=np.log10(circ_limit/NCIRC)
    scale_circ=np.log10(circ_limit)-np.log10(circ_limit/NCIRC)
    # sample energy and lz
    sign_z=-1.0
    for i in range(num_particles):
      while True:
        # sample energy
        while True:
          Renergy[i]=10**uniform.rvs(loc=loc_logb,scale=scale_logb,size=1)
          energy_pos=np.array([Renergy[i],0,0])
          energy[i]=lhy.Phi(energy_pos)[0]
          if energy[i]<1.02*energy_limit:
            break
      
        # sample cvalue
        while True:
          cvalue = -10**uniform.rvs(loc = loc_circ, scale = scale_circ, size = 1)
          if cvalue>cvalue_limit:
            break
        # find maximum lz for an energy
        max_lz, max_lz_radius = find_max_lz(energy[i],Renergy[i],Phi=self.Phi)
        
        # sample lz
        #sign_lz=np.random.choice([-1.0,1.0]) #set sign for lz???
        if use_logz:
          lz[i] = (cvalue + 1.0) * max_lz # Question: why lz only have positive value?
        else:
          lz[i]=uniform.rvs(loc=-max_lz,scale=2*max_lz,size=num_particles)  
        if abs(lz[i])>0.0001 and abs(lz[i])<max_lz-0.0001:
          break
      
      #sample R and z
      
      stop_z=False

      while True:

        stop_R = False

        while True:
          R[i] =  uniform.rvs( loc= 0.0, scale = self.size, size = 1 )
          stop_R =  inside_zvc(R[i], 0.0, energy[i], lz[i], Phi = self.Phi )
          if stop_R:
            break

        z[i] = sign_z * find_zvc_z(energy[i], lz[i], R[i], Phi = self.Phi, size= self.size)
        stop_z = R[i] * R[i] + z[i] * z[i]  < self.size * self.size
        if stop_z:
          break
      sign_z *= -1.0 

    #Convert R,z lz to x,v
    vtheta = lz/R
    theta = uniform.rvs(loc=0.0,scale=2.0*np.pi,size=num_particles)
    xx = np.zeros( [num_particles, 3] )
    vv = np.zeros( [num_particles, 3] )

    xx[:,0] = R * np.cos(theta)
    xx[:,1] = R * np.sin(theta)
    xx[:,2] = z

    vv[:,0] = - vtheta * np.sin(theta)
    vv[:,1] = vtheta * np.cos(theta)
    vv[:,2] = 0.0

    #plot some figures if plot is True
    if plot:
      fig=plt.figure()
      ax=fig.add_subplot(1,1,1)
      ax.plot(energy,lz,'.',markersize=2.5,alpha=0.6)
      ax.set_xlabel('Energy')
      ax.set_ylabel('Lz')
      fig.savefig('%s/%s/elzELz.png'%(self.folder, self.particle_folder))

      fig=plt.figure()
      ax=fig.add_subplot(1,1,1)
      ax.plot(R,z,'.',markersize=2.5,alpha=0.6)
      ax.set_xlabel('R')
      ax.set_ylabel('z')
      fig.savefig('%s/%s/elzRz.png'%(self.folder, self.particle_folder))      

    return xx, vv
    #plt.hist(energy,bins=100)
    #plt.hist(cvalue,bins=100)
    #plt.hist(Renergy,bins=100)
    #plt.hist(lz,bins=100)
    #plt.plot(energy,lz,'.')
    #print R,z
    #plt.plot(R,z,'.')
    #plt.show()

  def output_ics_file(self, xx, vv, weight, fname = 'coordinates'):
    if len(weight) != self.num_particles:
      print 'Error - input array size does not match the cfg file in output_ics_file!'
      exit(1)
    with open('%s/%s/%s'%(self.folder, self.particle_folder, fname),'w') as ff:
      print >> ff, '%d'%self.num_particles
      for i in range(len(weight)):
        print >> ff, '%+e %+e %+e %+e %+e %+e %+e'%(xx[i,0],xx[i,1],xx[i,2],vv[i,0],vv[i,1],vv[i,2],weight[i])  
    
    print 'Total time for creating ICS: %.1f'%(time()-self.start_time)
    

if __name__=='__main__':
  parser = OptionParser()
  parser.add_option('-f', action='store',type='string' ,dest='folder',default=None,help='file list')
  (options, args) = parser.parse_args()
  if len(args)!=1:
    print 'Error - model name must be provided!'
  model_name=args[0]
  folder=options.folder
  lhy = create_ics(model_name, folder = folder)
  xx,vv = lhy.Elz()
  lhy.output_ics_file(xx,vv,lhy.weight)
    
