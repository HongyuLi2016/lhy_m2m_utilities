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
import cProfile
import pstats
date=strftime('%Y-%m-%d %X',localtime())

#check if the R vaule inside the energy lz sruface
def inside_zvc(R, z, energy, lz, Phi=None):
  return energy >= Phi(R, 0.0, z) + 0.5 * lz * lz / (R * R)

#the function return the angular momentum given energy and R
def max_lz_fn(R,energy,Phi=None):
  patr = Phi(  R, 0.0 , 0.0  )
  if patr > energy:
    return 0.0
  else:
    return -np.sqrt(2.0 * R * R * ( energy - patr ) )

#find out the maximum lz given R and energy
def find_max_lz(energy, Renergy, Phi=None):
  res = minimize(max_lz_fn, 0.5*Renergy, method = 'SLSQP', bounds = [[0.0, Renergy]], args=(energy,Phi) ,options={'maxiter':50})
  if not res.success:
    print 'Warning - minimize in find max lz failed for (energy, Renergy) = (%.4f, %.4f)'%(energy, Renergy)
  return -res.fun[0], res.x[0]

#the function used to find out the z value for a given lz, energy and R
def zvc_fn(z, energy, lz, R, Phi=None):    
  return Phi(R , 0.0, z) + 0.5 * lz * lz /  (R * R) - energy
   
#the function used to find out the z value for a given lz, energy and R
def find_zvc_z(energy, lz, R, Phi=None, size = None):
  res = brentq(zvc_fn, 0.0, size, args=(energy, lz, R, Phi), xtol=1e-8, rtol=1e-8, maxiter=50)
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
    print 'Mass-to-light ratio used: %.2f'%self.mass_to_light

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

  def random_table(self,index): 
    tablesize=self.num_particles * 5
    if self.initialize:
      binsize_radial = tablesize / self.LOGB 
      binsize_circ = tablesize / self.NCIRC
      tablesize = binsize_radial * self.LOGB
      radial_bin = np.zeros(self.LOGB +1)
      circ_bin = np.zeros(self.NCIRC +1)
      radial_bin[1:] = (self.size + 1.0)**((np.arange(self.LOGB,dtype=float)+1)/self.LOGB) - 1.0
      circ_bin[1:] = (self.cvalue_limit + 1.0)**((np.arange(self.NCIRC,dtype=float)+1)/self.LOGB) - 1.0
      radial_bin[-1] = self.size
      circ_bin[-1] = self.cvalue_limit
      Renergy = []
      cvalue = []
      R = []

      for i in range(self.LOGB):
        Renergy.append(uniform.rvs(loc=radial_bin[i],scale=radial_bin[i+1]-radial_bin[i],size=binsize_radial))
        R.append(uniform.rvs(loc=radial_bin[i],scale=radial_bin[i+1]-radial_bin[i],size=binsize_radial))
      for i in range(self.NCIRC):
        cvalue.append(uniform.rvs(loc=circ_bin[i],scale=circ_bin[i+1]-circ_bin[i],size=binsize_circ))      

      #print 'initialize'      
      self.rtable=np.zeros([tablesize,3])
      self.rtable[:,0]=np.array(Renergy).ravel()[np.random.permutation(tablesize)]
      self.rtable[:,1]=-np.array(cvalue).ravel()[np.random.permutation(tablesize)]
      self.rtable[:,2]=np.array(R).ravel()[np.random.permutation(tablesize)]
      self.table_index=np.array([0,0,0],dtype=int)
      self.initialize = False
    rst = self.rtable[self.table_index[index],index]
    self.table_index[index]+=1
    if (self.table_index > tablesize - 5).sum()>0:
      self.initialize=True
    return rst


  def Elz(self,use_logz=True,LOGB=100,NCIRC=100,cvalue_limit=1.0,num_particles=None,\
          plot=True):
    if num_particles is None:
      num_particles = self.num_particles
    self.LOGB = LOGB
    self.NCIRC = NCIRC
    self.cvalue_limit = cvalue_limit
    origin=np.asarray([0.0,0.0,0.0]) 
    limit=np.asarray([self.size,0.0,0.0])
    energy_limit  = self.mge.phi(limit[0],limit[1],limit[2])
    energy_origin = self.mge.phi(origin[0],origin[1],origin[2])
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
    circ_limit=cvalue_limit
    loc_circ=np.log10(circ_limit/NCIRC)
    scale_circ=np.log10(circ_limit)-np.log10(circ_limit/NCIRC)
    #initialize random table, index 0: Renergy 1: cvalue 2: R
    self.boundary=np.array([[loc_logb,scale_logb],[loc_circ,scale_circ],[0.0,self.size]])
    self.initialize=True

    # sample energy and lz
    sign_z=-1.0
    for i in range(num_particles):
      while True:
        # sample energy
        while True:
          #Renergy[i]=10**uniform.rvs(loc=loc_logb,scale=scale_logb,size=1)
          Renergy[i]=self.random_table(0)
          energy_pos=np.array([Renergy[i],0,0])
          energy[i]=self.mge.phi(energy_pos[0],energy_pos[1],energy_pos[2])
          if energy[i]<1.02*energy_limit:
            break
      
        # sample cvalue
        while True:
          #cvalue = -10**uniform.rvs(loc = loc_circ, scale = scale_circ, size = 1)
          cvalue = self.random_table(1)
          if cvalue>-cvalue_limit:
            break
        # find maximum lz for an energy
        max_lz, max_lz_radius = find_max_lz(energy[i],Renergy[i],Phi=self.mge.phi)
        
        # sample lz
        #sign_lz=np.random.choice([-1.0,1.0]) #set sign for lz???
        if use_logz:
          lz[i] = (cvalue + 1.0) * max_lz # Question: why lz only have positive value?
        else:
          lz=uniform.rvs(loc=-max_lz,scale=2*max_lz,size=num_particles)  
        if abs(lz[i])>0.0001 and abs(lz[i])<max_lz-0.0001:
          break
      
      #sample R and z
      
      stop_z=False

      while True:

        stop_R = False

        while True:
          #R[i] =  uniform.rvs( loc= 0.0, scale = self.size, size = 1 )
          R[i] = self.random_table(2)
          stop_R =  inside_zvc(R[i], 0.0, energy[i], lz[i], Phi = self.mge.phi )
          if stop_R:
            break

        z[i] = sign_z * find_zvc_z(energy[i], lz[i], R[i], Phi = self.mge.phi, size= self.size)
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

      fig=plt.figure()
      ax=fig.add_subplot(1,1,1)
      ax.hist(energy,bins=300)
      ax.set_xlabel('energy')
      ax.set_ylabel('Number of particles')
      fig.savefig('%s/%s/elzEnergy.png'%(self.folder, self.particle_folder))

      fig=plt.figure()
      ax=fig.add_subplot(1,1,1)
      ax.hist(lz,bins=300)
      ax.set_xlabel('Lz')
      ax.set_ylabel('Number of particles')
      fig.savefig('%s/%s/elzLz.png'%(self.folder, self.particle_folder))

      fig=plt.figure()
      ax=fig.add_subplot(1,1,1)
      ax.hist(Renergy,bins=300)
      ax.set_xlabel('Renergy')
      ax.set_ylabel('Number of particles')
      fig.savefig('%s/%s/elzRenergy.png'%(self.folder, self.particle_folder))

      fig=plt.figure()
      ax=fig.add_subplot(1,1,1)
      ax.hist(R,bins=300)
      ax.set_xlabel('R')
      ax.set_ylabel('Number of particles')
      fig.savefig('%s/%s/elzR.png'%(self.folder, self.particle_folder))

      fig=plt.figure()
      ax=fig.add_subplot(1,1,1)
      ax.hist(z,bins=300)
      ax.set_xlabel('z')
      ax.set_ylabel('Number of particles')
      fig.savefig('%s/%s/elzZ.png'%(self.folder, self.particle_folder))

      

    self.lz=lz
    self.energy=energy
    self.Renergy=Renergy
    self.R=R
    self.z=z

    return xx, vv


  def output_ics_file(self, xx, vv, weight, fname = 'coordinates'):
    if len(weight) != self.num_particles:
      print 'Error - input array size does not match the cfg file in output_ics_file!'
      exit(1)
    with open('%s/%s/%s'%(self.folder, self.particle_folder, fname),'w') as ff:
      print >> ff, '%d'%self.num_particles
      for i in range(len(weight)):
        print >> ff, '%+e %+e %+e %+e %+e %+e %+e'%(xx[i,0],xx[i,1],xx[i,2],vv[i,0],vv[i,1],vv[i,2],weight[i])  
    
    pot_energy = self.Phi(xx)
    kin_energy = 0.5 * (vv[:,0]**2 + vv[:,1]**2 + vv[:,2]**2)
    particle_energy = pot_energy + kin_energy
    relative_error = np.abs((particle_energy - self.energy) / self.energy )
    bad_energy = relative_error > 1e-6
    r2 = xx[:,0]**2 + xx[:,1]**2 +xx[:,2]**2
    bad_position = r2 > self.size**2
    print 'number of bad energy: %d'%(bad_energy.sum())
    print 'number of bad position: %d'%(bad_position.sum())
    print 'Total time for creating ICS: %.1f'%(time()-self.start_time)
    np.save('%s/%s/particles.npy'%(self.folder, self.particle_folder),[xx,vv])
    np.save('%s/%s/others.npy'%(self.folder, self.particle_folder),[self.lz,self.energy,self.Renergy,self.R,self.z])


def main():
  parser = OptionParser()
  parser.add_option('-f', action='store',type='string' ,dest='folder',default=None,help='folder name')
  (options, args) = parser.parse_args()
  if len(args)!=1:
    print 'Error - model name must be provided!'
  model_name=args[0]
  folder=options.folder
  lhy = create_ics(model_name, folder = folder)
  xx,vv = lhy.Elz()
  lhy.output_ics_file(xx,vv,lhy.weight)
    
if __name__=='__main__':
  main()
  #cProfile.run('main()','profile.log')
  #p=pstats.Stats('profile.log')
  #p.sort_stats('time').print_stats(20)
