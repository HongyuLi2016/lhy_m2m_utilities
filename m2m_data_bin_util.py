#!/usr/bin/env python
'''
v0.0 Create constraint data and bin scheme
'''
version='v0.0'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
import util_config as uc
from optparse import OptionParser
import os
import sys
import util_runtime as rt
from scipy.stats import uniform
from scipy.optimize import minimize,brentq
from time import time,localtime,strftime
from scipy.spatial import ConvexHull
import pyfits
from m2m_manga_utils import symmetrize_velfield
import cProfile
import pstats

date=strftime('%Y-%m-%d %X',localtime())

def mge(sol,x,y):
  rst = np.zeros_like(x)
  rho0 = sol[:,0] / ( 2.0 * np.pi * sol[:,1]**2 *sol[:,2])
  sigma = sol[:,1]
  q_obs = sol[:,2]
  for i in range(sol.shape[0]):
    rst += rho0[i] * np.exp(-(x**2 + y**2 / q_obs[i]**2) / ( 2.0 * sigma[i]**2))
  return rst

def mge3d(sol,R,z,inc_deg):
  cosinc = np.cos(inc_deg/180.0*np.pi)
  sininc = np.sin(inc_deg/180.0*np.pi)
  rst = np.zeros_like(R)
  sigma = sol[:,1]
  q_int = (sol[:,2]**2-cosinc**2)**0.5/sininc
  #rho0 = sol[:,0]*2.0*np.pi*(sigma)**2*sol[:,2]/((sigma)**3*(2*np.pi)**1.5*q_int)
  rho0 = sol[:,0]/((sigma)**3*(2*np.pi)**1.5*q_int)
  for i in range(sol.shape[0]):
    rst += rho0[i] * np.exp( -(R**2 + z**2/q_int[i]**2) / (2.0 * sigma[i]**2))
  return rst


class create:
  '''
  Create data files and bin files for a pyM2M code
  '''
  def __init__(self,model_name, folder=None ):
    '''
    model_name: the name of the model, must be the same as in the .cfg file
    folder: the folder contain the .cfg files and other data files
    '''
    if folder is None:
      folder=model_name
    input_cfg=model_name+'.cfg'
    self.model_name=model_name
    self.folder=folder
    self.obs_folder = '%sdata'%model_name
    self.bin_folder = '%sbins'%model_name
    os.system('mkdir -p %s/%s'%(self.folder, self.obs_folder))
    os.system('mkdir -p %s/%s'%(self.folder, self.bin_folder))
    self.start_time=time()
    if not os.path.exists(folder):
      print 'Error - folder {0} does not exist'.format(folder)
      sys.exit()
    if not os.path.exists(folder+'/'+input_cfg):
      print 'Error - file {0} does not exist'.format(input_cfg)
      sys.exit()
    # restore .cfg file
    self.xmodel_name, self.xconfig = uc.get_config(folder+'/'+input_cfg, model_name)
    self.Constraints_section = self.xconfig.get(self.xmodel_name, 'Constraints')

  def IFU(self,xbin,ybin,vel=None,vel_err=None,disp=None,disp_err=None,\
         h3=None, h3_err=None, h4=None, h4_err=None, good=None, Re=None,\
         dist=None, rebin_x=None, rebin_y=None, n_part=None, plot=False,\
         symmetrize=True,vertexStep=1):
    if good is None:
      good = np.ones(len(xbin), dtype=int)
    goodbins = good.astype(bool)

    if Re is None:
      print 'Error - Re must be proviede in create_data_bin.IFU'
      exit(1)

    if dist is None:
      print 'Error - distance must be proviede in create_data_bin.IFU'
      exit(1)

    if n_part is None:
      n_part = 100000

    #unit conversion
    distance_mpc = dist
    #as2Re =
    R_e_as = Re
    as_pc = np.pi / 0.648
    Re_kpc = R_e_as * distance_mpc * as_pc * 1e-3
    pc_km = 3.0856775975e13
    tenmegayear = 3600.0 * 24.0 * 365 * 1e7

    xbin /= R_e_as
    ybin /= R_e_as
    if rebin_x is not None:
      rebin_x /= R_e_as
      rebin_y /= R_e_as


    #create IFU bin hull
    if rebin_x is not None:
      hull = ConvexHull(np.array([rebin_x,rebin_y]).T)
      x_hull = rebin_x[hull.vertices][::-vertexStep] # vertexStep != 1 means do not use all the vertices
      y_hull = rebin_y[hull.vertices][::-vertexStep] # must be clockwise!
    else:
      print 'Error - rebined data position should be provided!'
      exit(0)
    R = (x_hull**2 + y_hull**2)**0.5
    Rmax = np.mean(R.max())
    r =  R.copy() * 0.0
    for i in range(len(r)):
      position = np.array([[x_hull[i-1], x_hull[i]], [y_hull[i-1], y_hull[i]]])
      vector1 = np.array([position[0,1], position[1,1]])
      vector2 = np.array([position[0,1]-position[0,0], position[1,1]-position[1,0]])
      project_length = abs( np.dot(vector1, vector2) / np.sqrt(np.dot(vector2, vector2)) )
      r[i] = (np.dot(vector1,vector1) -project_length**2 )**0.5
    Rmin = np.mean(r.min())
    Rect_x_min = np.mean(x_hull.min())
    Rect_x_max = np.mean(x_hull.max())
    Rect_y_min = np.mean(y_hull.min())
    Rect_y_max = np.mean(y_hull.max())
    with open('%s/%s/IFU_hull'%(self.folder,self.bin_folder),'w') as ff:
      print >>ff, '{0:d}  {1:+e}  {2:+e}  {3:+e}  {4:+e}  {5:+e}  {6:+e}'.format(len(x_hull)+1, Rmin, Rmax,\
                   Rect_x_min, Rect_x_max, Rect_y_min, Rect_y_max)
      for i in range(len(x_hull)):
        print >>ff, '{0:+e}  {1:+e}'.format(x_hull[i], y_hull[i])
      print >>ff, '{0:+e}  {1:+e}'.format(x_hull[0], y_hull[0])
    if plot:
      fig = plt.figure(figsize=(6,6))
      ax = fig.add_subplot(1,1,1)
      ax.plot(xbin,ybin,'o',markersize = 5.0)
      ax.plot(rebin_x,rebin_y,'.r',markersize = 1.0)
      for simplex in hull.simplices:
        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'k-')
      circle=Circle(xy=(0.0,0.0),fc='none',radius=Rmin,ec='c',zorder=1,lw=2.)
      ax.add_artist(circle)
      circle=Circle(xy=(0.0,0.0),fc='none',radius=Rmax,ec='c',zorder=1,lw=2.)
      ax.add_artist(circle)
      squre=Rectangle(xy=(Rect_x_min,Rect_y_min),fc='none',width=Rect_x_max-Rect_x_min, height=Rect_y_max-\
            Rect_y_min,ec='r',zorder=1,lw=2.)
      ax.add_artist(squre)
      ax.set_aspect(1)
      ax.set_aspect('equal',adjustable='box',anchor='C')
      xlim=[Rect_x_min,Rect_x_max]
      ylim=[Rect_y_min,Rect_y_max]
      lim=[min(xlim[0],ylim[0]),max(xlim[1],ylim[1])]
      ax.set_xlim(lim)
      ax.set_ylim(lim)
      ax.set_xlabel('x/Re')
      ax.set_ylabel('y/Re')
      fig.savefig('%s/%s/IFU_hull.png'%(self.folder,self.bin_folder),dpi=300)
      plt.close(fig)
    # calculate bin area
    x_part = uniform.rvs(loc=Rect_x_min,scale=Rect_x_max-Rect_x_min,size=n_part)
    y_part = uniform.rvs(loc=Rect_y_min,scale=Rect_y_max-Rect_y_min,size=n_part)
    R = (x_part**2 + y_part**2)**0.5
    # find out the particles inside the convex hull
    i_in_convex = np.zeros(n_part, dtype = bool)
    i_in_Rmax = R < Rmax
    i_in_Rmin = R < Rmin
    i_between = (~i_in_Rmin)
    i_between_in = i_between.copy()
    for i in range(n_part):
      if i_between[i]:
        for j in range(len(x_hull) - 1):
          x1 = x_hull[j+1]
          y1 = y_hull[j+1]
          x0 = x_hull[j]
          y0 = y_hull[j]
          dx = x1 - x0
          dy = y1 - y0
          if (y_part[i] - y0) * dx - (x_part[i] - x0) * dy > 0:
            i_between_in[i] = False
            continue
    i_in_convex = i_in_Rmin +  i_between_in
    hull_volume = i_in_convex.sum()/float(n_part) * (Rect_x_max-Rect_x_min)*(Rect_y_max-Rect_y_min)
    n_part = i_in_convex.sum()
    x_part = x_part[i_in_convex]
    y_part = y_part[i_in_convex]
    dist2 = np.zeros([n_part,len(xbin)])

    for i in range(len(xbin)):
      dist2[:,i] = (x_part - xbin[i])**2 + (y_part - ybin[i])**2
    bin_index = np.zeros(n_part,dtype=int)
    for i in range(n_part):
      bin_index[i] = np.where(dist2[i,:]== np.min(dist2[i,:]))[0][0]

    area = np.zeros(len(xbin))

    if plot:
      fig = plt.figure(figsize=(6,6))
      ax = fig.add_subplot(1,1,1)
      for simplex in hull.simplices:
        ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], 'k-',zorder=0)
      ax.plot(xbin,ybin,'or',markersize = 3.0,markeredgecolor='none',alpha=0.6,zorder=2)
      #ax.plot(x_part,y_part,'.k',markersize=1.0)
      color = ['r','k','b','g','y','c','m','w']
    for i in range(len(xbin)):
      iii = bin_index == i
      area[i] = iii.sum()/float(n_part) * hull_volume
      if plot:
        color_index=i%8
        ax.plot(x_part[iii],y_part[iii],'.',color=color[color_index],markersize=1.0,zorder=0)
    if plot:
      ax.set_aspect(1)
      ax.set_xlim(lim)
      ax.set_ylim(lim)
      ax.set_xlabel('x/Re')
      ax.set_ylabel('y/Re')
      fig.savefig('%s/%s/IFU_bins.png'%(self.folder,self.bin_folder),dpi=300)
      plt.close(fig)

    # create IFU bin file
    bin_output_name='IFU_bfile1'
    with open('%s/%s/%s'%(self.folder,self.bin_folder,bin_output_name),'w') as ff:
      for i in range(len(xbin)):
        print >>ff, '{0:4d}  {1:+e}  {2:+e}  {3:+e}  {4}'.format(i, xbin[i], ybin[i], area[i] ,good[i])

    # create data files
    if vel is not None:
      vel *= tenmegayear * 1e-3 / pc_km / Re_kpc
      vel_err *= tenmegayear * 1e-3 / pc_km / Re_kpc
      if symmetrize:
        vel_new = symmetrize_velfield(xbin[goodbins],ybin[goodbins],vel[goodbins],sym=1)
        vel_err_new = symmetrize_velfield(xbin[goodbins],ybin[goodbins],vel_err[goodbins],sym=2)
        vel[goodbins] = vel_new
        vel_err[goodbins] = vel_err_new
      vel_output_name='IFU_veldfile1'
      with open('%s/%s/%s'%(self.folder,self.obs_folder,vel_output_name),'w') as ff:
        for i in range(len(xbin)):
          print >>ff, '{0:4d}  {1:+e}  {2:+e}  {3}'.format(i, vel[i], vel_err[i], good[i])

    if disp is not None:
      disp *= tenmegayear * 1e-3 / pc_km / Re_kpc
      disp_err *= tenmegayear * 1e-3 / pc_km / Re_kpc
      if symmetrize:
        disp_new = symmetrize_velfield(xbin[goodbins],ybin[goodbins],disp[goodbins],sym=2)
        disp_err_new = symmetrize_velfield(xbin[goodbins],ybin[goodbins],disp_err[goodbins],sym=2)
        disp[goodbins] = disp_new
        disp_err[goodbins] = disp_err_new
      disp_output_name='IFU_dispdfile1'
      with open('%s/%s/%s'%(self.folder,self.obs_folder,disp_output_name),'w') as ff:
        for i in range(len(xbin)):
          print >>ff, '{0:4d}  {1:+e}  {2:+e}  {3}'.format(i, disp[i]**2, 2.0*disp[i]*disp_err[i], good[i])

    if h3 is not None:
      if symmetrize:
        h3_new = symmetrize_velfield(xbin[goodbins],ybin[goodbins],h3[goodbins],sym=1)
        h3_err_new = symmetrize_velfield(xbin[goodbins],ybin[goodbins],h3_err[goodbins],sym=2)
        h3[goodbins] = h3_new
        h3_err[goodbins] = h3_err_new
      h3_output_name = 'IFU_h3dfile1'
      with open('%s/%s/%s'%(self.folder,self.obs_folder,h3_output_name),'w') as ff:
        for i in range(len(xbin)):
          print >>ff, '{0:4d}  {1:+e}  {2:+e}  {3}'.format(i, h3[i], h3_err[i], good[i])

    if h4 is not None:
      if symmetrize:
        h4_new = symmetrize_velfield(xbin[goodbins],ybin[goodbins],h4[goodbins],sym=2)
        h4_err_new = symmetrize_velfield(xbin[goodbins],ybin[goodbins],h4_err[goodbins],sym=2)
        h4[goodbins] = h4_new
        h4_err[goodbins] = h4_err_new
      h4_output_name = 'IFU_h4dfile1'
      with open('%s/%s/%s'%(self.folder,self.obs_folder,h4_output_name),'w') as ff:
        for i in range(len(xbin)):
          print >>ff, '{0:4d}  {1:+e}  {2:+e}  {3}'.format(i, h4[i], h4_err[i], good[i])

  def luminosity_density(self,sol,inc_deg):
    scheme_type      = self.xconfig.get('sec:ld', 'scheme_type')
    radial_interval  = self.xconfig.get('sec:ld', 'radial_interval')
    log_base         = self.xconfig.getfloat('sec:ld', 'log_base') if radial_interval == 'log' else 0.0
    num_intervals    = self.xconfig.get('sec:ld', 'num_intervals')
    scheme_size      = self.xconfig.get('sec:ld', 'size')

    # split up num_intervals and scheme size

    xintervals = uc.separate_values(num_intervals)

    num_coord1 = int(xintervals[0])  # r
    num_coord2 = int(xintervals[1])  # theta

    num_edges_coord1 = num_coord1 + 1
    num_edges_coord2 = num_coord2 + 1

    xsize = uc.separate_values(scheme_size)

    min_coord1 = 0.0
    max_coord1 = float(xsize[0])  # r
    min_coord2 = -0.5 * np.pi
    max_coord2 =  0.5 * np.pi    # theta

    # create binning scheme edge values

    edges_coord2 = np.linspace(min_coord2, max_coord2, num_edges_coord2)

    edges_coord1 = np.zeros(num_edges_coord1)
    if radial_interval == 'Sellwood_2003':
      edges_coord1[0] = 0.0
      for i in xrange(1, num_edges_coord1):
        edges_coord1[i] = pow(max_coord1 + 1.0, float(i)/float(num_coord1)) - 1.0
    elif radial_interval == 'log':
      edges_coord1[0] = 0.0
      pn = 1.0
      for i in xrange(num_coord1, 0, -1):
        edges_coord1[i] = max_coord1 / pn
        pn *= log_base
    else:
      edges_coord1 = np.linspace(0.0, max_coord1, num_edges_coord1)

    #print 'edges_coord1', self.edges_coord1
    #print 'edges_coord2', self.edges_coord2

    # construct the bin R and z values

    num_bins = num_coord1 * num_coord2

    bin_R = np.zeros(num_bins)
    bin_z = np.zeros(num_bins)

    for i in xrange(num_coord1):
      r = 0.5 * (edges_coord1[i] + edges_coord1[i+1])

      for j in xrange(num_coord2):
        theta = 0.5 * (edges_coord2[j] + edges_coord2[j+1])

        bin_index = i * num_coord2 + j
        bin_R[bin_index] = r * np.cos(theta)
        bin_z[bin_index] = r * np.sin(theta)

    # calculate bin volume
    volume = np.zeros(num_bins)
    for i in xrange(num_coord1):
      for j in xrange(num_coord2):
        bin_index = i * num_coord2 + j
        volume[bin_index] = (np.sin(edges_coord2[j+1]) - np.sin(edges_coord2[j])) * \
                      (edges_coord1[i+1]**3 - edges_coord1[i]**3) * 2.0 * np.pi / 3.0

    #calculate bin value
    bin_value = mge3d(sol, bin_R, bin_z, inc_deg)
    good = np.ones_like(bin_value,dtype=int)

    # write data to the data file and bin file
    sb_output_name = 'ld_datadfile1'
    bin_output_name = 'ld_bfile1'
    with open('%s/%s/%s'%(self.folder,self.obs_folder,sb_output_name),'w') as ff:
      for i in range(len(bin_value)):
        print >>ff, '{0:4d}  {1:+e}  {2:+e}  {3}'.format(i, bin_value[i], bin_value[i]*0.1, good[i])
    with open('%s/%s/%s'%(self.folder,self.bin_folder,bin_output_name),'w') as ff:
      for i in range(len(volume)):
        print >>ff, '{0:4d}  {1:+e} {2}'.format(i,volume[i],good[i])

  def surface_brightness(self, sol):
    '''
    input mge parameters should be in unit: Luminosiyt (10^10 M_sun), sigma (R_e), flat
    '''
    scheme_type = self.xconfig.get('sec:sb', 'scheme_type')
    radial_interval = self.xconfig.get('sec:sb', 'radial_interval')
    num_intervals = self.xconfig.get('sec:sb', 'num_intervals')
    scheme_size = self.xconfig.get('sec:sb', 'size')
    log_base = config.getfloat('sec:sb', 'log_base') if radial_interval == 'log' else 0.0
    xintervals = uc.separate_values(num_intervals)
    num_coord1 = int(xintervals[0])
    num_coord2 = int(xintervals[1])
    num_edges_coord1 = num_coord1 + 1
    num_edges_coord2 = num_coord2 + 1
    xsize = uc.separate_values(scheme_size)
    min_coord1 = 0.0
    max_coord1 = float(xsize[0])
    min_coord2 = 0.0
    max_coord2 = 2 * np.pi
    edges_coord2 = np.linspace(0.0, max_coord2, num_edges_coord2)
    edges_coord1 = np.zeros(num_edges_coord1)
    if radial_interval == 'Sellwood_2003':
      edges_coord1[0] = 0.0
      for i in xrange(1, num_edges_coord1):
        edges_coord1[i] = pow(max_coord1 + 1.0, float(i)/float(num_coord1)) - 1.0
    elif radial_interval == 'log':
      edges_coord1[0] = 0.0
      pn = 1.0
      for i in xrange(num_coord1, 0, -1):
        edges_coord1[i] = max_coord1 / pn
        pn *= log_base
    else:
      edges_coord1 = np.linspace(0.0, max_coord1, num_edges_coord1)

    #calculate bin position
    num_bins = num_coord1 * num_coord2
    bin_X = np.zeros(num_bins)
    bin_Y = np.zeros(num_bins)

    for i in xrange(num_coord1):
      R = 0.5 * (edges_coord1[i] + edges_coord1[i+1])
      for j in xrange(num_coord2):
        theta = 0.5 * (edges_coord2[j] + edges_coord2[j+1])
        bin_index = i * num_coord2 + j
        bin_X[bin_index] = R * np.cos(theta)
        bin_Y[bin_index] = R * np.sin(theta)

    #calculate bin area
    area = np.zeros(num_bins)
    for i in xrange(num_coord1):
      for j in xrange(num_coord2):
        bin_index = i * num_coord2 + j
        area[bin_index] = 0.5 * (edges_coord2[j+1] - edges_coord2[j]) * (edges_coord1[i+1]**2 - edges_coord1[i]**2)

    #calculate bin value
    bin_value = mge(sol, bin_X, bin_Y)
    good = np.ones_like(bin_value,dtype=int)
    #x = np.linspace(-10,10,1000)
    #y = np.linspace(-10,10,1000)
    #X,Y = np.meshgrid(x,y)
    #Z = mge(sol, X, Y)
    #plt.imshow(np.log10(Z))
    #plt.show()

    # write data to the data file and bin file
    sb_output_name = 'sb_datadfile1'
    bin_output_name = 'sb_bfile1'
    with open('%s/%s/%s'%(self.folder,self.obs_folder,sb_output_name),'w') as ff:
      for i in range(len(bin_value)):
        print >>ff, '{0:4d}  {1:+e}  {2:+e}  {3}'.format(i, bin_value[i], bin_value[i]*0.1, good[i])
    with open('%s/%s/%s'%(self.folder,self.bin_folder,bin_output_name),'w') as ff:
      for i in range(len(area)):
        print >>ff, '{0:4d}  {1:+e} {2}'.format(i,area[i],good[i])


  def specline(self, line_name, xbin, ybin, value, error,
               good=None, symmetrize=True):
    line_output_name = '%sdfile1' % line_name
    if good is None:
      good = np.ones(len(value), dtype=int)
    goodbins = good ==1
    if symmetrize:
      value_new = symmetrize_velfield(xbin[goodbins], ybin[goodbins],
                                      value[goodbins], sym=2)
      error_new = symmetrize_velfield(xbin[goodbins], ybin[goodbins],
                                      error[goodbins], sym=2)
      value[goodbins] = value_new
      error[goodbins] = error_new
    with open('%s/%s/%s'%(self.folder,self.obs_folder,line_output_name),'w') as ff:
      for i in range(len(value)):
        print >>ff, '{0:4d}  {1:+e}  {2:+e}  {3}'.format(i, value[i], error[i], good[i])

  def discrete_data(self):
    pass


def main():
  parser = OptionParser()
  parser.add_option('-f', action='store',type='string' ,dest='folder',default=None,help='folder name')
  (options, args) = parser.parse_args()
  if len(args)!=1:
    print 'Error - model name must be provided!'
  model_name=args[0]
  folder=options.folder
  lhy = create(model_name, folder = folder)
  data = pyfits.open('/Users/lhy/m2m/manga_1007_8313-12705.fits')[1].data
  data1 =pyfits.open('/Users/lhy/m2m/manga_1007_8313-12705.fits')[2].data
  xbin = data['xbin']
  ybin = data['ybin']
  v0 = data['v0']
  r = (xbin**2 + ybin**2)**0.5
  ii = r<3.0
  vel = v0 - v0[ii].mean()
  vel_err = data['v0_err']
  disp = data['vd']
  disp_err = data['vd_err']
  h3 = data['h3']
  h3_err = data['h3_err']
  h4 = data['h4']
  h4_err = data['h4_err']
  rebin_x = data1['rebin_x']
  rebin_y = data1['rebin_y']

  #lhy.IFU(xbin,ybin,vel=vel,vel_err=vel_err,disp=disp,disp_err=disp_err,h3=h3,h3_err=h3_err,\
  #        h4=h4,h4_err=h4_err,Re=8.0,dist=100.0, rebin_x=rebin_x, rebin_y=rebin_y, plot=True,\
  #        n_part=300000)
  #lhy.specline('hbeta',vel,vel_err)
  sol = np.load('/Users/lhy/m2m/mge.npy')[0]
  inc = 85.0
  #lhy.surface_brightness(sol)
  lhy.luminosity_density(sol,inc)

if __name__=='__main__':
  main()

