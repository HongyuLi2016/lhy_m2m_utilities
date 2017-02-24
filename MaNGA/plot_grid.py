#!/usr/bin/env python
import numpy as np
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors,rc
from matplotlib.patches import Rectangle,Circle
from optparse import OptionParser
import os
rc('mathtext',fontset='stix')
_cdict = {'red': ((0.000,   0.01,   0.01),
                  (0.170,   0.0,    0.0),
                  (0.336,   0.4,    0.4),
                  (0.414,   0.5,    0.5),
                  (0.463,   0.3,    0.3),
                  (0.502,   0.0,    0.0),
                  (0.541,   0.7,    0.7),
                  (0.590,   1.0,    1.0),
                  (0.668,   1.0,    1.0),
                  (0.834,   1.0,    1.0),
                  (1.000,   0.9,    0.9)),
         'green':((0.000,   0.01,   0.01), 
                  (0.170,   0.0,    0.0),
                  (0.336,   0.85,   0.85),
                  (0.414,   1.0,    1.0),
                  (0.463,   1.0,    1.0),
                  (0.502,   0.9,    0.9),
                  (0.541,   1.0,    1.0),
                  (0.590,   1.0,    1.0),
                  (0.668,   0.85,   0.85),
                  (0.834,   0.0,    0.0),
                  (1.000,   0.9,    0.9)),
          'blue':((0.000,   0.01,   0.01),
                  (0.170,   1.0,    1.0),
                  (0.336,   1.0,    1.0),
                  (0.414,   1.0,    1.0),
                  (0.463,   0.7,    0.7),
                  (0.502,   0.0,    0.0),
                  (0.541,   0.0,    0.0),
                  (0.590,   0.0,    0.0),
                  (0.668,   0.0,    0.0),
                  (0.834,   0.0,    0.0),
                  (1.000,   0.9,    0.9))
          }

sauron = colors.LinearSegmentedColormap('sauron', _cdict)

ticks_font = matplotlib.font_manager.FontProperties(family='times new roman', style='normal', size=8, weight='bold', stretch='normal')
text_font = matplotlib.font_manager.FontProperties(family='times new roman', style='normal', size=10, weight='bold', stretch='normal')
ticks_font1 = matplotlib.font_manager.FontProperties(family='times new roman', style='normal', size=8, weight='bold', stretch='normal')

def plot_lines(x,y,z,ax=None,ylim=None):
  # plot x vs. z, color coded by y
  if ax is None:
    ax = plt.gca()
  y_unique = np.unique(y)
  for i in range(len(y_unique)):
    norm = colors.Normalize(vmin=y_unique.min()*0.9, vmax=y_unique.max()*1.1)
    ii = y == y_unique[i]
    jj = np.argsort(x[ii])
    xx = x[ii][jj] 
    zz = z[ii][jj]
    ax.plot(xx,zz,'.',markersize=5,color=sauron(norm(y_unique[i])))
    ax.plot(xx,zz,'-',lw=1.5,color=sauron(norm(y_unique[i])))
    if ylim is not None:
      ax.set_ylim(ylim)
    for l in ax.get_xticklabels():
      #l.set_rotation(45) 
      l.set_fontproperties(ticks_font)
    for l in ax.get_yticklabels():
      #l.set_rotation(45) 
      l.set_fontproperties(ticks_font)

def plot_2d(x,y,z,ax=None,log=True,c_lim=None):
  if ax is None:
    ax = plt.gca()
  fig = plt.gcf()
  if c_lim is not None:
    z = z.clip(10**c_lim[0],10**c_lim[1])
  if log:
    z = np.log10(z)
  for l in ax.get_xticklabels():
    #l.set_rotation(45) 
    l.set_fontproperties(ticks_font)
  for l in ax.get_yticklabels():
    #l.set_rotation(45) 
    l.set_fontproperties(ticks_font)
  y_unique = np.unique(y)
  x_unique = np.unique(x)
  pos_x = np.zeros_like(x_unique)
  pos_y = np.zeros_like(y_unique)
  size_x = np.zeros_like(x_unique)
  size_y = np.zeros_like(y_unique)
  for i in range(1,len(pos_x)):
    pos_x[i]=(x_unique[i]+x_unique[i-1])*0.5
  pos_x[0] = x_unique[0] - (pos_x[1] - x_unique[0])
  for i in range(0,len(pos_x)-1):
    size_x[i] = pos_x[i+1] - pos_x[i]
  size_x[-1] = (x_unique[-1]-pos_x[-1])*2.0
  #print pos_x
  #print size_x
  for i in range(1,len(pos_y)):
    pos_y[i]=(y_unique[i]+y_unique[i-1])*0.5
  pos_y[0] = y_unique[0] - (pos_y[1] - y_unique[0])
  for i in range(0,len(pos_y)-1):
    size_y[i] = pos_y[i+1] - pos_y[i]
  size_y[-1] = (y_unique[-1]-pos_y[-1])*2.0
  norm = colors.Normalize(vmin=z.min()-abs(z.min()*0.1), vmax=z.max()+abs(z.min()*0.1))
  for i in range(len(pos_x)):
    for j in range(len(pos_y)):
      iii = (x==x_unique[i])*(y==y_unique[j])
      clr = z[iii][0]
      squre=Rectangle(xy=(pos_x[i],pos_y[j]),\
       fc=sauron(norm(clr)),width=size_x[i], height=size_y[j],\
       ec='none',zorder=1,lw=0.)
      ax.add_artist(squre)
      #ax.plot(pos_x[i],pos_y[j],'g.')
  ax.set_xlim([pos_x[0],pos_x[-1]+size_x[-1]])
  ax.set_ylim([pos_y[0],pos_y[-1]+size_y[-1]])
  pos=ax.get_position()
  px=pos.x1
  py=pos.y0
  ph=pos.y1-pos.y0
  axc=fig.add_axes([px,py,0.03,ph])
  matplotlib.colorbar.ColorbarBase(axc,orientation='vertical',norm=norm, cmap=sauron)
  if log:
    axc.set_ylabel('$\mathbf{log \ \chi^2}$',fontsize=8)
  else:
    axc.set_ylabel('$\mathbf{\chi^2}$',fontsize=8)
  for l in axc.get_xticklabels():
    #l.set_rotation(45) 
    l.set_fontproperties(ticks_font1)
  for l in axc.get_yticklabels():
    #l.set_rotation(45) 
    l.set_fontproperties(ticks_font1)

def plot_chi2(x,y,z,xlabel='$\mathbf{M/L}$',ylabel='$\mathbf{inclination}$',zlabel='$\mathbf{\chi^2}$',\
              out_name='chi2.png',rst_folder='./',log=True,c_lim=None,ylim=None):
  fig = plt.figure(figsize = (4,3))
  fig.subplots_adjust(left=0.12, bottom=0.08, right=0.96, top=0.98,wspace=0.2, hspace=0.4)
  ax1 = fig.add_subplot(2,2,1)
  ax1.set_ylabel(zlabel)
  ax1.set_xlabel(xlabel)
  plot_lines(x,y,z,ax=ax1,ylim=ylim)
  ax2 = fig.add_subplot(2,2,2)
  #ax2.set_ylabel(r'$\mathbf{\chi^2}$')
  ax2.set_xlabel(ylabel)
  plot_lines(y,x,z,ax=ax2,ylim=ylim)
  ax3 = fig.add_subplot(2,2,3)
  plot_2d(x,y,z,ax=ax3,log=log,c_lim=c_lim)
  #plt.show()
  fig.savefig('{}/{}'.format(rst_folder,out_name),dpi=300)




if __name__ == '__main__':
  parser = OptionParser()
  parser.add_option('-g', action='store',type='string' ,dest='gname',default=None,help='galaxy name')
  (options, args) = parser.parse_args()
  gname = options.gname
  rst_folder = '{}/grid_rst'.format(gname)
  inc, ml = np.load('{}/grid.npy'.format(rst_folder))
  chi2_list = glob.glob('{}/chi2_*.npy'.format(rst_folder))
  lambda_list = glob.glob('{}/lambda_*.npy'.format(rst_folder))
  for chi2_path in chi2_list:
    out_name = 'Chi2_{}.png'.format(chi2_path.split('/')[-1][5:-4])
    chi2 = np.load(chi2_path)
    #plot_chi2(ml,inc,chi2,rst_folder=rst_folder,out_name=out_name,c_lim=[-1.0,1.5],ylim=[0,10])
    plot_chi2(ml,inc,chi2,rst_folder=rst_folder,out_name=out_name)
  for lambda_path in lambda_list:
    out_name = 'lambda_{}.png'.format(lambda_path.split('/')[-1][7:-4])
    lambda_obs = np.load(lambda_path)
    plot_chi2(ml,inc,lambda_obs,rst_folder=rst_folder,out_name=out_name,log=False)



