# Functions commonly used in the x-ray data analysis process at CXI Hutch

import numpy as np
import pandas as pd
import h5py

import psana

import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter as gf
from scipy.ndimage import gaussian_filter1d as gf1
from scipy.signal import find_peaks
from scipy.stats import pearsonr as pr
from scipy.optimize import curve_fit
import scipy.optimize as opt 
import scipy.special
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.optimize import minimize

import math
import time
import os


## Functions related to reading .h5 files
def get_tree(f):
    """List the full tree of the h5 file. Not currently used in Radial Analyzer code."""
    def printname(name):  # needed because .visit needs callable
        print(name, type(f[name]))
    f.visit(printname)
    
def is_leaf(dataset):
    return isinstance(dataset,h5py.Dataset)

def get_leaves(f,saveto,verbose=True,keys=None):
    """Way of organizing the data I think? Used to put together combined data in the CombineRuns function"""
    def return_leaf(name):
        if is_leaf(f[name]):
            if verbose:
                print(name,f[name][()].shape)
            if keys is None or name in keys:
                saveto[name] = f[name][()]
    f.visit(return_leaf)

def group_var_by_stage(var_array, stage_array):
    """Function which returns variables which have been grouped by their stage position. Currently this is done through indexing which might be the best
    approach"""
    unique_pos = np.unique(stage_array)
    grouped_vars = []
    for i in range(len(unique_pos)):
        groups = var_array[np.where(scan == unique_pos[i])[0]]
        #print(len(groups))
        grouped_vars.append(groups)
    return unique_pos, grouped_vars


## Math Functions
def runNumToString(num):
    """Effective way of converting run number to a string of length 4"""
    numstr = str(num)
    while len(numstr)<4:
        numstr = '0'+numstr
    return numstr

def normalize(array):
    """Normalizing data to 0-1 scale. Will throw warning about dividing by 0 sometimes but works anyway"""
    min_val = np.min(array)
    max_val = np.max(array)
    try:
        norm_array = (array - min_val) / (max_val - min_val)
    except:
        norm_array = array
    return(norm_array)

def find_least_sqs(array1, array2):
    """Returns the r squared value showing the relationship between two arrays. Recall an r squared of 1 means perfectedly correlated and an r squared of 0 means
    not at all correlated.""" 
    correlation_matrix = np.corrcoef(array1, array2)
    correlation_xy = correlation_matrix[0,1]
    r_squared= correlation_xy**2
    return r_squared

def ttcorr(ttpos,ttpoly):
    return ttpoly[0]*ttpos+ttpoly[1]

def special_erf_func(z,a,b):
    return a*scipy.special.erf(z)+b

def theta_to_q(theta):
    return 4*np.pi*np.sin(theta/2.)/(wavelength)

def q_to_theta(q):
    return 2*np.arcsin((wavelength*q)/(4*np.pi))

def xyz_to_phi(x,y,z):
    return np.arctan2(y,x)
                
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def xyz_to_q(x,y,z):
    r_xy = np.sqrt(x**2+y**2)
    theta = np.arctan(r_xy/z)
    Q_abs = theta_to_q(theta)
    return Q_abs


## Plotting Functions
param_dict = {'spinewidth':2,
              'linewidth':4,
              'ticklength':6,
              'tickwidth':3,
              'ticklabelsize':20,
              'axislabelsize':20,
              'titlesize':25}

def neaten_plot(neatenme, param_dict=param_dict):
    """Function for producing neat plot with set fontsize, dimensions, etc.
    Inputs include a call to plot such as plt.gcf() and a dictionary of parameters as defined above."""
    if isinstance(neatenme,matplotlib.figure.Figure):
        for ax in neatenme.axes:
            neaten_plot(ax)
        plt.tight_layout()
    elif isinstance(neatenme,matplotlib.axes.Axes):
        neatenme.tick_params(labelsize=param_dict['ticklabelsize'],length=param_dict['ticklength'],
                             width=param_dict['tickwidth'])
        neatenme.xaxis.get_label().set_fontsize(param_dict['axislabelsize'])
        neatenme.yaxis.get_label().set_fontsize(param_dict['axislabelsize'])
        neatenme.title.set_fontsize(param_dict['titlesize'])
        for axis in ['top','bottom','left','right']:
            neatenme.spines[axis].set_linewidth(param_dict['spinewidth'])
        for line in neatenme.lines:
            line.set_linewidth(param_dict['linewidth'])
            
def plot_jungfrau(x,y,z,ax=None,shading='nearest',*args,**kwargs):
    """Plots images from the jungfrau camera. Never called in the Radial Analyzer code. Inputs are x,y,z values read from the 
    camera named Jungfrau """
    if not ax:
        ax=plt.gca()
    for i in range(8):
        pcm = ax.pcolormesh(x[i],y[i],z[i],shading=shading,*args,**kwargs)
    return pcm


## Simulation Functions

def load_xyz(filepath):
    '''
    filepath : path to .xyz file, or just molecule name if file is in xyz/ directory
    
    Returns atoms : list  of format [['atom1',x1,y1,z1],['atom2',x2,y2,z2],...]
    '''
    if not filepath.startswith('xyz/'):
        filepath = 'xyz/'+filepath
    with open(filepath,'r') as f:
        lines = f.readlines()
        numel_line = lines[0].strip()
        numel = int(numel_line[0])
        atoms = []
        for line in lines[2:]:
            line = line.strip()
            splitline = line.split()
            atom = [splitline[0]]
            for i in range(1,4):
                atom.append(float(splitline[i]))
            atoms.append(atom)
    return atoms

def load_form_fact(Element):
    '''
    Element : string indicating atomic species
    
    Returns ff : function that takes 3D Q vectors and spits out f(Q)
    '''
    coeffs = form_factors[Element]
    
    t1 = lambda q: coeffs[0]*np.exp(-1*coeffs[1]*(q/(4*np.pi))**2)
    t2 = lambda q: coeffs[2]*np.exp(-1*coeffs[3]*(q/(4*np.pi))**2)
    t3 = lambda q: coeffs[4]*np.exp(-1*coeffs[5]*(q/(4*np.pi))**2)
    t4 = lambda q: coeffs[6]*np.exp(-1*coeffs[7]*(q/(4*np.pi))**2) + coeffs[8]
    
    ff = lambda q: t1(q)+t2(q)+t3(q)+t4(q)
    
    return ff

def form_factor(q, ff):
    output = np.zeros_like(q)
    output[q>np.max(Q)] = ff(np.max(Q))
    output[q<np.min(Q)] = ff(np.min(Q))
    output[(q<=np.max(Q))&(q>=np.min(Q))] = ff(q[(q<=np.max(Q))&(q>=np.min(Q))])
    return output

def form_factor_total(q, ff):
    output = np.zeros_like(q)
    output[q>np.max(Q_total)] = ff_total(np.max(Q_total))
    output[q<np.min(Q_total)] = ff_total(np.min(Q_total))
    output[(q<=np.max(Q_total))&(q>=np.min(Q_total))] = ff_total(q[(q<=np.max(Q_total))&(q>=np.min(Q_total))])
    return output

def isotropic_scattering(xyz,QQ):
    output = np.zeros(QQ.shape,dtype=float)
    i_atomic = np.zeros(QQ.shape, dtype=float)
    for atom in xyz:
        output += np.abs(load_form_fact(atom[0])(QQ))**2
        i_atomic += np.abs(load_form_fact(atom[0])(QQ))**2
    for i in range(len(xyz)):
        for j in range(i+1,len(xyz)):
            atom1 = xyz[i][0]
            atom2 = xyz[j][0]
            xyz1 = np.array(xyz[i][1:])
            xyz2 = np.array(xyz[j][1:])
            ff1 = load_form_fact(atom1)
            ff2 = load_form_fact(atom2)
            r_ij = np.linalg.norm(xyz1-xyz2)
            output += 2*np.abs(ff1(QQ)*ff2(QQ))*np.sinc(1/np.pi*QQ*r_ij)
    return output, i_atomic

def symm_stretch(ts,amp,omega):
    xyzs = []
    for t in ts:
        C = ['C',0.0,0.0,0.0]
        S1 = ['S',-1.5904-amp*np.sin(omega*t),0,0]
        S2 = ['S',1.5904+amp*np.sin(omega*t),0,0]
        xyz = [C,S1,S2]
        xyzs.append(xyz)
    return xyzs

def get_relgeos(xyzs):
    from scipy.special import comb
    from numpy.linalg import norm
    
    relgeos = np.zeros((comb(len(xyzs[0]),2,exact=True),len(xyzs)))
    atompairs = []
    for i in range(len(xyzs[0])):
        for j in range(i+1,len(xyzs[0])):
            atom1, atom2 = xyzs[0][i][0], xyzs[0][j][0]
            atompairs.append([atom1,atom2])
    
    for it, atoms in enumerate(xyzs):
        pairindex = 0
        for i in range(len(atoms)):
            for j in range(i+1,len(atoms)):
                atom1, atom2 = np.array(atoms[i][1:]), np.array(atoms[j][1:])
                relgeos[pairindex,it] = norm(atom1-atom2,axis=0)
                pairindex+=1
    return np.array(relgeos), atompairs

def trajectory_scattering(xyzs,QQ):
    '''
    xyzs : len(xyzs)=Nt, each entry is an xyz list that isotropic_scattering() eats
    QQ : the Q array to feed to isotropic_scattering()
    '''
    output = np.zeros((len(xyzs),QQ.size))
    for i, xyz in enumerate(xyzs):
        scattering = isotropic_scattering(xyz,QQ)
        output[i] = scattering
    return output

def get_thomson_correction(x,y,z,phi0=0):
    r_xy = np.sqrt(x**2+y**2)
    theta = np.arctan(r_xy/z)
    phi = np.arctan2(y,x)+phi0
    correction = 1/(np.sin(phi)**2+np.cos(theta)**2*np.cos(phi)**2)
    return correction

def get_geometry_correction(x,y,z):
    # correction due to intensity falling off as 1/R**2
    R2 = (x**2+y**2+z**2)/z**2
    # correction due to flux through pixel area falling off with increased angle
    r_xy = np.sqrt(x**2+y**2)
    theta = np.arctan(r_xy/z)
    A = 1/np.cos(theta)
    return R2*A

def fitting_function(xy,xcenter,ycenter,z0,amplitude):
    """ Fits data from jungfrau image. xy represents the detector grid and z0 is maximum radial distance -- hypotenuse"""
    x = np.ravel(xy[0]) # One dimensional list of numbers
    y = np.ravel(xy[1]) # One dimensional list of numbers
    xcent = x-xcenter 
    ycent = y-ycenter
    r_xy = np.sqrt(xcent**2+ycent**2)
    theta = np.arctan(r_xy/z0)
    Q_abs = theta_to_q(theta)
    ff = amplitude*form_factor(Q_abs)
    thomson_correction = 1/get_thomson_correction(xcent,ycent,z0)
    geometry_correction = 1/get_geometry_correction(xcent,ycent,z0)
    ff = ff*thomson_correction*geometry_correction
    return ff

def fitting_function_freephi(xy,xcenter,ycenter,z0,amplitude,phi0):
    x = np.ravel(xy[0])
    y = np.ravel(xy[1])
    xcent = x-xcenter
    ycent = y-ycenter
    r_xy = np.sqrt(xcent**2+ycent**2)
    theta = np.arctan(r_xy/z0)
    Q_abs = theta_to_q(theta)
    ff = amplitude*form_factor(Q_abs)
    thomson_correction = 1/get_thomson_correction(xcent,ycent,z0,phi0=phi0)
    geometry_correction = 1/get_geometry_correction(xcent,ycent,z0)
    ff = ff*thomson_correction*geometry_correction
    return ff

def get_1d_fit(params,fitting_function):
    x0,y0,z0 = params[0],params[1],params[2]
    if len(params<5):
        phi0 = 0
    elif len(params)==5:
        phi0 = params[4]
    thomson_correction = get_thomson_correction(x-x0,y-y0,z0,phi0=phi0)
    PHI = xyz_to_phi(x-x0,y-y0,z0)
    geometry_correction = get_geometry_correction(x-x0,y-y0,z0)
    all_corrections = thomson_correction*geometry_correction
    fit_pattern = np.reshape(fitting_function([x,y],*params),(8,512,1024))*all_corrections
    data_corrected = jungfrau_sum*all_corrections

    QQ_data = xyz_to_q(x-x0,y-y0,z0)
    isotropic_data_1d = []
    isotropic_fit_1d = []
    QQ_1d_data = np.linspace(np.min(QQ_data),np.max(QQ_data),51)
    dQQ_1d_data = np.mean(np.diff(QQ_1d_data))
    QQ_1d_data = QQ_1d_data[:-1]
    residuals = []
    sems = []
    from scipy.stats import sem
    for qval in QQ_1d_data:
        isotropic_data_1d.append(np.mean(data_corrected[(QQ_data>qval)&(QQ_data<qval+dQQ_1d_data)]))
        sems.append(sem(data_corrected[(QQ_data>qval)&(QQ_data<qval+dQQ_1d_data)]))
        isotropic_fit_1d.append(np.mean(fit_pattern[(QQ_data>qval)&(QQ_data<qval+dQQ_1d_data)]))
        residuals.append(np.mean(np.abs((jungfrau_sum-fit_pattern)[(QQ_data>qval)&(QQ_data<qval+dQQ_1d_data)])))
    return QQ_1d_data, isotropic_data_1d, isotropic_fit_1d, fit_pattern


## Other Functions

def createBinsFromCenters(centers):
    """Inputs center values related to the time delay and makes bins of finite sizes with respect to the center. 
    Returns a sorted array of the bins"""
    bins = []
    nc = centers.size
    for idx,c in enumerate(centers):
        if idx == 0:
            dc = np.abs( c - centers[idx+1])/2.
            bins.append(c-dc)
            bins.append(c+dc)
        elif idx == nc-1:
            dc = np.abs( c - centers[idx-1])/2.
            bins.append(c+dc)
        else:
            dc = np.abs( c - centers[idx+1])/2.
            bins.append(c+dc)
    return np.sort(np.array(bins))

def fast_erf_fit(array, min_val = 0.1, max_val = 0.9):
    """Find position of line at 10% and at 90% and divide by two to approximate the center of the erf.
    Uses 1D data as array. Returns center positions, center amplitudes, and normalized data."""
    norm_temp = normalize(array)
    try:
        min_pos = np.max(np.where(norm_temp < min_val))
        max_pos = np.min(np.where(norm_temp > max_val))
        range_val = max_pos - min_pos
        if (min_pos < max_pos):
            range_val = max_pos - min_pos
            cent_pos = int(((max_pos-min_pos)/2)+min_pos)
            cent_amp = norm_temp[cent_pos]
            norm_data = norm_temp
            slope_val = (norm_temp[max_pos] - norm_temp[min_pos])/(max_pos - min_pos)
        else: 
            range_val = 0
            cent_pos = 0
            cent_amp = 0
            norm_data = np.zeros(length)
            slope_val = 0
    except: 
        return 0, 0, 0, 0
    
    return cent_pos, cent_amp, slope_val, norm_data

def detection_ratio(angle):
    detection_ratio=(1 - np.exp((-1/np.cos(angle))*((linear_attenuation_coefficient_mu_Al*thickness_1) + (linear_attenuation_coefficient_mu_Be*thickness_2)
                                          + (linear_attenuation_coefficient_mu_Si*thickness_3))))
    return detection_ratio