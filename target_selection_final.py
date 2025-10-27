"""
Created on Mon Oct 27 13:10:38 2025

Select targets from NASA Exoplanet Archive for oblate planet transit fitting.

@author: Qianyi Shen, Xingjie Zhao, Haoxu Feng
"""

import jax
import sys
#import warnings
import astropy.units as u
import astropy.constants as const
import corner
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from exotic_ld import StellarLimbDarkening
from exotic_ld.ld_laws import nonlinear_4param_ld_law, quadratic_ld_law
#from gadfly import (Hyperparameters,PowerSpectrum,ShotNoiseKernel,StellarOscillatorKernel,)
#from lightkurve import search_lightcurve
#from nautilus import Prior, Sampler
#from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
#from tqdm import tqdm
import emcee
import os   
from squishyplanet import OblateSystem
jax.config.update("jax_enable_x64", True)
np.random.seed(13)

df = pd.read_csv('PS_2025.09.21_06.55.14.csv', skiprows=114)


df = df.dropna(subset=['pl_orbsmax', 'st_rad', 'st_met', 'st_teff', 'st_logg'], how='any')
df = df.dropna(subset=['pl_orbincl', 'pl_imppar'], how='all')
df.loc[df['pl_dens'].isnull(), 'pl_dens'] = df['pl_bmassj'] * const.M_jup / ( (4/3) * np.pi * (df['pl_radj'] * const.R_jup)**3 ) * 1e-3

df = df[(df['pl_orbper'] > 30) & (df['pl_orbper'] < 300)]  
df = df[df['pl_dens'] < 1.64]  


a = df['pl_orbsmax'] * const.au / (df['st_rad'] * const.R_sun)
df.loc[df['pl_ratdor'].isnull(), 'pl_ratdor'] = a

i = np.arccos(df['pl_imppar'] / df['pl_ratdor'] 
                * (1 + df['pl_orbeccen'] * np.sin(df['pl_orblper'] * np.pi / 180)) / (1 - df['pl_orbeccen'] ** 2)) * 180 / np.pi
#df.loc[df['pl_orbincl'].isnull(), 'pl_orbincl'] = np.arccos(df['pl_imppar'] / df['pl_ratdor']) * 180 / np.pi
df.loc[df['pl_orbincl'].isnull(),['pl_orbincl']] = i
#df.loc[df['pl_orbincl'].isnull(),['pl_orbincl']] = np.arccos(df['pl_imppar'] / df['pl_ratdor'] * (1 + df['pl_orbeccen'] * np.sin(df['pl_orblper'] * np.pi / 180)) / (1 - df['pl_orbeccen'] ** 2)) # * 180 / np.pi
df = df.dropna(subset=['pl_orbincl'])


b = df['pl_ratdor'] * np.cos(df['pl_orbincl'] * np.pi / 180) * (1 - df['pl_orbeccen'] ** 2) / (1 + df['pl_orbeccen'] * np.sin(df['pl_orblper'] * np.pi / 180))
df.loc[df['pl_imppar'].isnull(), 'pl_imppar'] = b
#df['pl_imppar'] = b

df.loc[df['pl_orblper'] < 0, 'pl_orblper'] += 360

term_in_arcsin = np.sqrt((1 + (df['pl_radj'] * const.R_jup) / (df['st_rad'] * const.R_sun))**2 - df['pl_imppar']**2) / (df['pl_ratdor'] * np.sin(df['pl_orbincl'] * np.pi / 180))
eccentricity_term = np.sqrt(1 - df['pl_orbeccen']**2) / (1 + df['pl_orbeccen'] * np.sin(df['pl_orblper'] * np.pi / 180)) 
transit_duration_days = (df['pl_orbper'] / np.pi) * np.arcsin(term_in_arcsin) * eccentricity_term
transit_duration_hours = transit_duration_days * 24
df.loc[df['pl_trandur'].isnull(), 'pl_trandur'] = transit_duration_hours
#df.loc[df['pl_trandur'].isnull(), 'pl_trandur'] = transit_duration_hours
#df = df.dropna(subset=['pl_imppar'])

columns_to_keep = ['pl_name', 'st_met', 'st_teff', 'st_logg', 'pl_orbper', 'pl_ratdor',
                    'pl_radj', 'st_rad', 'pl_orbincl', 'sy_kmag', 'pl_trandur', 'pl_imppar', 'pl_orbeccen', 'pl_orblper']

df = df[columns_to_keep]
new_column_names = [
    'Planet Name', 'M_H', 'Teff', 'logg',
    'P', 'a', 'R_p', 'R_*',
    'i', 'kmag', 't14', 'b', 'e', 'omega'
]

df.columns = new_column_names
#df = df.sort_values(by='result', ascending=False)
filepath = 'selected_targets_v01.csv'
df.to_csv(filepath, index=False, header=True)
#print(df.head(10))







