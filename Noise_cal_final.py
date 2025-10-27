"""
Created on Mon Oct 27 12:47:59 2025

Calculate JWST NIRISS SOSS exposure parameters and noise for a given exoplanet target.

@author: Qianyi Shen, Xingjie Zhao, Haoxu Feng
"""


import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
import astropy.constants as const
jax.config.update("jax_enable_x64", True)
import astropy.units as u
import corner
from math import ceil
import matplotlib.pyplot as plt
from exotic_ld import StellarLimbDarkening
from scipy.interpolate import CubicSpline
from squishyplanet import OblateSystem
import emcee
from pandeia.engine.calc_utils import build_default_calc
from pandeia.engine.perform_calculation import perform_calculation
import matplotlib.ticker as mticker
import sys


np.random.seed(13)


target = "TOI-2537 b"  
df = pd.read_csv('selected_targets_v01.csv')
df['t_exp'] = 0.0
df['t_total'] = 0.0
df['sigma_frac'] = 0.0
params = df.loc[df['Planet Name'] == target].iloc[0].to_dict()

teff=params['Teff']
logg=params['logg']
mh=params['M_H']
ksmag=params['kmag']
period = params['P']
a_s = params['a']
r_s = (params['R_p'] * const.R_jup)/ (params['R_*'] * const.R_sun)  
t14 = params['t14']  # hours
t14s = t14 * 3600
b = params['b']
i = params['i']  

f1 =  0.09796
theta = 26.73

n_init = 22 # initial ngroups

print('\na_s=',a_s)
print('r_s=',r_s)


#=========================================================================================================#
################################################# 计算噪声 ##################################################
#=========================================================================================================#
calculation = build_default_calc("jwst", "niriss", "soss")#method="specapphot"不知道是啥
calculation['configuration']['instrument']['aperture'] = 'soss'
calculation['configuration']['instrument']['mode'] = 'soss'
calculation['configuration']['instrument']['disperser'] = 'gr700xd'
calculation['configuration']['instrument']['filter'] = 'clear'
calculation['configuration']['detector']['ngroup'] = 10
calculation['configuration']['detector']['nint'] = 1
calculation['configuration']['detector']['nexp'] = 1
calculation['configuration']['detector']['subarray'] = 'substrip256'
calculation['configuration']['detector']['readout_pattern'] = 'nisrapid'
calculation["background"] = "minzodi"
calculation["background_level"] = "low"
scene = calculation["scene"][0]
scene['position'] = {'x_offset': 0., 'y_offset': 0., 'orientation': 0., 'position_parameters': ['x_offset', 'y_offset', 'orientation']}
scene['shape'] = {'geometry': 'point'}
scene['spectrum'] = {'name': 'Phoenix Spectrum', 'spectrum_parameters': ['sed', 'normalization']}
scene['spectrum']['sed'] = {
    'sed_type': 'phoenix',
    't_eff': teff,      # 注意这里必须是 t_eff
    'log_g': logg,
    'metallicity': mh
}
scene['spectrum']['normalization'] = {
    'type': 'photsys',                 # 选择 photsys，因为 2MASS 属于 photometric system
    'bandpass': '2mass,ks',
    'norm_flux': ksmag,
    'norm_fluxunit': 'vegamag'
}

scene['spectrum']['lines'] = [] #必须存在，哪怕是空列表也行
scene['spectrum']['extinction'] = {'bandpass': 'j', 'law': 'mw_rv_31', 'unit': 'mag', 'value': 0}
calculation['scene'][0] = scene

def is_saturated(report):
    warnings = report.get('warnings', {})
    if 'full_saturated' in warnings:
        return True, warnings['full_saturated']
    elif 'partial_saturated' in warnings:
        return True, warnings['partial_saturated']
    return False, 0

def run_calculation(calculation, n, nint=1):
    calculation['configuration']['detector']['ngroup'] = n
    calculation['configuration']['detector']['nint'] = nint
    report = perform_calculation(calculation, webapp=False, dict_report=True)
    return report

def compute_snr(report):
    snr = np.array(report["1d"]["sn"])          # (2, 2040)
    order1_wave = snr[0, :]
    order2_snr  = snr[1, :]

    wave = np.array(order1_wave)
    snr  = np.array(order2_snr)
    mask = np.isfinite(wave) & np.isfinite(snr)
    wave = wave[mask]
    snr  = snr[mask]
    snr_int = np.trapz(snr**2, wave)
    total_snr = np.sqrt(snr_int)
    sigma_frac = 1e6 / total_snr
    return total_snr, sigma_frac


n = n_init
loop_count = 0
report = run_calculation(calculation, n, nint=1)
sat, n_sat = is_saturated(report)

if sat:
    while sat and n > 1:
        loop_count += 1
        print(f"[Loop {loop_count}] n = {n}, sat, n_pixel = {n_sat}")
        n -= 1
        report = run_calculation(calculation, n, nint=1)
        sat, n_sat = is_saturated(report)
    n_f = n

else:
    while not sat:
        loop_count += 1
        print(f"[Loop {loop_count}] n = {n}, not sat")
        n += 1
        report = run_calculation(calculation, n, nint=1)
        sat, n_sat = is_saturated(report)
        if sat:
            print(f"Loop {loop_count}] n = {n}, sat, npixel = {n_sat}")
    n_f = n - 1
    report = run_calculation(calculation, n_f, nint=1)


ngroups = ceil(n_f / 2) # final ngroups
report = run_calculation(calculation, ngroups, nint=1)
t_per_int = report["information"]["exposure_specification"]["exposure_time"]
print('t_exp=',t_per_int)
nint = ceil(t14s / t_per_int)
calculation['configuration']['detector']['ngroup'] = ngroups
calculation['configuration']['detector']['nint']   = nint

final_report = run_calculation(calculation, ngroups, nint=nint)
total_snr, sigma_frac = compute_snr(final_report)
t_total = final_report["information"]["exposure_specification"]["total_exposure_time"]
t_total /= 3600  
t_exp = t_per_int * u.s 
t_exp_day = t_exp.to(u.day).value


print(f"\nngroup = {ngroups}, nint = {nint}")
print(f"Total exposure time = {t_total:.2f} hours")
print(f"One exposure time = {t_exp:.2f} ")
print(f"Integrated SNR = {total_snr:.6g} (SNR·μm)")
print(f"sigma_frac = 1/total_snr = {sigma_frac:.6g} (ppm)")



t_exp_value = t_exp.to(u.s).value
#t_exp_value = int(t_exp_value)
#t_total_value = int(t_total)  
#sigma_frac_value = sigma_frac = int(sigma_frac)
df.loc[df['Planet Name'] == target, 't_exp'] = t_exp_value
df.loc[df['Planet Name'] == target, 't_total'] = t_total
df.loc[df['Planet Name'] == target, 'sigma_frac'] = sigma_frac

filepath = 'selected_targets_v02.csv'
df.to_csv(filepath, index=False, header=True)


