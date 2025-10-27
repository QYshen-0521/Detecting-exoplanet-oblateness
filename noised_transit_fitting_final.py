"""
Created on Mon Oct 27 22:05:25 2025

Modify from https://squishyplanet.readthedocs.io/en/latest/tutorials/fit_oblate_transit.html

Using emcee to fit a noisy transit lightcurve of an oblate planet.

@author: Qianyi Shen, Xingjie Zhao, Haoxu Feng
"""

import jax
import jax.numpy as jnp
import numpy as np
import os
jax.config.update("jax_enable_x64", True)
import astropy.units as u
import corner
import matplotlib.pyplot as plt
from exotic_ld import StellarLimbDarkening
from scipy.interpolate import CubicSpline
from squishyplanet import OblateSystem
import emcee

np.random.seed(13)

def get_unique_filepath(base_path):
    """
       'path/file.txt' -> 'path/file_01.txt' -> 'path/file_02.txt'
    """
    if not os.path.exists(base_path):
        return base_path

    path, ext = os.path.splitext(base_path)
    i = 1
    while True:
        new_path = f"{path}_{i:02d}{ext}"
        if not os.path.exists(new_path):
            return new_path
        i += 1

def plot_corner(samples, param_names):
        target_names = ['projected_f', 'projected_theta']
        indices = [i for i, name in enumerate(param_names) if name in target_names]
        labels = [r'$f_p$', r'$\theta_p$']
        fig = corner.corner(
            samples[:, indices], 
            labels=labels, 
            quantiles=[0.16, 0.5, 0.84], 
            show_titles=True,
            title_fmt=".2g"
        )
        
        corner_plot_path = get_unique_filepath(f'{target}/{target}_oblate_corner_f={f1}_obliq={theta}.png')
        fig.savefig(corner_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

target = "TOI-2537 b"
t_exp = 65.95 * u.s
t_total = 4.945 
timemax = (t_total + 1) * 1.2 / 2 
times = jnp.arange(-timemax, timemax, t_exp.to(u.hour).value) * u.hour.to(u.day)
theta = 26.73  

# generate the stellar intensities
sld = StellarLimbDarkening(
    M_H=0.08,
    Teff=4870,
    logg=4.551,
    ld_model="mps1",
    ld_data_path="exotic_ld_data",
    interpolate_type="trilinear",
    verbose=1
)
sld._integrate_I_mu(
    wavelength_range=[8_300, 28_100],
    mode="JWST_NIRISS_SOSSo1",
    custom_wavelengths=None,
    custom_throughput=None,
)

# interpolate the stellar intensities
mu_grid = np.linspace(0.0, 1.0, 1_000)
order = np.argsort(sld.mus)
f = CubicSpline(x=sld.mus[order], y=sld.I_mu[order])
interpolated_vals = f(mu_grid)
# fit the limb darkening profile
u_coeffs = OblateSystem.fit_limb_darkening_profile(
    intensities=interpolated_vals, mus=mu_grid, order=14
)
period = 94.1022
a_au = 0.3715
r_star = 0.771
r_jup = 1.004
b = 0.480
i = 89.592
e = 0.364
omega_deg = 75.2
f1 =  0.09796
a_s = ((a_au * u.au) / (r_star * u.R_sun)).to(u.dimensionless_unscaled).value
r_s = 0.1338

# create the planet
injected_state = {
    "t_peri": -period / 4,  # t_c=0
    "times": times,
    "exposure_time": t_exp.to(u.day).value,
    "oversample": 3,  # 3x more samples under-the-hood, then binned back down
    "oversample_correction_order": 2,
    "a": a_s,  # unit a/R_*
    "period": period,
    "r": r_s,
    "i": i * jnp.pi / 180,
    "ld_u_coeffs": u_coeffs,
    "f1": f1,  #
    "obliq": 26.73 * jnp.pi / 180,
    "prec": 0.0 * jnp.pi / 180,  # randomly chosen
    "tidally_locked": False,
}
injected_planet = OblateSystem(**injected_state)
# create a spherical planet with the same projected area and orbital parameters for comparison
spherical_planet_state = injected_state.copy()
spherical_planet_state["r"] = injected_planet.state["projected_effective_r"]
spherical_planet_state["f1"] = 0.0
spherical_planet = OblateSystem(**spherical_planet_state)
# generate the lightcurves
injected_transit = injected_planet.lightcurve()
spherical_transit = spherical_planet.lightcurve()
# add noise to the injected light curve
# scale the shot noise/integration to our longer binned light curve
shot_noise_amplitude = (110.46 * 1e-6) 
shot_noise = (
        jax.random.normal(jax.random.PRNGKey(0), times.shape) * shot_noise_amplitude
)
# shot_noise = shot_noise * 0
noised_data = injected_transit + shot_noise
if False: # plot the injected lightcurve and the noised data
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(8, 7))
    ax[0].plot(times * u.day.to(u.hour), injected_transit, label="injected transit")
    ax[0].plot(times * u.day.to(u.hour), noised_data, marker="o", markersize=2, ls="none", label="noised data")
    ax[1].plot(times * u.day.to(u.hour), (injected_transit - spherical_transit) * 1e6, label="injected oblate- spherical")
    ax[1].set(ylabel="diff. w/ \nsph. planet [ppm]", xlabel="time [hours]")
    plt.subplots_adjust(hspace=0)
    plt.close()
# plt.show()
# sys.exit()

init_state = {
    "times": times,
    "data": noised_data,  # first time we've used this: by saving data, we can use the loglike method
    "uncertainties": jnp.ones_like(noised_data) * shot_noise_amplitude,  # also need these
    "exposure_time": t_exp.to(u.day).value,
    "oversample": 3,
    "oversample_correction_order": 2,
    "period": period,
    "t_peri": (-period / 4 + 0.01),  
    "tidally_locked": False,
    "parameterize_with_projected_ellipse": True,
    "a": a_s,
    "i": i * jnp.pi / 180,
    "ld_u_coeffs": jnp.array([0.3, 0.2]),
    "projected_effective_r": r_s,
    "projected_f": 0.1,
    "projected_theta": 0 * jnp.pi / 180,  
    "log_jitter": -jnp.inf,
}
planet = OblateSystem(**init_state)  # the system we'll use for all our fits!
###################################### emcee ###########################################
param_names = ['t_peri', 'a', 'impact_param', 'q1', 'q2', 'projected_r', 'projected_f', 'projected_theta', 'log_jitter']
ndim = len(param_names)
def translate_to_dict(x):
    t_peri = x[0]
    a = x[1]
    impact_param = x[2]
    i = jnp.arccos(impact_param / a)
    q1 = x[3]  # using the Kipping 2013 uninformative quadratic limb darkening setup
    q2 = x[4]
    u1 = 2 * jnp.sqrt(q1) * q2
    u2 = jnp.sqrt(q1) * (1 - 2 * q2)
    projected_effective_r = x[5]
    projected_f = x[6]
    projected_theta = x[7]
    log_jitter = x[8]
    param_dict = {
        "t_peri": t_peri,
        "a": a,
        "i": i,
        "ld_u_coeffs": jnp.array([u1, u2]),
        "projected_effective_r": projected_effective_r,
        "projected_f": projected_f,
        "projected_theta": projected_theta,
        "log_jitter": log_jitter,
    }
    return param_dict


def log_likelihood(p_array):
    param_dict = translate_to_dict(p_array)
    return planet.loglike(param_dict)


def log_prior(p_array):
    t_peri, a, impact_param, q1, q2, projected_r, projected_f, projected_theta, log_jitter = p_array
    if not (-period / 4 - 5 < t_peri < -period / 4 + 5): return -np.inf
    if not (a_s - 5 < a < a_s + 5): return -np.inf
    if not (0.0 < impact_param < 1.0): return -np.inf
    if not (0.0 < q1 < 1.0): return -np.inf
    if not (0.0 < q2 < 1.0): return -np.inf
    if not (0.05 < projected_r < 0.15): return -np.inf
    if not (0.0 < projected_f < 0.3): return -np.inf
    if not (0.0 < projected_theta < np.pi): return -np.inf
    if not (np.log(1e-7) < log_jitter < np.log(1e-4)): return -np.inf
    return 0.0


def log_probability(p_array):
    lp = log_prior(p_array)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(p_array)

if __name__ == '__main__':
    p0 = np.array([
        -period / 4 + 0.05,
        a_s + 0.1,
        a_s * np.cos(i * np.pi / 180) + 0.005,
        0.3,
        0.2,
        r_s * 1.05,
        0.1,
        np.pi / 4,
        np.log(1e-6)
    ])

    nwalkers = 24
    nsteps = 5000
    burn_in = 1000
    pos = p0 + 1e-4 * np.random.randn(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    print("Starting burn-in...")
    sampler.run_mcmc(pos, burn_in, progress=True)
    print("Starting main MCMC...")
    sampler.reset()
    sampler.run_mcmc(pos, nsteps, progress=True)
    all_samples = sampler.get_chain(flat=True, discard=burn_in)


    ################## plotting corner plot ##################
    plot_corner(all_samples, param_names)

    # print best-fit parameters
    best_params = np.median(all_samples, axis=0)
    print("\nBest-fit parameters:")
    for name, value in zip(param_names, best_params):
        print(f"{name}: {value:.4f}")
    print("\nDone.")