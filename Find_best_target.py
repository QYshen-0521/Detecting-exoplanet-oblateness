from tqdm import tqdm
from scipy.interpolate import interp1d
import jax
import jax.numpy as jnp
import numpy as np
import os
import pandas as pd
import astropy.units as u
import astropy.constants as const
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
#Used to control terminal output format
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
np.random.seed(13)

#================ Calculate the average value of exoplanets ================
csv_file = "cal_mean.csv"# You can find this CSV file on our GitHub page
df = pd.read_csv(csv_file, skiprows=113)

distances = df['sy_dist'].dropna().values
peri = df['pl_orbper'].dropna().values
inclinations = df['pl_orbincl'].dropna()
inclinations = inclinations.astype(float)

M_Ks_sun = 3.28
m_sun_virtual = M_Ks_sun + 5 * np.log10(distances) - 5
# Convert to flux and take the average
flux = 10 ** (-0.4 * m_sun_virtual)
mean_flux = np.mean(flux)
m_avg = -2.5 * np.log10(mean_flux)

print(f"The Sun’s average apparent magnitude at the distances of these exoplanets (Ks band) = {m_avg:.3f}")
period_mean = np.mean(peri) * u.day
print(f"Average period = {period_mean:.2f}")
mean_inclination = inclinations.mean()
print(f"Average orbital inclination = {mean_inclination:.4f} degree")
print("="*50)

#================ Parameters ================
# Stellar parameters (Sun)
teff = 5772
logg = 4.438
mh = 0.0
ksmag = m_avg
# Planetary parameters (Saturn-like)
period = period_mean.value
a_s = (9.537 / 1.0)
a_au = 9.537
r_star = 1.0
r_jup = 0.843
r_s = ((r_jup * u.R_jup) / (r_star * u.R_sun)).to(u.dimensionless_unscaled).value

# Other parameters
i = 89.99
i_rad = np.deg2rad(i)
f1 = 0.09796
theta = 26.73
e = 0.0565
omega = 113.7

# Define the eccentricity factor
eccentricity_term = np.sqrt(1 - e**2) / (1 + e * np.sin(omega * np.pi / 180))
# Impact parameter b accounting for eccentricity
b = (a_au * u.au * np.cos(i_rad)).to(u.R_sun).value * eccentricity_term
# Calculate the t₁₄ transit duration
term_in_arcsin = np.sqrt((1 + (r_jup * const.R_jup) / (r_star* const.R_sun))**2 - b**2) / (a_s * np.sin(i * np.pi / 180))
transit_duration_days = (period / np.pi) * np.arcsin(term_in_arcsin) * eccentricity_term
transit_duration_hours = transit_duration_days * 24
t14 = transit_duration_hours
t14 = t14.value
t14s = t14 * 3600

# Initial value of n_groups,
# used as the starting point for the n_groups loop in the subsequent noise calculation program
n_init = 3

# Let's check whether all parameters are correct
print("=== Stellar & Planetary Parameters ===")
print(f"Teff       = {teff} (K")
print(f"logg       = {logg}")
print(f"[M/H]      = {mh}")
print(f"Kmag       = {ksmag}")
print(f"Period     = {period} (days")
print(f"a/R*       = {a_s}")
print(f"R_star     = {r_star} (R_sun")
print(f"R_planet   = {r_jup} (R_jup")
print(f"a/R_star   = {a_s:.4f}")
print(f"R_p/R_star = {r_s:.4f}")
print(f"impact parameter  = {b}")
print(f"inclination       = {i} (degree")
print(f"t14        = {t14} (hours")
print("="*50,'All of the above variables are dimensionless')

# ================ Generate stellar model ================

t_exp = 5.49
t_bond = 24.45

times = jnp.arange(-t_bond / 2, t_bond / 2, t_exp / 3600) / 24
t_exp_day = t_exp / ( 3600 * 24 )
sld = StellarLimbDarkening(
    M_H=mh,
    Teff=teff,
    logg=logg,
    ld_model="mps1",
    ld_data_path="exotic_ld_data",
    interpolate_type="trilinear",
    verbose=2
)
sld._integrate_I_mu(
    wavelength_range=[8_300, 28_100],
    mode="JWST_NIRISS_SOSSo1",
    custom_wavelengths=None,
    custom_throughput=None,
)

mu_grid = np.linspace(0.0, 1.0, 1000)
order = np.argsort(sld.mus)
f = CubicSpline(x=sld.mus[order], y=sld.I_mu[order])
interpolated_vals = f(mu_grid)
# Fit limb darkening coefficients
u_coeffs = OblateSystem.fit_limb_darkening_profile(
    intensities=interpolated_vals,
    mus=mu_grid,
    order=14
)

# ================ Calculate the interpolated f(b) ================

# Since b fluctuates more for values greater than 0.8,
# we use more interpolation points in the 0.8–1.0 range
b1 = np.linspace(0.05, 0.8, 14,endpoint=False)
b2 = np.linspace(0.8, 1.0, 12)
b_values = np.concatenate([b1, b2])

a = a_s
signal_strengths = []
residuals_list = []
time_array = np.array(times, dtype=np.float64)

# Loop over different values of b
for b in tqdm(b_values, desc="Computing signal strengths"):
    # If b ≥ a, there is no transit
    if b >= a:
        signal_strengths.append(0.0)
        continue
    # b = a * cos(i)
    i = jnp.arccos(b / ( a * eccentricity_term ) )

    injected_state = {
        "t0": 0,
        "times": times,
        "exposure_time": t_exp_day,
        "oversample": 3,
        "oversample_correction_order": 2,
        "a": a_s,
        "period": period,
        "r": r_s,
        "i": i,
        "ld_u_coeffs": np.array(u_coeffs),
        "f1": 0.09796,
        "obliq": 26.73 * jnp.pi / 180,
        "prec": 0.0 * jnp.pi / 180,
        "tidally_locked": False ,
        "omega" : omega ,
        "e" : e
    }

    injected_planet = OblateSystem(**injected_state)

    spherical_planet_state = injected_state.copy()
    spherical_planet_state["r"] = injected_planet.state["projected_effective_r"]
    spherical_planet_state["f1"] = 0.0
    spherical_planet = OblateSystem(**spherical_planet_state)

    injected_transit = np.array(injected_planet.lightcurve(), dtype=np.float64)
    spherical_transit = np.array(spherical_planet.lightcurve(), dtype=np.float64)

    # Signal strength = ∫ |Δflux| dt -> ppm
    diff_flux = (injected_transit - spherical_transit) ** 2
    signal = np.trapz(diff_flux, x=time_array) * 1e10  # ppm
    signal_strengths.append(signal)

signal_strengths = np.array(signal_strengths, dtype=np.float64)
b_unique, idx = np.unique(b_values, return_index=True)
signal_unique = signal_strengths[idx]
f_b_interp = interp1d(
    b_unique,
    signal_unique,
    kind="cubic",
    fill_value="extrapolate"
)
# Let's check whether the output results are reasonable
print("\n=== Integrated Signal Strengths ===")
for b, s in zip(b_values, signal_strengths):
    print(f"b={b:.2f}, integrated signal={s:.6e}")

# Draw a plot
b_fit = np.linspace(0, 1.0, 500)
plt.figure(figsize=(8, 5))
plt.scatter(b_values, signal_strengths, color="red", label="data", zorder=3)
plt.xlabel("Impact parameter b")
plt.ylabel("Signal Strength (ppm)")
plt.title("Fitted Signal Strength f(b)")
plt.legend()
plt.grid(True, ls="--", alpha=0.5)
plt.show()

# ================ Sort by considering all parameters together ================

df = pd.read_csv("PS_2025.09.21_06.55.14.csv", skiprows=114)
df.columns = df.columns.str.strip()
df = df[(df["pl_imppar"] >= 0) & (df["pl_imppar"] <= 1)].copy()

# Calculate the transit depth (if missing)
RJ_to_Rsun = 0.10045
mask_missing_trandep = df["pl_trandep"].isna()
mask_has_radii = df["pl_radj"].notna() & df["st_rad"].notna()
mask_to_fill = mask_missing_trandep & mask_has_radii
df.loc[mask_to_fill, "pl_trandep"] = (
    (df.loc[mask_to_fill, "pl_radj"] * RJ_to_Rsun / df.loc[mask_to_fill, "st_rad"]) ** 2 * 100
)

# Preliminary selection based on density and period
df = df[
    (df["pl_orbper"] >= 20) &
    (df["pl_orbper"] <= 300) &
    (df["pl_dens"] <= 1.64)
    ].copy()

# Calculate f(b) for each planet using the interpolation function
b=df["pl_imppar"]
df["f_b"] = f_b_interp(df["pl_imppar"])

# Calculate ingress time
Rp = df["pl_radj"]
P_days = df["pl_orbper"]
b = df["pl_imppar"]
a = df["pl_orbsmax"]
T_ingress_days = (2 * Rp / ( (2 * np.pi * a * 2092.7/ P_days) * np.sqrt(1 - b**2) )) #1AU≈2093RJ
df["T_ingress_days"] = T_ingress_days

# Remove missing values
df = df.dropna(subset=['pl_orbsmax', 'st_rad', 'st_met', 'st_teff', 'st_logg'], how='any')
df = df.dropna(subset=['pl_orbincl', 'pl_imppar'], how='all')
df = df.dropna(subset=["pl_trandep", "f_b", "T_ingress_days", "sy_kmag"])
df = df[df["sy_kmag"] > 0]

# Calculate Figure of Merit
df["FoM"] = (df["pl_trandep"] * df["f_b"] * T_ingress_days) / np.sqrt(df["sy_kmag"])

# Sort
df_sorted = df.sort_values(by="FoM", ascending=False)

# Output the top 60 entries (including the specified 5 fields)
output_cols = ["pl_name", "pl_dens", "pl_orbper", "sy_kmag", "pl_trandep", "pl_imppar", "T_ingress_days", "FoM"]
print(df_sorted[output_cols].head(60))
