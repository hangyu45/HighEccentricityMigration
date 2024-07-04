import numpy as np

### Physical constants ###
G=6.67259e-8
c=2.99792458e10
hbar=6.6260755e-27/2./np.pi
kb=1.380658e-16
sigma_sb=5.67051e-5
a=4.*sigma_sb/c

keV2erg=1.6021773e-9
eV2K = 1.16045e+4

### Astro constants ###
#distances
AU=1.4960e13
pc=3.0857e18

# sum
Ms=1.989e33
Rs=6.958e10
Ts=5770.
Ls=4.0*np.pi*Rs**2.0*sigma_sb*Ts**4.0

P_yr=86400.0*365.25

#Earth
Me=5.97e27
Re=6.37e8
we=7.27e-5
P_day=86400.

#Moon
Mm=7.35e25
a_em=3.84e10

#Jupiter
Mj=1.899e30
Rj=7.1492e9

# electron
me=9.11e-28
theta_T_e=6.65e-25 # Thomson scattering cross section
kappa_T_e = theta_T_e/me # opacity due to Thomson scattering

# proton
mp=1.67262178e-24
e_esu=4.803e-10 # in esu
rp=1.e-13

mn=1.674927471e-24

m_N=0.5*(mp+mn)

# atomic mass to Mev
u_Mev=931.494061

# fusion
H2He = (4.*1.0078250 - 4.0026032) * mp * c**2.

# cosmology
km_s_Mpc2cgs = 1.e5/(1.e6*pc)