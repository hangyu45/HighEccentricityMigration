"""
    Source code for simulating Lidov-Kozai oscillations with tidal dissipation.  
    This code is used for generating results presented in 
    https://arxiv.org/abs/2406.00187    
    
    Copyright (C) 2024,  Hang Yu (hang.yu2@montana.edu)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


import numpy as np
import scipy.interpolate as interp
import scipy.signal as sig
import scipy.optimize as opt
import scipy.integrate as integ
import scipy.linalg as sla
import h5py as h5
from numba import jit, prange, njit

from myConstants import *

##########################################################
###             scalar factors                         ###
##########################################################

@njit
def get_t_lk(M1, M2, M3, ai, ao):
    omega_i = np.sqrt(G*(M1+M2)/ai**3.)
    t_lk = 1./omega_i * (M1+M2)/M3 * (ao/ai)**3.
    return t_lk

@njit
def get_Omega_ps(M1, M2, a, e):
    omega = np.sqrt(G*(M1+M2)/a**3.)

    e2 = e**2
    e4 = e2*e2
    e6 = e4*e2

    f2 = 1. + 7.5*e2 + 45.*e4/8. + 5.*e6/16.
    f5 = 1. + 3.*e2 + 0.375*e4
    
    Omega_ps = f2 / (f5 * np.sqrt(1-e2)**3.) * omega
    return Omega_ps    


##########################################################
###             various dy/dt                          ###
##########################################################

@njit
def get_dy_LK_da(y_LK_vect, par, par_LK):
    """
    Lidov-Kozai
    """
    # parse input
    Li_x, Li_y, Li_z, ei_x, ei_y, ei_z, \
    Lo_x, Lo_y, Lo_z, eo_x, eo_y, eo_z\
                = y_LK_vect
        
    # global par that stays constant
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
        
    # par for LK calc
    M2, \
    mu_i, mu_o, omega_i, ai, \
    Li_e0, Lo_e0, ei, eo, eff_i, eff_o,\
    uLi_x, uLi_y, uLi_z, \
    uLo_x, uLo_y, uLo_z\
                = par_LK
    # note eff_i = sqrt(1-e_i^2) and similarly for eff_o
    # Li_e0 = Li / eff_i = orb ang momentum at e_i=0

    # scalar quantities
    t_LK = (1./omega_i) * (M1+M2)/M3 * (ao*eff_o/ai)**3.
    p75_t_LK = 0.75/t_LK
    L_io_e0 = Li_e0/Lo_e0
    
    # directional products
    ji_v  = np.array([uLi_x, uLi_y, uLi_z]) * eff_i
    ei_v  = np.array([ei_x,  ei_y,  ei_z]) 
    uLo_v = np.array([uLo_x, uLo_y, uLo_z])
    eo_v = np.array([eo_x, eo_y, eo_z])
    
    ji_d_uLo = inner(ji_v, uLo_v)
    ei_d_uLo = inner(ei_v, uLo_v)
    ei_sq = ei**2.
    
    ji_c_uLo_v = cross(ji_v, uLo_v)
    ei_c_uLo_v = cross(ei_v, uLo_v)
    ji_c_ei_v  = cross(ji_v, ei_v)
    ji_c_eo_v  = cross(ji_v, eo_v)
    ei_c_eo_v  = cross(ei_v, eo_v)
    uLo_c_eo_v = cross(uLo_v, eo_v)
    
    # derivatives of dir vects 
    dji_v = p75_t_LK * (ji_d_uLo*ji_c_uLo_v - 5.*ei_d_uLo*ei_c_uLo_v)
    dei_v = p75_t_LK * (ji_d_uLo*ei_c_uLo_v + 2.*ji_c_ei_v - 5.*ei_d_uLo*ji_c_uLo_v)
    djo_v = p75_t_LK * L_io_e0 * (-ji_d_uLo*ji_c_uLo_v + 5.*ei_d_uLo*ei_c_uLo_v)
    deo_v = p75_t_LK * L_io_e0/eff_o \
            * (-ji_d_uLo*ji_c_eo_v + 5.*ei_d_uLo*ei_c_eo_v \
               -(0.5 - 3.*ei_sq + 12.5*ei_d_uLo**2. - 2.5*ji_d_uLo**2.)*uLo_c_eo_v\
              ) 
    
    if oct_flag and eo>0:
        eps_oct = (M1-M2)/(M1+M2) * (ai/ao) * (eo)/(1-eo**2.)
        oct_scale = -75./64. * eps_oct / t_LK
        
        ji_d_eo = inner(ji_v, eo_v)
        ei_d_eo = inner(ei_v, eo_v)
        
        ji_d_ueo = ji_d_eo / eo
        ei_d_ueo = ei_d_eo / eo
        
        ji_c_ueo_v = ji_c_eo_v / eo
        ei_c_ueo_v = ei_c_eo_v / eo
        
        dji_v_oct = 2. * (ei_d_ueo*ji_d_uLo + ei_d_uLo*ji_d_ueo) * ji_c_uLo_v\
                  + 2. * (ji_d_ueo*ji_d_uLo - 7.*ei_d_ueo*ei_d_uLo) * ei_c_uLo_v\
                  + 2. * (ei_d_uLo * ji_d_uLo) * ji_c_ueo_v\
                  + (1.6*ei_sq-0.2-7.*ei_d_uLo**2. +ji_d_uLo**2.) * ei_c_ueo_v 
        dji_v_oct *= oct_scale
        
        dei_v_oct = 2. * (ei_d_uLo*ji_d_uLo) * ei_c_ueo_v\
                  + (1.6*ei_sq-0.2-7.*ei_d_uLo**2.+ji_d_uLo**2.) * ji_c_ueo_v\
                  + 2. * (ei_d_ueo*ji_d_uLo+ei_d_uLo*ji_d_ueo) * ei_c_uLo_v\
                  + 2. * (ji_d_ueo*ji_d_uLo - 7.*ei_d_ueo*ei_d_uLo) * ji_c_uLo_v\
                  + 3.2 * ei_d_ueo * ji_c_ei_v
        dei_v_oct *= oct_scale        
        
        # adding back-reaction oct term 
        d_o_comp = (1.6*ei_sq-0.2-7.*ei_d_uLo**2.+ji_d_uLo**2.)

        djo_v_oct = - 2. * (ei_d_uLo*ji_d_ueo + ei_d_ueo*ji_d_uLo) * ji_c_uLo_v\
                    - 2. * ei_d_uLo * ji_d_uLo * ji_c_ueo_v\
                    - (2.*ji_d_ueo*ji_d_uLo - 14.*ei_d_ueo*ei_d_uLo) * ei_c_uLo_v\
                    - d_o_comp * ei_c_ueo_v
        djo_v_oct *= oct_scale * L_io_e0

        
        deo_v_oct = - 2. * (ei_d_uLo*ji_d_eo + ji_d_uLo*ei_d_eo) * ji_c_ueo_v\
                    - 2. * (1.-eo**2.)/eo * ei_d_uLo * ji_d_uLo * ji_c_uLo_v\
                    - 2. * (ji_d_eo*ji_d_uLo - 7.*ei_d_eo*ei_d_uLo) * ei_c_ueo_v\
                    - (1.-eo**2.)/eo * d_o_comp * ei_c_uLo_v\
                    + (\
                       2. * (0.2-1.6*ei_sq) * ei_d_ueo\
                     + 14. * ei_d_uLo * ji_d_uLo * ji_d_ueo\
                     + 7. * ei_d_ueo * d_o_comp\
                      ) * uLo_c_eo_v
        deo_v_oct *= oct_scale * L_io_e0/eff_o
        
        dji_v += dji_v_oct
        dei_v += dei_v_oct
        djo_v += djo_v_oct
        deo_v += deo_v_oct
        
    
    # derivatives on the angular momenta
    # the GW part have been accounted for in get_dy_orb_GR_GW()
    # should not need to include it again in LK
    dLi_v = Li_e0 * dji_v
    dLo_v = Lo_e0 * djo_v
    
    dLo_v *= br_flag
    deo_v *= br_flag
    
    dy_LK_vect = np.array([\
                 dLi_v[0], dLi_v[1], dLi_v[2], \
                 dei_v[0], dei_v[1], dei_v[2], \
                 dLo_v[0], dLo_v[1], dLo_v[2], \
                 deo_v[0], deo_v[1], deo_v[2]\
        ])
    
    return dy_LK_vect

@njit
def get_dy_orb_GR_GW(y_orb_vect, par, par_GR):
    """
    GR precession & GW decay
    """
    # parse input
    Li_x, Li_y, Li_z, \
    ei_x, ei_y, ei_z\
                = y_orb_vect
    
    # global par that stays constant
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
        
    # par for GR & GW calc
    M2, \
    mu_i, omega_i, ai, \
    Li_e0, ei, eff_i,\
    uLi_x, uLi_y, uLi_z\
                = par_GR
        
    # scalars
    omega_GR = 3.*G*(M1+M2)/(c**2.*ai*eff_i**2.) * omega_i
    
    G3_c5 = G**3./c**5.
    ei2 = ei**2.
    ei4 = ei**4.
  

    # do not compute the GW part to save computation
    dLi_GW = 0.
    dei_GW = 0.
    
#     dLi_GW = - (32./5.*G3_c5*np.sqrt(G)) * mu_i**2.*(M1+M2)**2.5 / (ai**3.5) \
#           * (1.+0.875*ei2)/(eff_i**4.)
#     dei_GW = - (304./15.*G3_c5) * mu_i*(M1+M2)**2./(ai**4.)\
#           * (1.+121./304.*ei2)/(eff_i**5.)
        
    # vectors
    uLi_v = np.array([uLi_x, uLi_y, uLi_z])
    ei_v  = np.array([ei_x, ei_y, ei_z])
    uLi_c_ei = cross(uLi_v, ei_v)
    
    # GR precession 
    dei_GR_v = omega_GR * uLi_c_ei
    
    # GW radiation
    dLi_GW_v = dLi_GW * uLi_v
    dei_GW_v = dei_GW * ei_v
        
    # total
    dLi_v = dLi_GW_v
    dei_v = dei_GR_v + dei_GW_v
    
    dy_orb_vect = np.array([\
                    dLi_v[0], dLi_v[1], dLi_v[2], \
                    dei_v[0], dei_v[1], dei_v[2]])
    return dy_orb_vect

@njit
def get_dy_ST_rot(y_ST_vect, par, par_ST):
    """
    static tide & rotational quadrupole in the planet 
    assumes S2_v is always aligned with Li_v and therefore only track its amplitude
    """
    # parse input
    Li_x, Li_y, Li_z, \
    ei_x, ei_y, ei_z, \
    S2\
                = y_ST_vect
    
    # global par that stays constant
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    # par for ST calc
    M2, \
    mu_i, omega_i, ai, \
    Li_e0, ei, eff_i, \
    uLi_x, uLi_y, uLi_z, \
    R2, tauST2_scale\
                = par_ST
    
    # scalars
    Li = Li_e0 * eff_i
    
    ei2 = ei**2
    ei4 = ei2*ei2
    ei6 = ei4*ei2
#     ei8 = ei6*ei2
    
#     f1 = 1. + 15.5*ei2 + 255.*ei4/8. + 185.*ei6/16. + 25*ei8/64.
    f2 = 1. + 7.5*ei2 + 45.*ei4/8. + 5.*ei6/16.
    f3 = 1. + 15.*ei2/4. + 15.*ei4/8. + 5*ei6/64.
    f4 = 1. + 1.5*ei2 + 0.125*ei4
    f5 = 1. + 3.*ei2 + 0.375*ei4
    
    j3 = eff_i**3.
    j4 = j3 * eff_i
    j10 = j3**3. * eff_i
    j13 = j3 * j10
    
    Omega_s2 = S2/(I2*M2*R2**2)
    Omega_s2_in_dyn_freq = Omega_s2 / np.sqrt(G*M2/R2**3.)
    j3Omega_s2_omega_i = j3*Omega_s2/omega_i
    
    M1M2_R2a5 = (M1/M2)*(R2/ai)**5.
    
    # tidal precession
    omega_ST_pre = 7.5 * k2 * M1M2_R2a5 * f4/j10 * omega_i 
    
    # spin precession
    omega_s2_pre = 1.5 * Krot2 * (R2/ai)**2. * Omega_s2_in_dyn_freq**2 / j4 * omega_i
    
    # dissipation; 1/t_a in Anderson+ 16, eq. A14
    omega_ST_diss = 6. * k2 * (tauST2_scale*tauST2) * M1M2_R2a5 * omega_i**2.
    omega_ST_diss_j13 = omega_ST_diss / j13
    
    # vectors
    uLi_v = np.array([uLi_x, uLi_y, uLi_z])
    ei_v  = np.array([ei_x, ei_y, ei_z])
    uLi_c_ei = cross(uLi_v, ei_v)    
    
    
    # precession
    dei_pre_v = (omega_ST_pre + omega_s2_pre) * uLi_c_ei
    
    # tidal dissipation
    # note: assuming S2 and Li are always aligned!!!
    dS2_ST   = -0.5*omega_ST_diss_j13*Li*(f5*j3Omega_s2_omega_i - f2)
    dLi_ST_v = -dS2_ST * uLi_v
    dei_ST_v = 0.5*omega_ST_diss_j13*(5.5*f4*j3Omega_s2_omega_i - 9.*f3)*ei_v
    
    # total
    dLi_v = dLi_ST_v
    dei_v = dei_ST_v + dei_pre_v
    dS2   = dS2_ST
    
    dy_ST_vect = np.array([\
                    dLi_v[0], dLi_v[1], dLi_v[2], \
                    dei_v[0], dei_v[1], dei_v[2], \
                    dS2
                          ])
    return dy_ST_vect
    
@njit
def get_dy_star_SL(y_SL_vect, par, par_SL):
    """
    rotational quadrupole in the star
    """
    # parse input
    Li_x, Li_y, Li_z, \
    ei_x, ei_y, ei_z, \
    S1_x, S1_y, S1_z\
                = y_SL_vect
    
    # global par that stays constant
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    # par for S_star-L_i calc
    M2, \
    mu_i, omega_i, ai, \
    Li_e0, ei, eff_i, \
    uLi_x, uLi_y, uLi_z, \
                = par_SL  
    
    # scalars
    S1 = np.sqrt(S1_x**2 + S1_y**2 + S1_z**2)
    moment_I1 = I1*M1*R1**2
    Omega_s1 = S1/moment_I1
    omega_dyn1 = np.sqrt(G*M1/R1**3)
    Omega_s1_in_dyn_freq = Omega_s1 / omega_dyn1
    
    j3 = eff_i**3.
    j4 = j3*eff_i
    
    # vectors
    uLi_v = np.array([uLi_x, uLi_y, uLi_z])
    ei_v  = np.array([ei_x, ei_y, ei_z])
    S1_v  = np.array([S1_x, S1_y, S1_z]) 
    
    uLi_d_S1 = inner(uLi_v, S1_v)
    uLi_d_S1_moment_I1 = uLi_d_S1 / moment_I1
    
    uLi_c_S1 = cross(uLi_v, S1_v)
    uLi_c_ei = cross(uLi_v, ei_v) 
    S1_c_ei  = cross(S1_v, ei_v) 
    
    
#     omega_s1_pre_L = -1.5*(G*M2/(ai*eff_i)**3.) * (Krot1*M1*R1**2/S1) * Omega_s1_in_dyn_freq**2. * uLi_d_uS1
#     omega_s1_pre_e = 1.5 * Krot1 * (R1/ai)**2. * Omega_s1_in_dyn_freq**2./j4 * omega_i

    omega_s1_pre_L = -1.5 * (Krot1/I1) * (M2/M1) * (R1/ai)**3./j3 * uLi_d_S1_moment_I1
    omega_s1_pre_e_omega_s_sq = \
                    1.5 * Krot1 * (R1/ai)**2. * omega_i / (j4 * omega_dyn1**2.)

    
    # precession
    dS1_v = omega_s1_pre_L * uLi_c_S1
    dLi_v = - dS1_v
#     dei_v = - omega_s1_pre_e * (uLi_d_uS1*uS1_c_ei \
#                                 + 0.5 * (1.-5*uLi_d_uS1**2)*uLi_c_ei)
    dei_v = - omega_s1_pre_e_omega_s_sq \
        *(uLi_d_S1_moment_I1 * S1_c_ei / moment_I1 \
          + 0.5*(Omega_s1**2-5.*uLi_d_S1_moment_I1**2.)*uLi_c_ei \
         )
    
    
    dy_SL_vect = np.array([\
                    dLi_v[0], dLi_v[1], dLi_v[2], \
                    dei_v[0], dei_v[1], dei_v[2], \
                    dS1_v[0], dS1_v[1], dS1_v[2]
                          ])
    return dy_SL_vect

@njit
def get_dy_star_SD(y_SD_vect, par):
    """
    spin down of the star's spin
    """
    S1_x, S1_y, S1_z\
                = y_SD_vect
    
    # global par that stays constant
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par

    S1 = np.sqrt(S1_x**2 + S1_y**2 + S1_z**2)
    Omega_s1 = S1/(I1*M1*R1**2)
    uS1_v = np.array([S1_x, S1_y, S1_z]) / S1

    
    dS1_v = - al_MB * Omega_s1**2 * S1 * uS1_v
    
    dy_SD_vect = np.array([\
                    dS1_v[0], dS1_v[1], dS1_v[2]
                          ])
    return dy_SD_vect



@njit
def get_dy_MT_S07(y_MT_vect, par, par_MT):
    # parse input
    Li_x, Li_y, Li_z, \
    ei_x, ei_y, ei_z, \
    M2, R2\
                = y_MT_vect
    
    # global par that stays constant
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    # par for S_star-L_i calc
    uLi_x, uLi_y, uLi_z, \
    Li, ei, omega_i, \
    rt, rp,\
    AAA, BBB\
                = par_MT 

    # mass loss from M2 (ignoring M1 change; fine only when M1>>M2)
    # fig.10 from https://iopscience.iop.org/article/10.1088/0004-637X/732/2/74/pdf
    # AAA ~ 1e7, BBB ~ 9 gives |dlogM2| = 4e-4 at rp/rt=2.65 and 4e-3 at rp/rt=2.4 
    # note dM2 < 0    
    if rp>(5.*rt):
        dlogM2 = 0
    else:
        dlogM2 = -AAA * np.exp(-BBB*(rp/rt)) * (omega_i/2/np.pi)
#         print(dlogM2, rp/rt, omega_i, AAA)

#         if ei<0.9:
#             dlogM2 *= np.exp(10.*(ei-0.9))
        
    dM2 = dlogM2 * M2
    
    # Gamma=2 polytrope
    dR2 = 0
    
    # dlogai and dei from eqs. B8 & B9 of SEPINSKY+ 07
    # for v_rel along y (tangent to the orb; eq. b6), no radial force
    # direction of e_v does not change
    dlogai = 2.*(1.+ei)/(1.-ei) * (M2/M1 - 1.) * dlogM2
    dei = (1.-ei) * dlogai
    
    # change in orb L
    dLi = 0.
    
    # 
    dLi_v = dLi * np.array([uLi_x, uLi_y, uLi_z])
    dei_v = dei/ei * np.array([ei_x, ei_y, ei_z]) 
    
    dy_MT_vect = np.array([
        dLi_v[0], dLi_v[1], dLi_v[2], 
        dei_v[0], dei_v[1], dei_v[2], 
        dM2, dR2
    ])
    return dy_MT_vect


@njit(nopython=True)
def get_dy_MT_non_cons(y_MT_vect, par, par_MT):
    """
    assuming mass losses from the planet are all lost from the system
    with dL/L = dE_orb/E_orb = dM2/M2
    when M1>>M2, this means essentially da = de = 0.
    """
    # parse input
    Li_x, Li_y, Li_z, \
    ei_x, ei_y, ei_z, \
    M2, R2\
                = y_MT_vect
    
    # global par that stays constant
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    # par for S_star-L_i calc
    uLi_x, uLi_y, uLi_z, \
    Li, ei, omega_i, \
    rt, rp,\
    AAA, BBB\
                = par_MT 

    # mass loss from M2 (ignoring M1 change; fine only when M1>>M2)
    # fig.10 from https://iopscience.iop.org/article/10.1088/0004-637X/732/2/74/pdf
    # AAA ~ 1e7, BBB ~ 9 gives |dlogM2| = 4e-4 at rp/rt=2.65 and 4e-3 at rp/rt=2.4 
    # note dM2 < 0    
    if rp>(5.*rt):
        dlogM2 = 0
    else:
        dlogM2 = -AAA * np.exp(-BBB*(rp/rt)) * (omega_i/2/np.pi)
#         print(dlogM2, rp/rt, omega_i, AAA)

#         if ei<0.9:
#             dlogM2 *= np.exp(10.*(ei-0.9))
        
    dM2 = dlogM2 * M2
    
    # Gamma=2 polytrope
    dR2 = 0
    
    #
    dei = 0.
    
    # change in orb L
    dLi = Li * dlogM2
    
    # 
    dLi_v = dLi * np.array([uLi_x, uLi_y, uLi_z])
    dei_v = np.zeros(3)
    
    dy_MT_vect = np.array([
        dLi_v[0], dLi_v[1], dLi_v[2], 
        dei_v[0], dei_v[1], dei_v[2], 
        dM2, dR2
    ])
    return dy_MT_vect
    


@njit
def get_dR2_th(R2, par, par_th):
    """
    thermal effects on the planet's radius
    """
    dR2 = 0.0
    
    return dR2

@njit
def get_dM2_th(M2, par, par_th):
    """
    thermal effects on the planet's mass
    """
    dM2 = 0.0
    
    return dM2

@njit
def get_L_tide_Q_eff_eq(ai, ei, M2, R2, par, tauST2_scale=1.):
    # global par that stays constant
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    omega_i_sq = G*(M1+M2)/ai**3.
    omega_i = np.sqrt(omega_i_sq)
        
    ei2 = ei**2
    ei4 = ei2*ei2
    ei6 = ei4*ei2
    ei8 = ei6*ei2

    eff_i = np.sqrt(1.-ei2)
    
    f1 = 1. + 15.5*ei2 + 255.*ei4/8. + 185.*ei6/16. + 25*ei8/64.
    f2 = 1. + 7.5*ei2 + 45.*ei4/8. + 5.*ei6/16.
    
    j3 = eff_i**3.
    j5 = eff_i * eff_i * j3
    j10 = j5 * j5
    j15 = j5 * j10
    
    Omega_s2 = get_Omega_ps(M1, M2, ai, ei)
    j3Omega_s2_omega_i = j3*Omega_s2/omega_i
    
    L_tide = 3. * k2 * omega_i_sq * (tauST2_scale * tauST2) * G*M1**2./R2 * (R2/ai)**6. \
                * (f1 - f2*j3Omega_s2_omega_i) / j15
    
    inv_Q_eff = 2./3. * k2 * omega_i * (tauST2_scale * tauST2) * (f1 - f2*j3Omega_s2_omega_i) / j15
    
    return L_tide, 1./inv_Q_eff


@njit
def get_T_mig_est(ai, c_I_io, M2, R2, eo, par, tauST2_scale=1.):
    # global par that stays constant
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    mu_i = M1*M2/(M1+M2)
    mu_o = (M1+M2)*M3/(M1+M2+M3)
    
    eta_LK = mu_i/mu_o * np.sqrt((M1+M2) * ai/ ((M1+M2+M3) * ao * (1.-eo**2)))
    
    aaa = eta_LK**2.
    bbb = -(3. + 4.*eta_LK*c_I_io + 9./4.*eta_LK**2.)
    ccc = 5.*(c_I_io + 0.5 * eta_LK)**2.
    eff_i_min_sq = (-bbb - np.sqrt(bbb**2. - 4.*aaa*ccc))/(2.*aaa)
    ei_max = np.sqrt(1.-eff_i_min_sq)
    print('ei_max', ei_max)
    
    
    E_orb_abs = G*M1*M2/ai/2.
    L_tide, _ = get_L_tide_Q_eff_eq(ai, ei_max, M2, R2, par, tauST2_scale)
    
    # Vick, Lai, & Andersson 19, eq 44
    t_ST_LK = E_orb_abs/L_tide /np.sqrt(1.-ei_max**2.)
    return t_ST_LK
    

@njit
def get_R2_Millholland(
    ai, ei, M2,
    M20, R20,
    par,
    L1_Ls=1., dlogR2_adj=1., 
    dlogR2_cut=3.):        
    # R2 from eq. 14 of https://arxiv.org/pdf/1910.06794.pdf    
    
    
    # global par that stays constant
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    omega_i_sq = G*(M1+M2)/ai**3.
    omega_i = np.sqrt(omega_i_sq)
    eff_i = np.sqrt(1.-ei*ei)
    
    ai_eff_sq = ai**2. * eff_i
    
    # fit to M-R relation
    # power law index should be ~ 0.2
    R2_free = (M2/M20)**0.2 * R20
    
    __, Q_eff = get_L_tide_Q_eff_eq(ai, ei, M2, R2_free, par, tauST2_scale=1.)
    ai_eff_sq = ai**2. * eff_i 
    F2_100Fe = L1_Ls * (AU**2./ai_eff_sq) /100.
    
    # eq. 8 of https://doi.org/10.3847/1538-4357/ab959c
    Q_eff_solar = Q_eff * (M2/Ms)**(-2) * (2.*np.pi / (P_yr*omega_i)) * (ai/AU)**6.
    
    # fixing core mass to 10 Me
    fenv = (M2-10.*Me)/M2
    if fenv < 0:
        fenv = 0.
    dlogR2 = 0.29 * (M2/Me/10.)**(-0.9) * (fenv/0.05)**0.9 \
                  * F2_100Fe**3.11 * (Q_eff_solar/1e5)**(-0.88) \
                  * dlogR2_adj

    
    if dlogR2 > dlogR2_cut:
        dlogR2 = dlogR2_cut
    
    R2 = R2_free * (1.+dlogR2)
    
    return R2

@njit
def get_R2_eq_T21(
    ai, ei, M2, R2,
    par,
    L1_Ls=1., dlogR2_adj=1.):        
    # R2 from https://arxiv.org/pdf/2101.05285.pdf 
    
    L_tide, __ = get_L_tide_Q_eff_eq(ai, ei, M2, R2, par, tauST2_scale=1.)
    
    F_irr = L1_Ls * Ls / (4.*np.pi * ai**2 * np.sqrt(1.-ei**2.))
    F_tide = L_tide / (4.*np.pi*R2**2.)
    
    R2_eq = dlogR2_adj \
            * 1.21 * Rj * (M2/Mj)**(-0.045) \
            * (1e-9*(F_irr + F_tide))**(0.149 - 0.072 * np.log10(M2/Mj))
    
    R2_eq_min = Rj * (M2/Mj)**0.2
    if R2_eq < R2_eq_min:
        R2_eq = R2_eq_min
    
    return R2_eq

@njit
def get_dR2_T21(
    ai, ei, M2, R2,
    par,
    L1_Ls=1., tau_d=0.5e9*P_yr, tau_r=1e6*P_yr, dlogR2_adj=1.):
    # R2 from https://arxiv.org/pdf/2101.05285.pdf 
    
    R2_eq = get_R2_eq_T21(
                ai, ei, M2, R2,
                par,
                L1_Ls, dlogR2_adj=dlogR2_adj)
    
    if R2 > R2_eq:
        dR2 = (R2_eq - R2)/tau_d
    else:
        dR2 = (R2_eq - R2)/tau_r
        
    return dR2
    
##########################################################
###             ode functions                          ###
##########################################################

@njit
def evol_LK_da(t_nat, y_nat_vect, par):
    # parse parameters
    # 0-5
    # 6-11
    # 12-14
    # 15,
    # 16, 17
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    Lo_nat_x, Lo_nat_y, Lo_nat_z, eo_x, eo_y, eo_z, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat, \
    M2_nat, R2_nat\
                = y_nat_vect
    
    # global par that stays constant
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    # convert to cgs units
    Li_v = np.array([Li_nat_x, Li_nat_y, Li_nat_z]) * Li_unit
    Lo_v = np.array([Lo_nat_x, Lo_nat_y, Lo_nat_z]) * Lo_unit
    S1_v = np.array([S1_nat_x, S1_nat_y, S1_nat_z]) * S1_unit
    S2   = S2_nat * S2_unit 
    
    ei_v = np.array([ei_x, ei_y, ei_z])
    eo_v = np.array([eo_x, eo_y, eo_z])
    
    M2 = M2_nat * M2_unit
    R2 = R2_nat * R2_unit
    
    # scalar quantities that will be useful for the other parts
    mu_i = M1*M2/(M1+M2)
    mu_o = (M1+M2)*M3/(M1+M2+M3)
    Li = np.sqrt(inner(Li_v, Li_v))
    Lo = np.sqrt(inner(Lo_v, Lo_v))
    ei = np.sqrt(inner(ei_v, ei_v))
    eo = np.sqrt(inner(eo_v, eo_v))
    eff_i = np.sqrt(1.-ei**2.)
    eff_o = np.sqrt(1.-eo**2.)
    
    Li_e0 = Li/eff_i
    Lo_e0 = Lo/eff_o
    
    ai = Li_e0**2./(G*(M1+M2)*mu_i**2.)
    
    omega_i = np.sqrt(G*(M1+M2)/ai**3.)
    
    S1 = np.sqrt(inner(S1_v, S1_v))
    
    # unity vectors
    uLi_v = Li_v / (Li_e0 * eff_i)
    uLo_v = Lo_v / (Lo_e0 * eff_o)
#     uS1_v = S1_v / S1
    
    # get LK terms
    y_LK_vect = np.array([Li_v[0], Li_v[1], Li_v[2], 
                          ei_v[0], ei_v[1], ei_v[2], 
                          Lo_v[0], Lo_v[1], Lo_v[2], 
                          eo_v[0], eo_v[1], eo_v[2]])
    par_LK = np.array([M2, 
                       mu_i, mu_o, omega_i, ai, 
                       Li_e0, Lo_e0, ei, eo, eff_i, eff_o,
                       uLi_v[0], uLi_v[1], uLi_v[2], 
                       uLo_v[0], uLo_v[1], uLo_v[2]])
    
    dLi_LK_x, dLi_LK_y, dLi_LK_z, \
    dei_LK_x, dei_LK_y, dei_LK_z, \
    dLo_LK_x, dLo_LK_y, dLo_LK_z, \
    deo_LK_x, deo_LK_y, deo_LK_z\
        = get_dy_LK_da(y_LK_vect, par, par_LK)
    
    # get GR terms
    y_orb_vect = np.array([Li_v[0], Li_v[1], Li_v[2], 
                           ei_v[0], ei_v[1], ei_v[2]])
    par_GR = np.array([M2, 
                       mu_i, omega_i, ai, 
                       Li_e0, ei, eff_i,
                       uLi_v[0], uLi_v[1], uLi_v[2]])

    dLi_GR_x, dLi_GR_y, dLi_GR_z, \
    dei_GR_x, dei_GR_y, dei_GR_z \
        = get_dy_orb_GR_GW(y_orb_vect, par, par_GR)
        

    # get static tide & rotational quadrupole induced precession terms due to the planet
    
    # allow an ad hoc factor here for potentially 
    # approximate incorporation of the diffusive tide
    tauST2_scale = 1.
    
    y_ST_vect = np.array([Li_v[0], Li_v[1], Li_v[2], 
                          ei_v[0], ei_v[1], ei_v[2], 
                          S2])
    par_ST = np.array([M2, 
                       mu_i, omega_i, ai, 
                       Li_e0, ei, eff_i, 
                       uLi_v[0], uLi_v[1], uLi_v[2], 
                       R2, tauST2_scale])
    
    dLi_ST_x, dLi_ST_y, dLi_ST_z, \
    dei_ST_x, dei_ST_y, dei_ST_z, \
    dS2_ST\
        = get_dy_ST_rot(y_ST_vect, par, par_ST)
    
    
    # get rotational quadrupole induced precession terms due to the star
    y_SL_vect = np.array([Li_v[0], Li_v[1], Li_v[2], 
                          ei_v[0], ei_v[1], ei_v[2], 
                          S1_v[0], S1_v[1], S1_v[2]])
    par_SL = np.array([M2, 
                       mu_i, omega_i, ai, 
                       Li_e0, ei, eff_i, 
                       uLi_v[0], uLi_v[1], uLi_v[2]])
    
    dLi_SL_x, dLi_SL_y, dLi_SL_z, \
    dei_SL_x, dei_SL_y, dei_SL_z, \
    dS1_SL_x, dS1_SL_y, dS1_SL_z\
        = get_dy_star_SL(y_SL_vect, par, par_SL)
    
    
    # get the stellar spin down terms
    y_SD_vect = np.array([S1_v[0], S1_v[1], S1_v[2]])
    
    dS1_SD_x, dS1_SD_y, dS1_SD_z\
        = get_dy_star_SD(y_SD_vect, par)
    
    # get thermal variations in the planetary radius
#     y_th_vect = np.array([R2])
#     par_th = np.array([])
    
#     dR2_th = get_dR2_th(R2, par, par_th)
#     dM2_th = get_dM2_th(M2, par, par_th)

    # total changes
    # inner orb sees LK + GR + planetary ST + stellar SL
    dLi_nat_x = (dLi_LK_x + dLi_GR_x + dLi_ST_x + dLi_SL_x) / Li_unit
    dLi_nat_y = (dLi_LK_y + dLi_GR_y + dLi_ST_y + dLi_SL_y) / Li_unit
    dLi_nat_z = (dLi_LK_z + dLi_GR_z + dLi_ST_z + dLi_SL_z) / Li_unit
    dei_x = dei_LK_x + dei_GR_x +dei_ST_x + dei_SL_x
    dei_y = dei_LK_y + dei_GR_y +dei_ST_y + dei_SL_y
    dei_z = dei_LK_z + dei_GR_z +dei_ST_z + dei_SL_z
    
    # outer orb sees only LK
    dLo_nat_x = dLo_LK_x / Lo_unit
    dLo_nat_y = dLo_LK_y / Lo_unit
    dLo_nat_z = dLo_LK_z / Lo_unit
    deo_x = deo_LK_x 
    deo_y = deo_LK_y 
    deo_z = deo_LK_z 
    
    # S1 sees stellar SL & SD
    dS1_nat_x = (dS1_SL_x + dS1_SD_x) / S1_unit
    dS1_nat_y = (dS1_SL_y + dS1_SD_y) / S1_unit
    dS1_nat_z = (dS1_SL_z + dS1_SD_z) / S1_unit
    
    # S2 sees planetary ST
    dS2_nat = dS2_ST / S2_unit
    
    # M2 & R2 sees thermal effects
#     print(dR2_th, R2_unit)
#     dR2_nat = dR2_th / R2_unit
    dM2_nat = 0.
    dR2_nat = 0.
    
    
    dy_nat_vect = np.array([
            dLi_nat_x, dLi_nat_y, dLi_nat_z, dei_x, dei_y, dei_z, 
            dLo_nat_x, dLo_nat_y, dLo_nat_z, deo_x, deo_y, deo_z, 
            dS1_nat_x, dS1_nat_y, dS1_nat_z, 
            dS2_nat, 
            dM2_nat, dR2_nat
                            ]) * t_unit
    return dy_nat_vect


@njit
def evol_LK_da_fix_S2_R2(t_nat, y_nat_vect, par):
    # parse parameters
    # 0-5
    # 6-11
    # 12-14
    # 15,
    # 16, 17
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    Lo_nat_x, Lo_nat_y, Lo_nat_z, eo_x, eo_y, eo_z, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat, \
    M2_nat, R2_nat\
                = y_nat_vect
    
    # global par that stays constant
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    # convert to cgs units
    Li_v = np.array([Li_nat_x, Li_nat_y, Li_nat_z]) * Li_unit
    Lo_v = np.array([Lo_nat_x, Lo_nat_y, Lo_nat_z]) * Lo_unit
    S1_v = np.array([S1_nat_x, S1_nat_y, S1_nat_z]) * S1_unit
    
    ei_v = np.array([ei_x, ei_y, ei_z])
    eo_v = np.array([eo_x, eo_y, eo_z])
    
    M2 = M2_nat * M2_unit
    R2 = R2_nat * R2_unit
    
    # scalar quantities that will be useful for the other parts
    mu_i = M1*M2/(M1+M2)
    mu_o = (M1+M2)*M3/(M1+M2+M3)
    Li = np.sqrt(inner(Li_v, Li_v))
    Lo = np.sqrt(inner(Lo_v, Lo_v))
    ei = np.sqrt(inner(ei_v, ei_v))
    eo = np.sqrt(inner(eo_v, eo_v))
    eff_i = np.sqrt(1.-ei**2.)
    eff_o = np.sqrt(1.-eo**2.)
    
    Li_e0 = Li/eff_i
    Lo_e0 = Lo/eff_o
    
    ai = Li_e0**2./(G*(M1+M2)*mu_i**2.)
    
    omega_i = np.sqrt(G*(M1+M2)/ai**3.)
    
    S1 = np.sqrt(inner(S1_v, S1_v))
    
    
    # fix the spin of the planet at pseudo synchronization value
    Omega_ps = get_Omega_ps(M1, M2, ai, ei)
    S2 = Omega_ps * I2 * M2*R2**2.
    
    # unity vectors
    uLi_v = Li_v / (Li_e0 * eff_i)
    uLo_v = Lo_v / (Lo_e0 * eff_o)
#     uS1_v = S1_v / S1
    
    # get LK terms
    y_LK_vect = np.array([Li_v[0], Li_v[1], Li_v[2], 
                          ei_v[0], ei_v[1], ei_v[2], 
                          Lo_v[0], Lo_v[1], Lo_v[2], 
                          eo_v[0], eo_v[1], eo_v[2]])
    par_LK = np.array([M2, 
                       mu_i, mu_o, omega_i, ai, 
                       Li_e0, Lo_e0, ei, eo, eff_i, eff_o,
                       uLi_v[0], uLi_v[1], uLi_v[2], 
                       uLo_v[0], uLo_v[1], uLo_v[2]])
    
    dLi_LK_x, dLi_LK_y, dLi_LK_z, \
    dei_LK_x, dei_LK_y, dei_LK_z, \
    dLo_LK_x, dLo_LK_y, dLo_LK_z, \
    deo_LK_x, deo_LK_y, deo_LK_z\
        = get_dy_LK_da(y_LK_vect, par, par_LK)
    
    # get GR terms
    y_orb_vect = np.array([Li_v[0], Li_v[1], Li_v[2], 
                           ei_v[0], ei_v[1], ei_v[2]])
    par_GR = np.array([M2, 
                       mu_i, omega_i, ai, 
                       Li_e0, ei, eff_i,
                       uLi_v[0], uLi_v[1], uLi_v[2]])

    dLi_GR_x, dLi_GR_y, dLi_GR_z, \
    dei_GR_x, dei_GR_y, dei_GR_z \
        = get_dy_orb_GR_GW(y_orb_vect, par, par_GR)
        

    # get static tide & rotational quadrupole induced precession terms due to the planet
    
    # allow an ad hoc factor here for potentially 
    # approximate incorporation of the diffusive tide
    tauST2_scale = 1.
    
    y_ST_vect = np.array([Li_v[0], Li_v[1], Li_v[2], 
                          ei_v[0], ei_v[1], ei_v[2], 
                          S2])
    par_ST = np.array([M2, 
                       mu_i, omega_i, ai, 
                       Li_e0, ei, eff_i, 
                       uLi_v[0], uLi_v[1], uLi_v[2], 
                       R2, tauST2_scale])
    
    dLi_ST_x, dLi_ST_y, dLi_ST_z, \
    dei_ST_x, dei_ST_y, dei_ST_z, \
    dS2_ST\
        = get_dy_ST_rot(y_ST_vect, par, par_ST)
    
    
    # get rotational quadrupole induced precession terms due to the star
    y_SL_vect = np.array([Li_v[0], Li_v[1], Li_v[2], 
                          ei_v[0], ei_v[1], ei_v[2], 
                          S1_v[0], S1_v[1], S1_v[2]])
    par_SL = np.array([M2, 
                       mu_i, omega_i, ai, 
                       Li_e0, ei, eff_i, 
                       uLi_v[0], uLi_v[1], uLi_v[2]])
    
    dLi_SL_x, dLi_SL_y, dLi_SL_z, \
    dei_SL_x, dei_SL_y, dei_SL_z, \
    dS1_SL_x, dS1_SL_y, dS1_SL_z\
        = get_dy_star_SL(y_SL_vect, par, par_SL)
    
    
    # get the stellar spin down terms
    y_SD_vect = np.array([S1_v[0], S1_v[1], S1_v[2]])
    
    dS1_SD_x, dS1_SD_y, dS1_SD_z\
        = get_dy_star_SD(y_SD_vect, par)

    # total changes
    # inner orb sees LK + GR + planetary ST + stellar SL
    dLi_nat_x = (dLi_LK_x + dLi_GR_x + dLi_ST_x + dLi_SL_x) / Li_unit
    dLi_nat_y = (dLi_LK_y + dLi_GR_y + dLi_ST_y + dLi_SL_y) / Li_unit
    dLi_nat_z = (dLi_LK_z + dLi_GR_z + dLi_ST_z + dLi_SL_z) / Li_unit
    dei_x = dei_LK_x + dei_GR_x +dei_ST_x + dei_SL_x
    dei_y = dei_LK_y + dei_GR_y +dei_ST_y + dei_SL_y
    dei_z = dei_LK_z + dei_GR_z +dei_ST_z + dei_SL_z
    
    # outer orb sees only LK
    dLo_nat_x = dLo_LK_x / Lo_unit
    dLo_nat_y = dLo_LK_y / Lo_unit
    dLo_nat_z = dLo_LK_z / Lo_unit
    deo_x = deo_LK_x 
    deo_y = deo_LK_y 
    deo_z = deo_LK_z 
    
    # S1 sees stellar SL & SD
    dS1_nat_x = (dS1_SL_x + dS1_SD_x) / S1_unit
    dS1_nat_y = (dS1_SL_y + dS1_SD_y) / S1_unit
    dS1_nat_z = (dS1_SL_z + dS1_SD_z) / S1_unit
    
    # do not evolve S2, M2 and R2

    
    dy_nat_vect = np.array([
            dLi_nat_x, dLi_nat_y, dLi_nat_z, dei_x, dei_y, dei_z, 
            dLo_nat_x, dLo_nat_y, dLo_nat_z, deo_x, deo_y, deo_z, 
            dS1_nat_x, dS1_nat_y, dS1_nat_z, 
            0., 
            0., 0.
                            ]) * t_unit
    return dy_nat_vect




@njit
def evol_LK_da_fix_S2_R2_w_MT(t_nat, y_nat_vect, par, 
                              AAA=1e3, BBB=9.25, cons_MT = True):
    # parse parameters
    # 0-5
    # 6-11
    # 12-14
    # 15,
    # 16, 17
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    Lo_nat_x, Lo_nat_y, Lo_nat_z, eo_x, eo_y, eo_z, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat, \
    M2_nat, R2_nat\
                = y_nat_vect
    
    # global par that stays constant
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    # convert to cgs units
    Li_v = np.array([Li_nat_x, Li_nat_y, Li_nat_z]) * Li_unit
    Lo_v = np.array([Lo_nat_x, Lo_nat_y, Lo_nat_z]) * Lo_unit
    S1_v = np.array([S1_nat_x, S1_nat_y, S1_nat_z]) * S1_unit
    
    ei_v = np.array([ei_x, ei_y, ei_z])
    eo_v = np.array([eo_x, eo_y, eo_z])
    
    M2 = M2_nat * M2_unit
    R2 = R2_nat * R2_unit
    
    # scalar quantities that will be useful for the other parts
    mu_i = M1*M2/(M1+M2)
    mu_o = (M1+M2)*M3/(M1+M2+M3)
    Li = np.sqrt(inner(Li_v, Li_v))
    Lo = np.sqrt(inner(Lo_v, Lo_v))
    ei = np.sqrt(inner(ei_v, ei_v))
    eo = np.sqrt(inner(eo_v, eo_v))
    eff_i = np.sqrt(1.-ei**2.)
    eff_o = np.sqrt(1.-eo**2.)
    
    Li_e0 = Li/eff_i
    Lo_e0 = Lo/eff_o
    
    ai = Li_e0**2./(G*(M1+M2)*mu_i**2.)
    
    omega_i = np.sqrt(G*(M1+M2)/ai**3.)
    
    S1 = np.sqrt(inner(S1_v, S1_v))
    
    
    rp = ai * (1.-ei)
    rt = np.cbrt(M1/M2) * R2 
    
    
    # fix the spin of the planet at pseudo synchronization value
    Omega_ps = get_Omega_ps(M1, M2, ai, ei)
    S2 = Omega_ps * I2 * M2*R2**2.
    
    # unity vectors
    uLi_v = Li_v / (Li_e0 * eff_i)
    uLo_v = Lo_v / (Lo_e0 * eff_o)
#     uS1_v = S1_v / S1
    
    # get LK terms
    y_LK_vect = np.array([Li_v[0], Li_v[1], Li_v[2], 
                          ei_v[0], ei_v[1], ei_v[2], 
                          Lo_v[0], Lo_v[1], Lo_v[2], 
                          eo_v[0], eo_v[1], eo_v[2]])
    par_LK = np.array([M2, 
                       mu_i, mu_o, omega_i, ai, 
                       Li_e0, Lo_e0, ei, eo, eff_i, eff_o,
                       uLi_v[0], uLi_v[1], uLi_v[2], 
                       uLo_v[0], uLo_v[1], uLo_v[2]])
    
    dLi_LK_x, dLi_LK_y, dLi_LK_z, \
    dei_LK_x, dei_LK_y, dei_LK_z, \
    dLo_LK_x, dLo_LK_y, dLo_LK_z, \
    deo_LK_x, deo_LK_y, deo_LK_z\
        = get_dy_LK_da(y_LK_vect, par, par_LK)
    
    # get GR terms
    y_orb_vect = np.array([Li_v[0], Li_v[1], Li_v[2], 
                           ei_v[0], ei_v[1], ei_v[2]])
    par_GR = np.array([M2, 
                       mu_i, omega_i, ai, 
                       Li_e0, ei, eff_i,
                       uLi_v[0], uLi_v[1], uLi_v[2]])

    dLi_GR_x, dLi_GR_y, dLi_GR_z, \
    dei_GR_x, dei_GR_y, dei_GR_z \
        = get_dy_orb_GR_GW(y_orb_vect, par, par_GR)
        

    # get static tide & rotational quadrupole induced precession terms due to the planet
    
    # allow an ad hoc factor here for potentially 
    # approximate incorporation of the diffusive tide
    tauST2_scale = 1.
    
    y_ST_vect = np.array([Li_v[0], Li_v[1], Li_v[2], 
                          ei_v[0], ei_v[1], ei_v[2], 
                          S2])
    par_ST = np.array([M2, 
                       mu_i, omega_i, ai, 
                       Li_e0, ei, eff_i, 
                       uLi_v[0], uLi_v[1], uLi_v[2], 
                       R2, tauST2_scale])
    
    dLi_ST_x, dLi_ST_y, dLi_ST_z, \
    dei_ST_x, dei_ST_y, dei_ST_z, \
    dS2_ST\
        = get_dy_ST_rot(y_ST_vect, par, par_ST)
    
    
    # get rotational quadrupole induced precession terms due to the star
    y_SL_vect = np.array([Li_v[0], Li_v[1], Li_v[2], 
                          ei_v[0], ei_v[1], ei_v[2], 
                          S1_v[0], S1_v[1], S1_v[2]])
    par_SL = np.array([M2, 
                       mu_i, omega_i, ai, 
                       Li_e0, ei, eff_i, 
                       uLi_v[0], uLi_v[1], uLi_v[2]])
    
    dLi_SL_x, dLi_SL_y, dLi_SL_z, \
    dei_SL_x, dei_SL_y, dei_SL_z, \
    dS1_SL_x, dS1_SL_y, dS1_SL_z\
        = get_dy_star_SL(y_SL_vect, par, par_SL)
    
    
    # get the stellar spin down terms
    y_SD_vect = np.array([S1_v[0], S1_v[1], S1_v[2]])
    
    dS1_SD_x, dS1_SD_y, dS1_SD_z\
        = get_dy_star_SD(y_SD_vect, par)
    
    
    # get the mass transfer terms
    y_MT_vect = np.array([
        Li_v[0], Li_v[1], Li_v[2], 
        ei_v[0], ei_v[1], ei_v[2],
        M2, R2
    ])
    par_MT = np.array([
        uLi_v[0], uLi_v[1], uLi_v[2], \
        Li, ei, omega_i, \
        rt, rp, \
        AAA, BBB
    ])
    
    if cons_MT:
        dLi_MT_x, dLi_MT_y, dLi_MT_z, \
        dei_MT_x, dei_MT_y, dei_MT_z, \
        dM2_MT, dR2_MT\
            =  get_dy_MT_S07(y_MT_vect, par, par_MT)
    else:
        dLi_MT_x, dLi_MT_y, dLi_MT_z, \
        dei_MT_x, dei_MT_y, dei_MT_z, \
        dM2_MT, dR2_MT\
            =  get_dy_MT_non_cons(y_MT_vect, par, par_MT)
    
    
    

    # total changes
    # inner orb sees LK + GR + planetary ST + stellar SL
    dLi_nat_x = (dLi_LK_x + dLi_GR_x + dLi_ST_x + dLi_SL_x + dLi_MT_x) / Li_unit
    dLi_nat_y = (dLi_LK_y + dLi_GR_y + dLi_ST_y + dLi_SL_y + dLi_MT_y) / Li_unit
    dLi_nat_z = (dLi_LK_z + dLi_GR_z + dLi_ST_z + dLi_SL_z + dLi_MT_z) / Li_unit
    dei_x = dei_LK_x + dei_GR_x +dei_ST_x + dei_SL_x + dei_MT_x
    dei_y = dei_LK_y + dei_GR_y +dei_ST_y + dei_SL_y + dei_MT_y
    dei_z = dei_LK_z + dei_GR_z +dei_ST_z + dei_SL_z + dei_MT_z
    
    # outer orb sees only LK
    dLo_nat_x = dLo_LK_x / Lo_unit
    dLo_nat_y = dLo_LK_y / Lo_unit
    dLo_nat_z = dLo_LK_z / Lo_unit
    deo_x = deo_LK_x 
    deo_y = deo_LK_y 
    deo_z = deo_LK_z 
    
    # S1 sees stellar SL & SD
    dS1_nat_x = (dS1_SL_x + dS1_SD_x) / S1_unit
    dS1_nat_y = (dS1_SL_y + dS1_SD_y) / S1_unit
    dS1_nat_z = (dS1_SL_z + dS1_SD_z) / S1_unit

    
    dy_nat_vect = np.array([
            dLi_nat_x, dLi_nat_y, dLi_nat_z, dei_x, dei_y, dei_z, 
            dLo_nat_x, dLo_nat_y, dLo_nat_z, deo_x, deo_y, deo_z, 
            dS1_nat_x, dS1_nat_y, dS1_nat_z, 
            0., 
            dM2_MT/M2_unit, dR2_MT/R2_unit
                            ]) * t_unit
    return dy_nat_vect


@njit
def evol_LK_da_fix_S2_R2_w_MT_R_infl(t_nat, y_nat_vect, par, 
                                     AAA=1e3, BBB=9.25, 
                                     L1_Ls=1.,
                                     tau_d=0.5e9*P_yr, tau_r=1e6*P_yr,
                                     dlogR2_adj=1.,
                                     cons_MT = True):
    # parse parameters
    # 0-5
    # 6-11
    # 12-14
    # 15,
    # 16, 17
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    Lo_nat_x, Lo_nat_y, Lo_nat_z, eo_x, eo_y, eo_z, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat, \
    M2_nat, R2_nat\
                = y_nat_vect
    
    # global par that stays constant
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    # convert to cgs units
    Li_v = np.array([Li_nat_x, Li_nat_y, Li_nat_z]) * Li_unit
    Lo_v = np.array([Lo_nat_x, Lo_nat_y, Lo_nat_z]) * Lo_unit
    S1_v = np.array([S1_nat_x, S1_nat_y, S1_nat_z]) * S1_unit
    
    ei_v = np.array([ei_x, ei_y, ei_z])
    eo_v = np.array([eo_x, eo_y, eo_z])
    
    M2 = M2_nat * M2_unit
    R2 = R2_nat * R2_unit
    
    
    # scalar quantities that will be useful for the other parts
    mu_i = M1*M2/(M1+M2)
    mu_o = (M1+M2)*M3/(M1+M2+M3)
    Li = np.sqrt(inner(Li_v, Li_v))
    Lo = np.sqrt(inner(Lo_v, Lo_v))
    ei = np.sqrt(inner(ei_v, ei_v))
    eo = np.sqrt(inner(eo_v, eo_v))
    eff_i = np.sqrt(1.-ei**2.)
    eff_o = np.sqrt(1.-eo**2.)
    
    Li_e0 = Li/eff_i
    Lo_e0 = Lo/eff_o
    
    ai = Li_e0**2./(G*(M1+M2)*mu_i**2.)
    
    omega_i = np.sqrt(G*(M1+M2)/ai**3.)
    
    S1 = np.sqrt(inner(S1_v, S1_v))

    rp = ai * (1.-ei)
    rt = np.cbrt(M1/M2) * R2 
    
    # fix the spin of the planet at pseudo synchronization value
    Omega_ps = get_Omega_ps(M1, M2, ai, ei)
    S2 = Omega_ps * I2 * M2*R2**2.
    
    # unity vectors
    uLi_v = Li_v / (Li_e0 * eff_i)
    uLo_v = Lo_v / (Lo_e0 * eff_o)
#     uS1_v = S1_v / S1
    
    # get LK terms
    y_LK_vect = np.array([Li_v[0], Li_v[1], Li_v[2], 
                          ei_v[0], ei_v[1], ei_v[2], 
                          Lo_v[0], Lo_v[1], Lo_v[2], 
                          eo_v[0], eo_v[1], eo_v[2]])
    par_LK = np.array([M2, 
                       mu_i, mu_o, omega_i, ai, 
                       Li_e0, Lo_e0, ei, eo, eff_i, eff_o,
                       uLi_v[0], uLi_v[1], uLi_v[2], 
                       uLo_v[0], uLo_v[1], uLo_v[2]])
    
    dLi_LK_x, dLi_LK_y, dLi_LK_z, \
    dei_LK_x, dei_LK_y, dei_LK_z, \
    dLo_LK_x, dLo_LK_y, dLo_LK_z, \
    deo_LK_x, deo_LK_y, deo_LK_z\
        = get_dy_LK_da(y_LK_vect, par, par_LK)
    
    # get GR terms
    y_orb_vect = np.array([Li_v[0], Li_v[1], Li_v[2], 
                           ei_v[0], ei_v[1], ei_v[2]])
    par_GR = np.array([M2, 
                       mu_i, omega_i, ai, 
                       Li_e0, ei, eff_i,
                       uLi_v[0], uLi_v[1], uLi_v[2]])

    dLi_GR_x, dLi_GR_y, dLi_GR_z, \
    dei_GR_x, dei_GR_y, dei_GR_z \
        = get_dy_orb_GR_GW(y_orb_vect, par, par_GR)
        

    # get static tide & rotational quadrupole induced precession terms due to the planet
    
    # allow an ad hoc factor here for potentially 
    # approximate incorporation of the diffusive tide
    tauST2_scale = 1.
    
    y_ST_vect = np.array([Li_v[0], Li_v[1], Li_v[2], 
                          ei_v[0], ei_v[1], ei_v[2], 
                          S2])
    par_ST = np.array([M2, 
                       mu_i, omega_i, ai, 
                       Li_e0, ei, eff_i, 
                       uLi_v[0], uLi_v[1], uLi_v[2], 
                       R2, tauST2_scale])
    
    dLi_ST_x, dLi_ST_y, dLi_ST_z, \
    dei_ST_x, dei_ST_y, dei_ST_z, \
    dS2_ST\
        = get_dy_ST_rot(y_ST_vect, par, par_ST)
    
    
    # get rotational quadrupole induced precession terms due to the star
    y_SL_vect = np.array([Li_v[0], Li_v[1], Li_v[2], 
                          ei_v[0], ei_v[1], ei_v[2], 
                          S1_v[0], S1_v[1], S1_v[2]])
    par_SL = np.array([M2, 
                       mu_i, omega_i, ai, 
                       Li_e0, ei, eff_i, 
                       uLi_v[0], uLi_v[1], uLi_v[2]])
    
    dLi_SL_x, dLi_SL_y, dLi_SL_z, \
    dei_SL_x, dei_SL_y, dei_SL_z, \
    dS1_SL_x, dS1_SL_y, dS1_SL_z\
        = get_dy_star_SL(y_SL_vect, par, par_SL)
    
    
    # get the stellar spin down terms
    y_SD_vect = np.array([S1_v[0], S1_v[1], S1_v[2]])
    
    dS1_SD_x, dS1_SD_y, dS1_SD_z\
        = get_dy_star_SD(y_SD_vect, par)
    
    
    # get the mass transfer terms
    y_MT_vect = np.array([
        Li_v[0], Li_v[1], Li_v[2], 
        ei_v[0], ei_v[1], ei_v[2],
        M2, R2
    ])
    par_MT = np.array([
        uLi_v[0], uLi_v[1], uLi_v[2], \
        Li, ei, omega_i, \
        rt, rp, \
        AAA, BBB
    ])
    
    if cons_MT:
        dLi_MT_x, dLi_MT_y, dLi_MT_z, \
        dei_MT_x, dei_MT_y, dei_MT_z, \
        dM2_MT, dR2_MT\
            =  get_dy_MT_S07(y_MT_vect, par, par_MT)
    else:
        dLi_MT_x, dLi_MT_y, dLi_MT_z, \
        dei_MT_x, dei_MT_y, dei_MT_z, \
        dM2_MT, dR2_MT\
            =  get_dy_MT_non_cons(y_MT_vect, par, par_MT)        
        
        
    # get the planetary radius inflation
    dR2_th = get_dR2_T21(
                ai, ei, M2, R2,
                par,
                L1_Ls=L1_Ls, tau_d=tau_d, tau_r=tau_r, 
                dlogR2_adj=dlogR2_adj)        

    # total changes
    # inner orb sees LK + GR + planetary ST + stellar SL
    dLi_nat_x = (dLi_LK_x + dLi_GR_x + dLi_ST_x + dLi_SL_x + dLi_MT_x) / Li_unit
    dLi_nat_y = (dLi_LK_y + dLi_GR_y + dLi_ST_y + dLi_SL_y + dLi_MT_y) / Li_unit
    dLi_nat_z = (dLi_LK_z + dLi_GR_z + dLi_ST_z + dLi_SL_z + dLi_MT_z) / Li_unit
    dei_x = dei_LK_x + dei_GR_x +dei_ST_x + dei_SL_x + dei_MT_x
    dei_y = dei_LK_y + dei_GR_y +dei_ST_y + dei_SL_y + dei_MT_y
    dei_z = dei_LK_z + dei_GR_z +dei_ST_z + dei_SL_z + dei_MT_z
    
    # outer orb sees only LK
    dLo_nat_x = dLo_LK_x / Lo_unit
    dLo_nat_y = dLo_LK_y / Lo_unit
    dLo_nat_z = dLo_LK_z / Lo_unit
    deo_x = deo_LK_x 
    deo_y = deo_LK_y 
    deo_z = deo_LK_z 
    
    # S1 sees stellar SL & SD
    dS1_nat_x = (dS1_SL_x + dS1_SD_x) / S1_unit
    dS1_nat_y = (dS1_SL_y + dS1_SD_y) / S1_unit
    dS1_nat_z = (dS1_SL_z + dS1_SD_z) / S1_unit

    
    dy_nat_vect = np.array([
            dLi_nat_x, dLi_nat_y, dLi_nat_z, dei_x, dei_y, dei_z, 
            dLo_nat_x, dLo_nat_y, dLo_nat_z, deo_x, deo_y, deo_z, 
            dS1_nat_x, dS1_nat_y, dS1_nat_z, 
            0., 
            dM2_MT/M2_unit, (dR2_MT + dR2_th)/R2_unit
                            ]) * t_unit
    return dy_nat_vect
    

##########################################################
###             events functions                       ###
##########################################################

@njit
def disruption(t_nat, y_nat_vect, par, Rtide_th=2.7):
    # parse parameters
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    Lo_nat_x, Lo_nat_y, Lo_nat_z, eo_x, eo_y, eo_z, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat, \
    M2_nat, R2_nat \
                = y_nat_vect
    
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    # convert to cgs units
    M2 = M2_nat * M2_unit
    
    Li_v = np.array([Li_nat_x, Li_nat_y, Li_nat_z]) * Li_unit
    ei_v = np.array([ei_x, ei_y, ei_z])
    R2 = R2_nat * R2_unit
    Rtide = (M1/M2)**(1./3) * R2
    
    # scalar quantities 
    mu_i = M1*M2/(M1+M2)
    
    # find rp
    Li_sq = inner(Li_v, Li_v)
    ei = np.sqrt(inner(ei_v, ei_v))
    rp = Li_sq/(mu_i**2.*G*(M1+M2)*(1.+ei))
    
    resi = (rp-Rtide_th*Rtide)/R2_unit

    return resi


@njit
def disruption_M2(t_nat, y_nat_vect, par, M2_Mj_th=0.03):
    # parse parameters
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    Lo_nat_x, Lo_nat_y, Lo_nat_z, eo_x, eo_y, eo_z, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat, \
    M2_nat, R2_nat \
                = y_nat_vect
    
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    # convert to cgs units
    M2 = M2_nat * M2_unit
    
    resi = (M2/Mj - M2_Mj_th)
    return resi

@njit
def ejection(t_nat, y_nat_vect, par, 
            P_orb_day_th = 70):
    # parse parameters
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    Lo_nat_x, Lo_nat_y, Lo_nat_z, eo_x, eo_y, eo_z, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat, \
    M2_nat, R2_nat \
                = y_nat_vect
    
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    # convert to cgs units
    M2 = M2_nat * M2_unit
    Li_v = np.array([Li_nat_x, Li_nat_y, Li_nat_z]) * Li_unit
    ei_v = np.array([ei_x, ei_y, ei_z])
    
    # scalar quantities 
    GMt = G*(M1+M2)
    mu_i = M1*M2/(M1+M2)
    Li_sq = inner(Li_v, Li_v)
    ei_sq = inner(ei_v, ei_v)
    ai = Li_sq /(GMt*mu_i**2.*(1.-ei_sq))
    
    omega_orb = np.sqrt(GMt/ai**3.)
    P_orb_day = 2.*np.pi/omega_orb/86400.

    resi = (P_orb_day - P_orb_day_th)

    return resi


@njit
def mig_cir(t_nat, y_nat_vect, par, 
            P_orb_day_th = 10, ei_th = 0.1):
    # parse parameters
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    Lo_nat_x, Lo_nat_y, Lo_nat_z, eo_x, eo_y, eo_z, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat, \
    M2_nat, R2_nat \
                = y_nat_vect
    
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    # convert to cgs units
    M2 = M2_nat * M2_unit
    Li_v = np.array([Li_nat_x, Li_nat_y, Li_nat_z]) * Li_unit
    ei_v = np.array([ei_x, ei_y, ei_z])
    
    # scalar quantities 
    GMt = G*(M1+M2)
    mu_i = M1*M2/(M1+M2)
    Li_sq = inner(Li_v, Li_v)
    ei_sq = inner(ei_v, ei_v)
    ai = Li_sq /(GMt*mu_i**2.*(1.-ei_sq))
    
    omega_orb = np.sqrt(GMt/ai**3.)
    P_orb_day = 2.*np.pi/omega_orb/86400.

#    resi = P_orb_day - 10.
    
    if (P_orb_day < P_orb_day_th) and (ei_sq < ei_th**2.):
        resi = -1
    else:
        resi = 1.

    return resi


@njit
def no_mig(t_nat, y_nat_vect, par, 
           ai_0, dlogai_th=0.1, t_th=6.e7*P_yr):
    # parse parameters
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    Lo_nat_x, Lo_nat_y, Lo_nat_z, eo_x, eo_y, eo_z, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat, \
    M2_nat, R2_nat \
                = y_nat_vect
    
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    # convert to cgs units
    M2 = M2_nat * M2_unit
    Li_v = np.array([Li_nat_x, Li_nat_y, Li_nat_z]) * Li_unit
    ei_v = np.array([ei_x, ei_y, ei_z])
    
    # scalar quantities 
    GMt = G*(M1+M2)
    mu_i = M1*M2/(M1+M2)
    Li_sq = inner(Li_v, Li_v)
    ei_sq = inner(ei_v, ei_v)
    ai = Li_sq /(GMt*mu_i**2.*(1.-ei_sq))
    
    tt = t_nat * t_unit
    
    
    if (tt>t_th) and ((1. - ai/ai_0)<dlogai_th):
        resi = -1
    else:
        resi = 1.

    return resi


@njit
def WJ_formed(t_nat, y_nat_vect, par):
    # parse parameters
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    Lo_nat_x, Lo_nat_y, Lo_nat_z, eo_x, eo_y, eo_z, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat, \
    M2_nat, R2_nat \
                = y_nat_vect
    
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    # convert to cgs units
    M2 = M2_nat * M2_unit
    Li_v = np.array([Li_nat_x, Li_nat_y, Li_nat_z]) * Li_unit
    ei_v = np.array([ei_x, ei_y, ei_z])
    
    # scalar quantities 
    GMt = G*(M1+M2)
    mu_i = M1*M2/(M1+M2)
    Li_sq = inner(Li_v, Li_v)
    ei_sq = inner(ei_v, ei_v)
    ai = Li_sq /(GMt*mu_i**2.*(1.-ei_sq))
    
    omega_orb = np.sqrt(GMt/ai**3.)
    P_orb_day = 2.*np.pi/omega_orb/86400.
    
    resi = P_orb_day - 200.

    return resi

@njit
def HJ_formed(t_nat, y_nat_vect, par):
    # parse parameters
    Li_nat_x, Li_nat_y, Li_nat_z, ei_x, ei_y, ei_z, \
    Lo_nat_x, Lo_nat_y, Lo_nat_z, eo_x, eo_y, eo_z, \
    S1_nat_x, S1_nat_y, S1_nat_z, \
    S2_nat, \
    M2_nat, R2_nat \
                = y_nat_vect
    
    M1, M3, ao,\
    R1, I1, I2, Krot1, Krot2, k2, tauST2, \
    al_MB, \
    t_unit, Li_unit, Lo_unit, ai_unit, \
    M2_unit, R2_unit, S1_unit, S2_unit, \
    br_flag, oct_flag\
                = par
    
    # convert to cgs units
    M2 = M2_nat * M2_unit
    Li_v = np.array([Li_nat_x, Li_nat_y, Li_nat_z]) * Li_unit
    ei_v = np.array([ei_x, ei_y, ei_z])
    
    # scalar quantities 
    GMt = G*(M1+M2)
    mu_i = M1*M2/(M1+M2)
    Li_sq = inner(Li_v, Li_v)
    ei_sq = inner(ei_v, ei_v)
    ai = Li_sq /(GMt*mu_i**2.*(1.-ei_sq))
    
    omega_orb = np.sqrt(GMt/ai**3.)
    P_orb_day = 2.*np.pi/omega_orb/86400.
    
    resi = P_orb_day - 10.

    return resi


##########################################################
###             basic functions                        ###
##########################################################

@njit
def inner(xx, yy):
    return np.sum(xx*yy)

@njit
def cross(xx, yy):
    zz=np.array([\
         xx[1]*yy[2] - xx[2]*yy[1], \
         xx[2]*yy[0] - xx[0]*yy[2], \
         xx[0]*yy[1] - xx[1]*yy[0]
                ])
    return zz

@njit
def rot_z(al, vv_in):
    ca = np.cos(al)
    sa = np.sin(al)
    
    vv_out = np.array([
        ca * vv_in[0] - sa * vv_in[1], 
        sa * vv_in[0] + ca * vv_in[1], 
        vv_in[2]
    ])
    return vv_out

@njit
def construct_cord_zx(z_new_unnorm, xz_new_unnorm,  
                   m_thres=1e-16, p_thres=1e-8):
    z_new = z_new_unnorm / np.sqrt(inner(z_new_unnorm, z_new_unnorm))

    xz_new_mag = np.sqrt(inner(xz_new_unnorm, xz_new_unnorm))
    
    if xz_new_mag < m_thres:
        xz_new = np.array([1., 0., 0.])
    else:
        xz_new = xz_new_unnorm / xz_new_mag
    
    cth = inner(z_new, xz_new)
    sth = np.sqrt(1.-cth**2)
    
    if sth < p_thres:
        # z and xz are parallel
        xz_new = np.array([1., 0., 0.])
        cth = inner(z_new, xz_new)
        sth = np.sqrt(1.-cth**2)
        
        # z is further along [1, 0, 0]
        if sth < p_thres:
            xz_new = np.array([0., 1., 0.])
            
    y_new = cross(z_new, xz_new)
    y_new = y_new / np.sqrt(inner(y_new, y_new))
    
    x_new = cross(y_new, z_new)
    x_new = x_new / np.sqrt(inner(x_new, x_new))
    return x_new, y_new, z_new

@njit
def construct_cord_zy(z_new_unnorm, yz_new_unnorm,  
                   m_thres=1e-16, p_thres=1e-8):
    z_new = z_new_unnorm / np.sqrt(inner(z_new_unnorm, z_new_unnorm))

    yz_new_mag = np.sqrt(inner(yz_new_unnorm, yz_new_unnorm))
    
    if yz_new_mag < m_thres:
        yz_new = np.array([0., 1., 0.])
    else:
        yz_new = yz_new_unnorm / yz_new_mag
    
    cth = inner(z_new, yz_new)
    sth = np.sqrt(1.-cth**2)
    
    if sth < p_thres:
        # z and yz are parallel
        yz_new = np.array([0., 1., 0.])
        cth = inner(z_new, yz_new)
        sth = np.sqrt(1.-cth**2)
        
        # z is further along [0, 1, 0]
        if sth < p_thres:
            yz_new = np.array([1., 0., 0.])
            
    x_new = cross(yz_new, z_new)
    x_new = x_new / np.sqrt(inner(x_new, x_new))
    
    y_new = cross(z_new, x_new)
    y_new = y_new / np.sqrt(inner(y_new, y_new))
    return x_new, y_new, z_new

@njit
def construct_cord(z_new_unnorm, sec_new_unnorm, 
                   use_zx=True, m_thres=1e-16, p_thres=1e-8):
    if use_zx:
        x_new, y_new, z_new = construct_cord_zx(z_new_unnorm, sec_new_unnorm,  
                               m_thres=m_thres, p_thres=p_thres)
    else:
        x_new, y_new, z_new = construct_cord_zy(z_new_unnorm, sec_new_unnorm,  
                               m_thres=m_thres, p_thres=p_thres)
    return x_new, y_new, z_new
    

@njit
def proj_new_cord(vv, xx, yy, zz):
    vv_new = np.array([
        inner(vv, xx), 
        inner(vv, yy), 
        inner(vv, zz)
    ])
    return vv_new