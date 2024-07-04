#!/usr/bin/env python

"""
    Source code for evolving one system under high-e migration. The planetary radius is fixed throughout. 
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
import scipy.optimize as opt
import scipy.interpolate as interp
import scipy.integrate as integ
import h5py as h5
import argparse, os

from myConstants import *
import LKlibHJ as LK

def evol_one(idx, 
             M1, M2_0, M3, 
             R2_0, ai_0, 
             ao, eo_0, 
             I_io_0, phi_eo_0, 
             out_file, 
             **kwargs):    
    
    # kwargs to be explored
    tauST2 = kwargs.pop('tauST2', 1.)    # tidal lag time for eq. tide
    
    
    AAA = kwargs.pop('AAA', 1.e3)
    BBB = kwargs.pop('BBB', 9.25)
        
    br_flag = kwargs.pop('br_flag', 1)    # kozai back-reaction on the outer orbit
    oct_flag = kwargs.pop('oct_flag', 1)    # oct term during LK?
    
    cons_MT = kwargs.pop('cons_MT', 1)
    if cons_MT>0:
        cons_MT = True
    else:
        cons_MT = False
    
    print(idx,'\n', 'ai_0:', ai_0/AU, 'R2_0', R2_0/Rj, 'M2_0:', M2_0/Mj, 'M3:', M3/Mj)
    print('I_io_0:', I_io_0*180./np.pi, 'eo_0:', eo_0, 'phi_eo_0:', phi_eo_0, 'AAA:', AAA)
    
    tau_d_Myr = kwargs.pop('tau_d_Myr', 500)
    tau_r_Myr = kwargs.pop('tau_r_Myr', 10)
    
    # kwargs fixed
    dlogR2_cut = kwargs.pop('dlogR2_cut', 2.)
    dlogR2_adj = kwargs.pop('dlogR2_adj', 0.)   # adjustment of Millholland formula
    
    L1_Ls =kwargs.pop('L1_Ls', 0.132)   # stellar luminosity in Ls
    R1 = kwargs.pop('R1', 0.66 * Rs)    # radius of the host star
    I1 = kwargs.pop('I1', 0.1)    # moment of inertia of the star in M1*R1**2
    I2 = kwargs.pop('I2', 0.25)    # moment of inertia of the planet in M2*R2**2
    Krot1 = kwargs.pop('Krot1', 0.05)    # Stellar quadrupole due to rotation is Krot1*M1*R1**2
    Krot2 = kwargs.pop('Krot2', 0.17)    # planetary quadrupole due to rotation is Krot2*M2*R2**2
    k2 = kwargs.pop('k2', 0.37)    # planetary love number
    al_MB = kwargs.pop('al_MB', 0.)    # stellar spin down coefficient, dimensional
    Ps1_day = kwargs.pop('Ps1_day', 17.1)   # stellar spin period in days
    
    ei_0 = kwargs.pop('ei_0', 0.01)
        
    t_tot_yr = kwargs.pop('t_tot_yr', 600e6)
    
    rtol = kwargs.pop('rtol', 1e-7)
    atol = kwargs.pop('atol', 1e-7)    
    
    # 
    t_tot = t_tot_yr * P_yr
    
    mu_i = M1*M2_0/(M1+M2_0)
    mu_o = (M1+M2_0)*M3/(M1+M2_0+M3)
    
    eff_i_0 = np.sqrt(1.-ei_0**2.)
    eff_o_0 = np.sqrt(1.-eo_0**2.)
    
    Li_0 = mu_i * np.sqrt(G*(M1+M2_0)*ai_0)*eff_i_0
    Lo_0 = mu_o * np.sqrt(G*(M1+M2_0+M3)*ao)*eff_o_0
    
    S1 = 2.*np.pi/Ps1_day/P_day * I1*M1*R1**2.
    S2 = 0. # it will be set to pseudo sync value in the code        
    
    #
    t_unit=1.e3 * P_yr #t_LK_0
    Li_unit, Lo_unit = Li_0, Lo_0
    ai_unit = ai_0
    M2_unit = Mj
    R2_unit = Rj
    S1_unit = np.sqrt(G*Ms/Rs**3.)*Ms*Rs**2.
    S2_unit = np.sqrt(G*Mj/Rj**3.)*Mj*Rj**2.
    
    par_LK = np.array([M1, M3, ao,
                       R1, I1, I2, Krot1, Krot2, k2, tauST2, 
                       al_MB, 
                       t_unit, Li_unit, Lo_unit, ai_unit, 
                       M2_unit, R2_unit, S1_unit, S2_unit, 
                       br_flag, oct_flag])
    
    
    #
    uLi_v = np.array([np.sin(I_io_0), 0., np.cos(I_io_0)])
    Li_nat_v = uLi_v * Li_0 /Li_unit
    ei_v = np.array([0., ei_0, 0])
    Lo_nat_v = np.array([0., 0., 1.])*Lo_0/Lo_unit
    
    eo_v = np.array([-np.sin(phi_eo_0), np.cos(phi_eo_0), 0.]) * eo_0
    
    S1_nat_v = uLi_v*S1/S1_unit
    S2_nat = S2/S2_unit
    
    M2_0_nat = M2_0/M2_unit
    R2_0_nat = R2_0/R2_unit
    
    y_nat_vect_init = np.hstack([
        Li_nat_v, ei_v, 
        Lo_nat_v, eo_v, 
        S1_nat_v, S2_nat, 
        M2_0_nat, R2_0_nat])
    
    #
    disruption_M2_event = lambda t_nat_, y_nat_vect_:\
                    LK.disruption_M2(t_nat_, y_nat_vect_, par_LK, M2_Mj_th=0.03)
    disruption_M2_event.direction = -1
    disruption_M2_event.terminal = True
    
    ejection_event = lambda t_nat_, y_nat_vect_:\
                    LK.ejection(t_nat_, y_nat_vect_, par_LK, P_orb_day_th = 200)
    ejection_event.direction = 1
    ejection_event.terminal = True
    
    mig_cir_event = lambda t_nat_, y_nat_vect_:\
                    LK.mig_cir(t_nat_, y_nat_vect_, par_LK, 
                               P_orb_day_th = 10, ei_th = 0.1)
    mig_cir_event.direction = -1
    mig_cir_event.terminal = True
    
    no_mig_event = lambda t_nat_, y_nat_vect_:\
                    LK.no_mig(t_nat_, y_nat_vect_, par_LK, ai_0, 
                              dlogai_th=0.03, t_th=30.*1.e6*P_yr)
    no_mig_event.direction = -1
    no_mig_event.terminal = True
    
    
    # 
    int_func=lambda t_nat_, y_nat_vect_:\
    LK.evol_LK_da_fix_S2_R2_w_MT(t_nat_, y_nat_vect_, par_LK, 
                                 AAA=AAA, BBB=BBB, 
                                 cons_MT = cons_MT)
    
    sol=integ.solve_ivp(int_func, \
            t_span=(0, t_tot/t_unit), y0=y_nat_vect_init, rtol=rtol, atol=atol, 
            events=[disruption_M2_event, ejection_event, mig_cir_event, no_mig_event])

    print('t_events:', sol.t_events)

    
    tt = sol.t * t_unit
    
    t_fin = tt[-1]
    t_fin_Myr = tt[-1]/P_yr/1e6
    y_nat_fin = sol.y[:, -1]
        
    #
    M2 = sol.y[-2, :]*M2_unit

    mu_i = M1*M2/(M1+M2)

    Li_x = sol.y[0, :]*Li_unit 
    Li_y = sol.y[1, :]*Li_unit 
    Li_z = sol.y[2, :]*Li_unit 
    Li = np.sqrt(Li_x**2. + Li_y**2. + Li_z**2.)
    
    ei_x = sol.y[3, :]
    ei_y = sol.y[4, :]
    ei_z = sol.y[5, :]
    ei=np.sqrt(ei_x**2. + ei_y**2. + ei_z**2.)
    
    eff_i = np.sqrt(1.-ei**2.) 
    
    eo_x = sol.y[9, :]
    eo_y = sol.y[10, :]
    eo_z = sol.y[11, :]
    eo = np.sqrt(eo_x**2. + eo_y**2. + eo_z**2.)
    
    Li_e0 = Li / eff_i
    ai = Li_e0**2./(G*(M1+M2)*mu_i**2.)
    P_orb = 2.*np.pi/np.sqrt(G*(M1+M2)/ai**3.)

    rp = ai*(1-ei)

    R2 = R2_0
    Rtide = (M1/M2)**(1./3)*R2
    
    Lum_t, Q_eff_2 = LK.get_L_tide_Q_eff_eq(ai, ei, M2, R2, par_LK)
    
    #
    M2_fin = M2[-1]
    R2_fin_Rj = R2_0/Rj
    
    P_orb_fin_d = P_orb[-1]/86400.
    ei_fin = ei[-1]
    
    dM2_Mj = (M2[-1] - M2[0]) / Mj
    min_rp_Rtide = np.min(rp/Rtide)        
    min_rp_AU = np.min(rp/AU)
    
    Lum_t_fin_Ls = Lum_t[-1] / Ls
    Lum_t_m_Ls = integ.trapezoid(Lum_t, tt)/t_fin / Ls
        
    #
    eo_fin = eo[-1]
    max_eo = np.max(eo)
    
    cross_orb = np.max( ai*(1.+ei)/(ao*(1.-eo)) )

    # outcome
    # 0 for disruption; 
    # 1 for HJ formed
    # -1 for no migration  
    
    if ((sol.t_events[0].size>0) or (sol.t_events[1].size>0)) or cross_orb > 1.:
        print('Disruption!')        
        outcome = 0
        
    elif sol.t_events[2].size>0:
        print('HJ formed!')
        outcome = 1
        
    else:
        print('No migration')
        outcome = -1   
        
    print('t_fin_Myr', t_fin_Myr, 'cons_MT', cons_MT)
    print('max_eo', max_eo, 'eo_fin', eo_fin, 'cross_orb', cross_orb)
    print('dM2_Mj', dM2_Mj, 
          'R2_fin_Rj', R2_fin_Rj,
          'min_rp_Rtide', min_rp_Rtide, 
          'Lum_t_m_Ls', Lum_t_m_Ls)
    
    #
    Li_v_fin = np.array([Li_x[-1], Li_y[-1], Li_z[-1]])
    Lo_v_fin = sol.y[6:9, -1] * Lo_unit

    Li_fin = np.sqrt(LK.inner(Li_v_fin, Li_v_fin))
    Lo_fin = np.sqrt(LK.inner(Lo_v_fin, Lo_v_fin))

    uLi_v_fin = Li_v_fin/Li_fin
    uLo_v_fin = Lo_v_fin/Lo_fin
 
    c_I_io_fin = LK.inner(uLi_v_fin, uLo_v_fin)
    
    xx, yy, zz = LK.construct_cord(uLi_v_fin, uLo_v_fin)
    phi_N = np.arange(0, 2*np.pi, 0.01)
    M3_s_No = np.zeros(len(phi_N))
    for i in range(len(phi_N)):
        _phi = phi_N[i]
        _uN_v = np.cos(_phi) * xx + np.sin(_phi) * yy
        c_No = LK.inner(_uN_v, uLo_v_fin)
        s_No = np.sqrt(1.-c_No**2.)
        M3_s_No[i] = M3 * s_No
        
        
    max_M3_s_No_Mj = np.max(M3_s_No)/Mj
    min_M3_s_No_Mj = np.min(M3_s_No)/Mj
    
    print('M3_s_No_Mj', max_M3_s_No_Mj, min_M3_s_No_Mj)
        
        
    #
    fid = h5.File(out_file, 'a')
    grp_id = '%i'%idx    
    if grp_id in fid.keys():
        del fid[grp_id]
    grp = fid.create_group(grp_id)

    grp.attrs['outcome'] = outcome        
    grp.attrs['t_fin_Myr'] = t_fin_Myr
        
        
    grp.attrs['AAA'] = AAA
    grp.attrs['BBB'] = BBB
#     grp.attrs['dlogR2_adj'] = dlogR2_adj
#     grp.attrs['dlogR2_cut'] = dlogR2_cut
    grp.attrs['cons_MT'] = cons_MT
    
    grp.attrs['M3_Mj'] = M3/Mj
    grp.attrs['eo_0'] = eo_0        
    grp.attrs['ai_0_AU'] = ai_0/AU
    grp.attrs['M2_0_Mj'] = M2_0/Mj
    grp.attrs['R2_0_Rj'] = R2_0/Rj
    grp.attrs['I_io_0'] = I_io_0

    grp.attrs['dM2_Mj'] = dM2_Mj
    grp.attrs['R2_fin_Rj'] = R2_fin_Rj
    grp.attrs['min_rp_Rtide'] = min_rp_Rtide
    grp.attrs['min_rp_AU'] = min_rp_AU
    grp.attrs['Lum_t_m_Ls'] = Lum_t_m_Ls
    grp.attrs['Lum_t_fin_Ls'] = Lum_t_fin_Ls
    grp.attrs['I_io_fin'] = np.arccos(c_I_io_fin)
    grp.attrs['P_orb_fin_d'] = P_orb_fin_d
    grp.attrs['ei_fin'] = ei_fin
    
    grp.attrs['eo_fin'] = eo_fin
    grp.attrs['max_eo'] = max_eo
    grp.attrs['cross_orb'] = cross_orb
    grp.attrs['max_M3_s_No_Mj'] = max_M3_s_No_Mj
    grp.attrs['min_M3_s_No_Mj'] = min_M3_s_No_Mj
        
    grp.create_dataset('par_LK', shape=par_LK.shape, dtype=par_LK.dtype, data=par_LK)
    grp.create_dataset('y_nat_init', shape=y_nat_vect_init.shape, dtype=y_nat_vect_init.dtype, data=y_nat_vect_init)
    grp.create_dataset('y_nat_fin', shape=y_nat_fin.shape, dtype=y_nat_fin.dtype, data=y_nat_fin)
    
    fid.close()

def precision_round(number, digits=6):
    power = "{:e}".format(number).split('e')[1]
    return round(number, -(int(power) - digits))        


if __name__=='__main__':    
    # parse some args here
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--out-file-base', type=str, default='./data/evoled_wasp_107_debug')
    parser.add_argument('--out-file-id', type=int, default=0)
    
    parser.add_argument('--idx-init', type=int, default=0)
    parser.add_argument('--idx-fin', type=int, default=5)
    
    parser.add_argument('--tauST2', type=np.float64, default=10.)
    parser.add_argument('--oct-flag', type=int, default=1)
    parser.add_argument('--cons-MT', type=int, default=0)

    
    kwargs = parser.parse_args()
    # convert it to a dict
    kwargs=vars(kwargs)
    
    seed = kwargs['seed']
    if seed<0:
        rng = np.random.default_rng(None)
    else:
        rng = np.random.default_rng(seed)
        
    idx_init = kwargs['idx_init']
    idx_fin = kwargs['idx_fin']
    
    out_file_base = kwargs['out_file_base']
    out_file_id = kwargs['out_file_id']
    
    out_file = out_file_base + '%i'%out_file_id + '.h5'
    
    tauST2 = kwargs['tauST2']
    oct_flag = kwargs['oct_flag']
    
    cons_MT = kwargs['cons_MT']
    
    M1 = 0.69 * Ms
    ao = 1.84 * AU

    ai_f = 0.055*AU
    
    run_idx = np.arange(idx_init, idx_fin, 1)
    n_run = len(run_idx)
    
    for i in range(n_run):
        idx = run_idx[i]
        
        M2_0_Mj = rng.uniform(0.1, 0.5)
        M2_0_Mj = precision_round(M2_0_Mj)
        
        R2_0_Rj = rng.uniform(0.6, 1.5)
        R2_0_Rj = precision_round(R2_0_Rj)
        
        ai_0_AU = rng.uniform(0.15, 0.5)
        ai_0_AU = precision_round(ai_0_AU)
        
        M3_Mj = rng.uniform(.36, 1.5)
        M3_Mj = precision_round(M3_Mj)
        
        M3 = M3_Mj * Mj

        # debugging
#         c_I_io_0 = rng.uniform(np.cos(100.*np.pi/180.), np.cos(80.*np.pi/180.))
#         # full
        c_I_io_0 = rng.uniform(-0.9, 0.75)
        
        I_io_0 = np.arccos(c_I_io_0)
        I_io_0 = precision_round(I_io_0)
        
        if oct_flag:
            eo_0 = rng.uniform(0.2, 0.45)
            eo_0 = precision_round(eo_0)
        else:
            eo_0 = 0.28 # note that at quad order, |eo| stays constant as eo_v dot deo_v/dt = 0
            
        phi_eo_0 = rng.uniform(0, 2.*np.pi)
        phi_eo_0 = precision_round(phi_eo_0)

        AAA = 10.**rng.uniform(2.8, 5.)
        AAA = precision_round(AAA)
        
        evol_one(idx, 
             M1, M2_0_Mj*Mj, M3, 
             R2_0_Rj*Rj, ai_0_AU*AU, 
             ao, eo_0, 
             I_io_0, phi_eo_0, 
             out_file, 
             tauST2=tauST2, 
             oct_flag=oct_flag, 
             AAA = AAA,
             cons_MT=cons_MT
             )
