#!/usr/bin/env python

"""
Reproducing fig. 2 of 2406.00187.
Note that because octLK is chaotic, the final result may differ from machine to machine.

Copyright (C) 2024,  Hang Yu (hang.yu2@montana.edu)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.interpolate as interp
import scipy.integrate as integ
import h5py as h5
import argparse, os

plt.rc('figure', figsize=(9, 7))
plt.rcParams.update({'text.usetex': False,
                     'mathtext.fontset': 'cm',
                     'lines.linewidth': 2.5,
                     'font.size': 20,
                     'xtick.labelsize': 'large',
                     'ytick.labelsize': 'large',
                     'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'xtick.top': True,
                     'ytick.right': True,
                     'xtick.major.width': 1.7,
                     'ytick.major.width': 1.7,
                     'xtick.major.size': 7.,
                     'ytick.major.size': 7.,
                     'axes.labelsize': 'large',
                     'axes.titlesize': 'large',
                     'axes.grid': False,
                     'grid.alpha': 0.5,
                     'lines.markersize': 12,
                     'legend.borderpad': 0.2,
                     'legend.fancybox': True,
                     'legend.fontsize': 17,
                     'legend.framealpha': 0.7,
                     'legend.handletextpad': 0.5,
                     'legend.labelspacing': 0.2,
                     'legend.loc': 'best',
                     'savefig.bbox': 'tight',
                     'savefig.pad_inches': 0.05,
                     'savefig.dpi': 80,
                     'pdf.compression': 9})

from myConstants import *
import LKlibHJ as LK



###
###

M2_ob = 0.12 * Mj
R2_ob = 0.94 * Rj
ai_ob = 0.055 * AU
L1_Ls = 0.132

atol = 1e-7
rtol = 1e-7
tau_d = 0.5 * 1e6 * P_yr

t_tot = 600e6*P_yr


def plot_traj_from_file(fid, grp_id, fig_file):
    
    fid = h5.File(fid, 'r')
    grp = fid[grp_id]
    
    for key in grp.attrs.keys():
        print(key, grp.attrs[key])
        
    M2_0_Mj = grp.attrs['M2_0_Mj'] 
    R2_0_Rj = grp.attrs['R2_0_Rj']
    
    M2_0 = M2_0_Mj * Mj
    R2_0 = R2_0_Rj * Rj
    
    M3 = grp.attrs['M3_Mj'] * Mj
    ai_0 = grp.attrs['ai_0_AU'] * AU
        
    AAA = grp.attrs['AAA']
    BBB = grp.attrs['BBB']
    
    cons_MT = grp.attrs['cons_MT']
    tau_r = grp.attrs['tau_r_Myr'] * 1e6 * P_yr
    tau_d = grp.attrs['tau_d_Myr'] * 1e6 * P_yr
    dlogR2_adj = grp.attrs['dlogR2_adj']
    
    par_LK = grp['par_LK'][()]
    y_nat_init = grp['y_nat_init'][()]
    
    M1, M3, ao = par_LK[0:3]
    t_unit, Li_unit, Lo_unit, ai_unit, M2_unit, R2_unit  = par_LK[11:17]
    
    fid.close()
    
    
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
                               P_orb_day_th = 5.72, ei_th = 0.1)
    mig_cir_event.direction = -1
    mig_cir_event.terminal = True
    
    no_mig_event = lambda t_nat_, y_nat_vect_:\
                    LK.no_mig(t_nat_, y_nat_vect_, par_LK, ai_0, 
                              dlogai_th=0.05, t_th=30.*1.e6*P_yr)
    no_mig_event.direction = -1
    no_mig_event.terminal = True
    
    
    # 
    int_func=lambda t_nat_, y_nat_vect_:\
    LK.evol_LK_da_fix_S2_R2_w_MT_R_infl(t_nat_, y_nat_vect_, par_LK, 
                                 AAA=AAA, BBB=BBB, 
                                 cons_MT = cons_MT, 
                                 L1_Ls = L1_Ls, 
                                 tau_r = tau_r, tau_d=tau_d, 
                                 dlogR2_adj=dlogR2_adj
                                 )
    
    sol=integ.solve_ivp(int_func, \
            t_span=(0, t_tot/t_unit), y0=y_nat_init, rtol=rtol, atol=atol, 
            events=[disruption_M2_event, ejection_event, mig_cir_event, no_mig_event])
    
    print('t_events:', sol.t_events)
    
    #
    tt = sol.t * t_unit
    
    M2 = sol.y[-2, :] * M2_unit
    print('dM2/Mj:', M2[-1]/Mj-M2_0/Mj)
    
    R2 = sol.y[-1, :] * R2_unit
    
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
    
    Lo_x = sol.y[6, :]*Lo_unit
    Lo_y = sol.y[7, :]*Lo_unit
    Lo_z = sol.y[8, :]*Lo_unit
    
    
    eo_x = sol.y[9, :]
    eo_y = sol.y[10, :]
    eo_z = sol.y[11, :]
    eo = np.sqrt(eo_x**2. + eo_y**2. + eo_z**2.)
    
    Li_e0 = Li / eff_i
    ai = Li_e0**2./(G*(M1+M2)*mu_i**2.)
    P_orb = 2.*np.pi/np.sqrt(G*(M1+M2)/ai**3.)
    
    rp = ai*(1-ei)
    
    Eorb = -G*M1*M2/2/ai
        
    L_tide, Q_eff_2 = LK.get_L_tide_Q_eff_eq(ai, ei, M2, R2, par_LK)
    print('max R2', np.max(R2)/Rj)
    print('max L_tide, %e'%(np.max(L_tide)/Ls))
    
    Rtide = (M1/M2)**(1./3)*R2
    
    R2_eq = np.zeros(len(R2))
    for i in range(len(R2)):
        R2_eq[i] = LK.get_R2_eq_T21(
                    ai[i], ei[i], M2[i], R2[i],
                    par_LK,
                    L1_Ls=L1_Ls, dlogR2_adj=dlogR2_adj)

    fig=plt.figure(figsize=(9, 33))
    ax=fig.add_subplot(611)
    ax.plot(tt/P_yr/1e6, ei, 
              color='C7', alpha=0.6)
    
#     ax.axhline(0.1, color='tab:red', ls=':')
#     ax.axhline(0.02, color='tab:red', ls=':')
    ax.set_ylabel(r'$e$')
    # ax.set_yscale('log')
    
    
    ax=fig.add_subplot(613)
    ax.plot(tt/P_yr/1e6, rp/Rtide, 
              color='C7', alpha=0.7)
    # ax.axhline(2.7, color='tab:red', ls=':')
    ax.set_ylabel(r'$r_p/r_t$')
    ax.set_yscale('log')
        
    ax=fig.add_subplot(612)
    ax.plot(tt/P_yr/1e6, P_orb/86400., 
              color='C7', alpha=0.7)
    ax.set_ylabel(r'$P$ [d]')
#     ax.axhline(5.72, color='tab:red', ls=':')
    ax.set_yscale('log')
    
    # ax=fig.add_subplot(714)
    # ax.plot(tt/P_yr/1e6, eo, color='C7', alpha=0.7)
    # ax.set_ylabel('$e_o$')
    
    
    ax=fig.add_subplot(614)
    ax.plot(tt/P_yr/1e6, M2/Mj, 
              color='C7', alpha=0.7)
#    ax.set_ylim([0.04, M2_0/Mj+0.02])
    ax.set_ylabel(r'$M_p/M_j$')
    
    
    ax=fig.add_subplot(615)
    ax.plot(tt/P_yr/1e6, R2/Rj, 
              color='C7', alpha=0.7, label=r'$R_p$')
    ax.plot(tt/P_yr/1e6, R2_eq/Rj, 
              color='C8', alpha=0.7, label=r'$R_{\rm eq}$')
    ax.legend(loc='upper left')
    ax.set_ylabel(r'$R_p/R_j$')
    
    
    ax=fig.add_subplot(616)
    ax.plot(tt/P_yr/1e6, L_tide/Ls, 
              color='C7', alpha=0.7)
    ax.set_yscale('log')
    
    ax.set_ylim(bottom=3e-8)
    ax.set_ylabel(r'$L_t/L_\odot$')
    
    
    # ax.legend(loc='lower left')
    
    ax.set_xlabel(r'$t$ [Myr]')
    plt.savefig(fig_file)
    plt.close()
    
    
if __name__=='__main__':    
    # parse some args here
    parser = argparse.ArgumentParser()
    parser.add_argument('--fid', type=str, default='sample_data.h5')
    parser.add_argument('--grp-id', type=str, default=1)
    parser.add_argument('--fig-file', type=str, default='wasp_107b_samp_2.png')
    
    kwargs = parser.parse_args()
    # convert it to a dict
    kwargs=vars(kwargs)
    
    fid = kwargs['fid']
    grp_id = '%s'%kwargs['grp_id']
    fig_file = kwargs['fig_file']
    
    plot_traj_from_file(fid, grp_id, fig_file)
    
    
    
