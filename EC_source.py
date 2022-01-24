 #-*- coding: utf-8 -*-
"""
Created on 24/01/2022

@author: Peter Donnel
"""

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf as Phi

#Physical constants
charge = 1.602 * 10**(-19)   #C
mass = 9.109 * 10**(-31)     #kg
light_speed = 299792458      #m/s
epsilon0 = 8.854 * 10**(-12) #F.m^{-1}

# Plasma inputs
Ne = 1.0 * 10**(19)                  #m^{-3}
Te = 1.0 * 10**4 * 1.602 * 10**(-19) #J Boltzmann constant included
B = 1.4                              #T

sigma = 0.1

# Beam inputs
harmonic = 2.0
omega_b = 7.6 * 10**10 * 2 * np.pi  #Hz
theta0 = np.pi/2 + 0.001   


# Numerical imput data
vmax = 5
Nv = 40

# Vectors
Vpar = np.linspace(-vmax,vmax,2*Nv)
Vperp = np.linspace(0.,vmax,Nv)

# Compute the cyclotron frequency
def compute_Omega_ce(B_loc):
    return charge * B_loc / mass


# Compute the ratio vT / c where vT = sqrt(T / me)
def compute_vt_on_c(T_loc):
    return np.sqrt(T_loc/mass) / light_speed


# Compute P of Stix
def compute_P(omega_b_loc, N_loc):
    omega_p_loc = np.sqrt(N_loc * charge**2 / (epsilon0 * mass))
    return 1 - (omega_p_loc/omega_b_loc)**2


# Compute refractive index
def compute_N(theta_loc, P_loc, omega_ce_loc, omega_b_loc):
    normalized_freq = omega_ce_loc / omega_b_loc
    print(normalized_freq)
    R_loc = (P_loc - normalized_freq) / (1 - normalized_freq)
    L_loc = (P_loc + normalized_freq) / (1 + normalized_freq)
    S_loc = 0.5 * (R_loc + L_loc)
    tmp1 = (R_loc*L_loc + S_loc*P_loc) * np.tan(theta_loc)**2 + \
           2 * P_loc * S_loc 
    tmp2 = (S_loc*P_loc - R_loc*L_loc)**2 * np.tan(theta_loc)**4 + \
           P_loc**2 * (L_loc - R_loc)**2 * (np.tan(theta_loc)**2 + 1)
    tmp3 = 2 * (S_loc * np.tan(theta_loc)**2 + P_loc)
    Nx2 = (tmp1 - np.sqrt(tmp2)) / tmp3
    return np.sqrt(Nx2)
    

# Function to compute the resonant angle
def compute_theta_res(lambda_loc, P_loc, Omega_ce_on_omegab_loc):
    lambda2 = lambda_loc**2
    tmp1 = P_loc**2 * (2*lambda2 - P_loc) + \
           Omega_ce_on_omegab_loc**2 * (P_loc * (1 - lambda2)**2 - lambda2**2)
    tmp2 = Omega_ce_on_omegab_loc * (1 - P_loc) * lambda2 * \
           np.sqrt((Omega_ce_on_omegab_loc * (1 - lambda2))**2 + 4 * P_loc * lambda2)
    tmp3 =  P_loc * (P_loc**2 - Omega_ce_on_omegab_loc**2) + \
            lambda2 * Omega_ce_on_omegab_loc**2 * (P_loc - 1)
    x_lambda = (tmp1 + tmp2) / tmp3
    
    if ( lambda_loc>0 ):
        return 0.5*np.arccos(x_lambda)
    else:
        return np.pi - 0.5*np.arccos(x_lambda)


# Compute quantities depending only on space
Omega_ce = compute_Omega_ce(B)
vT_on_c = compute_vt_on_c(Te)
P_loc = compute_P(omega_b, Ne)
Nx = compute_N(theta0, P_loc, Omega_ce, omega_b)

# Compute the resonant diffusion coefficient
Dn = np.zeros((len(Vpar),len(Vperp)))

for i in range(len(Vpar)):  
    for j in range(len(Vperp)):
        lorentz = 1 / np.sqrt(1 - (Vpar[i]**2 + Vperp[j]**2) * vT_on_c**2)
        lambda_phys = (1 - harmonic * Omega_ce/omega_b / lorentz) / (Vpar[i] * vT_on_c)
        if (abs(lambda_phys)>Nx) :
            Dn[i,j] = 0
        else:
            theta_res = compute_theta_res(lambda_phys,P_loc,Omega_ce/omega_b)
            Dn[i,j] = np.exp(-((theta_res - 0.5*np.pi)/sigma)**2)

#Plot
fig0 = plt.figure(0,figsize=(12, 9))
ax0 = fig0.add_subplot(111)
plt.pcolor(Vpar, Vperp, np.transpose(Dn))
ax0.set_xlabel("$v_{\parallel}$", fontsize = 20)
ax0.set_ylabel("$v_{\perp}$", fontsize = 20)
ax0.set_title("$D_{n}$", fontsize = 20) 
plt.colorbar()

plt.show()


