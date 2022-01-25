 #-*- coding: utf-8 -*-
"""
Created on 24/01/2022

@author: Peter Donnel
"""

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jn

#Physical constants
charge = 1.602 * 10**(-19)   #C
mass = 9.109 * 10**(-31)     #kg
light_speed = 299792458      #m/s
epsilon0 = 8.854 * 10**(-12) #F.m^{-1}

# Plasma inputs
Ne = 1.0 * 10**(19)                  #m^{-3}
Te = 1.0 * 10**4 * 1.602 * 10**(-19) #J Boltzmann constant included
B = 1.4                              #T
Radius_loc = 1.0                          #m

# Beam inputs
harmonic = 2
omega_b = 7.6 * 10**10 * 2 * np.pi  #Hz
theta0 = np.pi/2    
W0 = 0.02 #m
Power = 1                        #W

# Numerical imput data
vmax = 3
Nv = 40

# Vectors
Vpar = np.linspace(-vmax,vmax,2*Nv)
Vperp = np.linspace(0.,vmax,Nv)


# Compute P of Stix
def compute_P(omega_b_loc, N_loc):
    omega_p_loc = np.sqrt(N_loc * charge**2 / (epsilon0 * mass))
    return 1 - (omega_p_loc/omega_b_loc)**2


# Compute refractive index
def compute_N(theta_loc, P_loc, omega_ce_loc, omega_b_loc):
    normalized_freq = omega_ce_loc / omega_b_loc
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
def compute_theta_res(lambda_loc, P_loc, Omegace_on_omegab_loc):
    lambda2 = lambda_loc**2
    tmp1 = P_loc**2 * (2*lambda2 - P_loc) + \
           Omegace_on_omegab_loc**2 * (P_loc * (1 - lambda2)**2 - lambda2**2)
    tmp2 = Omegace_on_omegab_loc * (1 - P_loc) * lambda2 * \
           np.sqrt((Omegace_on_omegab_loc * (1 - lambda2))**2 + 4 * P_loc * lambda2)
    tmp3 =  P_loc * (P_loc**2 - Omegace_on_omegab_loc**2) + \
            lambda2 * Omegace_on_omegab_loc**2 * (P_loc - 1)
    x_lambda = (tmp1 + tmp2) / tmp3

    if ( lambda_loc>0 ):
        return 0.5*np.arccos(x_lambda)
    else:
        return np.pi - 0.5*np.arccos(x_lambda)


# Function to compute Theta_n
def compute_Theta2_n(rho, theta, Ntheta, P_loc, omega_ce_loc, omega_b_loc, vpar, vperp):
    normalized_freq = omega_ce_loc / omega_b_loc
    R_loc = (P_loc - normalized_freq) / (1 - normalized_freq)
    L_loc = (P_loc + normalized_freq) / (1 + normalized_freq)
    S_loc = 0.5 * (R_loc + L_loc)
    T_loc = 0.5 * (R_loc - L_loc)
    # To avoid numerical divergence, we treat analytically the case vperp = 0
    if (abs(vperp) < 0.01):
        return 0.
    else:
        tmp1 = (1 + T_loc / (S_loc - Ntheta**2)) * jn(harmonic+1,rho)
        tmp2 = (1 - T_loc / (S_loc - Ntheta**2)) * jn(harmonic-1,rho)
        tmp3 = - 2 * Ntheta**2 * np.cos(theta) * np.sin(theta) * vpar * jn(harmonic,rho) / \
               (P_loc - (Ntheta*np.sin(theta))**2) / vperp
        tmp4 = 4 * (1 + (T_loc / (S_loc - Ntheta**2))**2 + \
                    (Ntheta**2 * np.cos(theta) * np.sin(theta) / (P_loc - (Ntheta*np.sin(theta))**2))**2)
        return (tmp1 + tmp2 + tmp3)**2 / tmp4

# Compute the electric field given the power of the beam
def compute_E2(Power_loc, R_loc, theta_loc, N_theta_loc, omega_p_loc, Omega_ce_loc, omega_b_loc):
    omega2_p = omega_p_loc**2
    Omega2_ce = Omega_ce_loc**2
    omega2_b = omega_b_loc**2
    tan2_theta = np.tan(theta_loc)**2
    f2 = Omega2_ce * tan2_theta**2 * omega2_b + 4 * (omega2_b - omega2_p)**2 * (tan2_theta + 1)
    f3 = 2 * ((omega2_b - omega2_p - Omega2_ce) * omega2_b * \
              (tan2_theta + 1) + omega2_p * Omega2_ce)
    g1 = 2 * (tan2_theta + 1) * omega2_b * (2 * (omega2_b - omega2_p) - Omega2_ce)
    g2 = 4 * (tan2_theta + 1) * (omega2_b**2 - omega2_p**2)
    g3 = 2 * ((tan2_theta + 1) * omega2_b**2 - omega2_p*Omega2_ce)
    vg = light_speed * N_theta_loc * f3 / \
         (g1 - Omega_ce_loc*omega2_p*g2/(2*omega_b_loc*np.sqrt(f2)) - N_theta_loc**2 * g3)
    return Power_loc / (np.pi**1.5 * R_loc * W0 * epsilon0 * np.sin(theta_loc)**3 * abs(vg))

# Compute quantities depending only on space
Omega_ce = charge * B / mass
omega_p = np.sqrt(Ne * charge**2 / (epsilon0 * mass))
vT_on_c = np.sqrt(Te/mass) / light_speed
P_loc = 1 - (omega_p/omega_b)**2
N0 = compute_N(theta0, P_loc, Omega_ce, omega_b)
sigma = light_speed / (omega_b * N0 * W0)
Nlim = compute_N(0., P_loc, Omega_ce, omega_b)
E2 = compute_E2(Power, Radius_loc, theta0, N0, omega_p, Omega_ce, omega_b) 

# Compute the resonant diffusion coefficient
Dn = np.zeros((len(Vpar),len(Vperp)))

for i in range(len(Vpar)):  
    for j in range(len(Vperp)):
        lorentz = 1 / np.sqrt(1 - (Vpar[i]**2 + Vperp[j]**2) * vT_on_c**2)
        lambda_phys = (1 - harmonic * Omega_ce/omega_b / lorentz) / (Vpar[i] * vT_on_c)
        if (abs(lambda_phys)>Nlim) :
            Dn[i,j] = 0
        else:
            theta_res = compute_theta_res(lambda_phys,P_loc,Omega_ce/omega_b)
            Nres = compute_N(theta_res, P_loc, Omega_ce, omega_b)
            rho = np.sin(theta0) * N0 * omega_b * Vperp[j] * vT_on_c * lorentz / Omega_ce
            Theta2_n = compute_Theta2_n(rho, theta0, N0, P_loc, Omega_ce, omega_b, Vpar[i], Vperp[j])
            Dn[i,j] = np.sqrt(np.pi) * charge**2 * N0 / \
                      (2 * mass**2 * omega_b * sigma * abs(Vpar[i]) * vT_on_c) * \
                      np.exp(-((theta_res - theta0)/sigma)**2) * Theta2_n

#Plot
fig0 = plt.figure(0,figsize=(12, 6))
ax0 = fig0.add_subplot(111)
plt.pcolor(Vpar, Vperp, np.transpose(Dn) / (Omega_ce * Te / mass))
ax0.set_xlabel("$v_{\parallel}$", fontsize = 20)
ax0.set_ylabel("$v_{\perp}$", fontsize = 20)
ax0.set_title("$\\frac{D_{n}}{v_{Te}^2 \Omega_{ce}}$", fontsize = 20) 
ax0.set_aspect('equal','box')
plt.colorbar()

plt.show()


