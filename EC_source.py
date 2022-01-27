 #-*- coding: utf-8 -*-
"""
@author: Peter Donnel
"""

# Import useful libraries
import numpy as np
from scipy.special import jn

#Physical constants
charge = 1.602 * 10**(-19)   #C
mass = 9.109 * 10**(-31)     #kg
light_speed = 299792458      #m/s
epsilon0 = 8.854 * 10**(-12) #F.m^{-1}

# Plasma inputs
Ne0 = 1.0 * 10**(19)                  #m^{-3}
Te0 = 2.0 * 10**3 * 1.602 * 10**(-19) #J Boltzmann constant included
B0 = 1.4                              #T
R0 = 1.0                              #m
a0 = 0.25                             #m

# Beam inputs
harmonic = 2
theta_in = np.pi/2    
omega_b = 7.6 * 10**10 * 2 * np.pi    #Hz
W0 = 0.02                             #m
Power_in = 1                          #W

# Numerical imput data
vmax = 4
Nv = 200
Nr = 200

# Vectors
Vpar = np.linspace(-vmax,vmax,2*Nv)
Vperp = np.linspace(0.,vmax,Nv)
vec_R = np.linspace(R0-a0,R0+a0,Nr)

# density and temperature profiles (assume a parabolic profiles)
vec_Ne = np.zeros(Nr)
vec_Te = np.zeros(Nr)
for iR in range(Nr):
    R_loc = vec_R[iR]
    vec_Ne[iR] = Ne0 * (1 - 0.9 * ((R_loc - R0) / a0)**2)
    vec_Te[iR] = Te0 * (1 - 0.9 * ((R_loc - R0) / a0)**2)


# Usefull quantities for integration
dR = vec_R[2] - vec_R[1]
dVperp = Vperp[2] -Vperp[1]
dVpar =  Vpar[2] -Vpar[1]

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
    # Compute analytically the result for really low lambda_loc
    if ( abs(lambda_loc) < 10**(-15)):
        return 0.5*np.pi
    else:
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
                    (Ntheta**2 * np.cos(theta) * np.sin(theta) / \
                     (P_loc - (Ntheta*np.sin(theta))**2))**2)
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

# Compute quantities at the entry in the plasma (=outer midplane)
Ne_in = vec_Ne[Nr-1]
Te_in = vec_Te[Nr-1]
R_in = vec_R[Nr-1]
B_in = B0 * R0 / R_in
Omega_ce_in = charge * B_in / mass
omega_p_in = np.sqrt(Ne_in * charge**2 / (epsilon0 * mass))
P_in = 1 - (omega_p_in/omega_b)**2
N_in = compute_N(theta_in, P_in, Omega_ce_in, omega_p_in)


# Compute quantities on the magnetic axis.
Omega_ce0 = charge * B0 / mass
# Compute the minimum / maximum absorption major radii
vT_on_c_max = np.sqrt(max(vec_Te)/mass) / light_speed
Npar_in = N_in*np.cos(theta_in)
R_res_max = np.sqrt((Npar_in*R_in)**2 + (harmonic * Omega_ce0 * R0 / omega_b)**2)
R_eff_min = - vmax * vT_on_c_max * abs(Npar_in) * R_in + \
            harmonic * Omega_ce0 * R0 * np.sqrt(1 - (vmax*vT_on_c_max)**2) / omega_b
R_eff_max = + vmax * vT_on_c_max * abs(Npar_in) * R_in + \
            harmonic * Omega_ce0 * R0 * np.sqrt(1 - (vmax*vT_on_c_max)**2) / omega_b
print(R_res_max,R_eff_max,R_eff_min)
# Allocation of the empty arrays
vec_Power = np.zeros(Nr)
vec_Power[Nr-1] = Power_in

Dn = np.zeros((Nr,2*Nv,Nv))
for iR in range(Nr-2,-1,-1):
    Power_absorbed = 0.
    R_loc = vec_R[iR] 
    if R_loc < max(R_res_max,R_eff_max) and R_loc > R_eff_min:
        Ne_loc = vec_Ne[iR]
        Te_loc = vec_Te[iR]
        B_loc = B0 * R0 / R_loc
        Omega_ce_loc = charge * B_loc / mass
        omega_p_loc = np.sqrt(Ne_loc * charge**2 / (epsilon0 * mass))
        P_loc = 1 - (omega_p_loc/omega_b)**2
        vT_on_c_loc = np.sqrt(Te_loc/mass) / light_speed
        arg_theta0 = R_in / R_loc  * N_in * np.cos(theta_in) 
        theta0_loc = compute_theta_res(arg_theta0,P_loc, Omega_ce_loc/omega_b)
        N0_loc = compute_N(theta0_loc, P_loc, Omega_ce_loc, omega_b)
        sigma_loc = light_speed / (omega_b * N0_loc * W0)
        Nlim_loc = compute_N(0., P_loc, Omega_ce_loc, omega_b)
        Power_loc = vec_Power[iR+1]
        E2_loc = compute_E2(Power_loc, R_loc, theta0_loc, N0_loc, omega_p_loc, \
                            Omega_ce_loc, omega_b)

        # Compute the resonant diffusion coefficient
        for ivpar in range(2*Nv):  
            for ivperp in range(Nv):
                lorentz = 1 / np.sqrt(1 - (Vpar[ivpar]**2 + Vperp[ivperp]**2) * vT_on_c_loc**2)
                lambda_phys = (1 - harmonic * Omega_ce_loc/omega_b / lorentz) / \
                              (Vpar[ivpar] * vT_on_c_loc)
                if (abs(lambda_phys)>Nlim_loc) :
                    Dn[iR, ivpar,ivperp] = 0
                else:
                    theta_res = compute_theta_res(lambda_phys,P_loc,Omega_ce_loc/omega_b)
                    if abs(theta_res - theta0_loc) < 5*sigma_loc:
                        Nres = compute_N(theta_res, P_loc, Omega_ce_loc, omega_b)
                        rho = np.sin(theta0_loc) * N0_loc * omega_b * Vperp[ivperp] * \
                              vT_on_c_loc * lorentz / Omega_ce_loc
                        Theta2_n = compute_Theta2_n(rho, theta0_loc, N0_loc, P_loc, Omega_ce_loc, \
                                                    omega_b, Vpar[ivpar], Vperp[ivperp])
                        Dn[iR, ivpar,ivperp] = np.sqrt(np.pi) * charge**2 * N0_loc / \
                                               (2 * mass**2 * omega_b * sigma_loc * \
                                                abs(Vpar[ivpar]) * vT_on_c_loc) * Theta2_n * \
                                               np.exp(-((theta_res - theta0_loc)/sigma_loc)**2)

                        Power_absorbed += Vperp[ivperp]**3 * Dn[iR, ivpar,ivperp] * \
                                          np.exp(- (Vpar[ivpar]**2 + Vperp[ivperp]**2)/2)
                
        # Normalisation of the Power absorbed
        Power_absorbed *= Ne_loc * mass * dVpar * dVperp * dR * R_loc * np.sqrt(2 * np.pi)

    # Fill the power vector
    vec_Power[iR] = vec_Power[iR+1] - Power_absorbed
    print("Power absorbed", Power_absorbed)


# Save arrays in prevision of their exploitation
np.save('vec_R.npy', vec_R)
np.save('vec_Ne.npy', vec_Ne)
np.save('vec_Te.npy', vec_Te)
np.save('Vpar.npy', Vpar)
np.save('Vperp.npy', Vperp)
np.save('vec_Power.npy', vec_Power)
np.save('Dn.npy', Dn)



