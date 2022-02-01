 #-*- coding: utf-8 -*-
"""
@author: Peter Donnel
"""

# Import useful libraries
import numpy as np
import math
from scipy.special import jn
from scipy.special import spherical_jn


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
omega_b = 7.8 * 10**10 * 2 * np.pi    #Hz
W0 = 0.02                             #m
Power_in = 1                          #W


# Numerical imput data
vmax = 4
Nv = 300
Nr = 400


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


# Compute gm(zn) and its derivatives
def Compute_Gm_derivatives(zn, m):
    if (m==0):
        Gm = np.sinh(zn) / zn
        dGm_dx = - np.sinh(zn) / zn**2 + np.cosh(zn) / zn
        d2Gm_dx2 = (2/zn**3 + 1/zn) * np.sinh(zn) - 2 * np.cosh(zn) / zn**2
    elif (m==1):
        Gm = - np.sinh(zn) / zn**2 + np.cosh(zn) / zn
        dGm_dx = (2/zn**3 + 1/zn) * np.sinh(zn) - 2 * np.cosh(zn) / zn**2
        d2Gm_dx2 = - (6/zn**4 + 3/zn**2) * np.sinh(zn) + (6/zn**3 + 1/zn) * np.cosh(zn)
    elif (m==2):
        Gm = (3/zn**3 + 1/zn) * np.sinh(zn) - 3 * np.cosh(zn) / zn**2
        dGm_dx = - (9/zn**4 + 4/zn**2) * np.sinh(zn) + (9/zn**3 + 1/zn) * np.cosh(zn)
        d2Gm_dx2 = (36/zn**5 + 17/zn**3 + 1/zn) * np.sinh(zn) - \
                   (36/zn**4 + 5/zn**2) * np.cosh(zn)
    elif (m==3):
        Gm = - (15/zn**4 + 6/zn**2) *  np.sinh(zn) + (15/zn**3 + 1/zn) * np.cosh(zn)
        dGm_dx = (60/zn**5 + 27/zn**3 + 1/zn) * np.sinh(zn) - \
                 (60/zn**4 + 7/zn**2) * np.cosh(zn)
        d2Gm_dx2 = - (300/zn**6 + 141/zn**4 + 8/zn**2) * np.sinh(zn) + \
                   (300/zn**5 + 41/zn**3 + 1/zn) * np.cosh(zn)
    elif (m==4):
        Gm = (105/zn**5 + 45/zn**3 + 1/zn) *  np.sinh(zn) - (105/zn**4 + 10/zn**2) * np.cosh(zn)
        dGm_dx = - (525/zn**6 + 240/zn**4 + 11/zn**2) * np.sinh(zn) + \
                 (525/zn**5 + 65/zn**3 + 1/zn) * np.cosh(zn)
        d2Gm_dx2 = (3150/zn**7 + 1485/zn**5 + 87/zn**3 + 1/zn) * np.sinh(zn) - \
                   (3150/zn**6 + 435/zn**4 + 12/zn**2) * np.cosh(zn)
    else:
        print('harmonic not handled:', m)
    return Gm, dGm_dx, d2Gm_dx2


# Compute derivatives of the |J_{m+1/2}(zn)|^2 / xn
def Compute_usefull_derivatives(xn, yn, m):
    if (4 * xn**2 > yn**2):
        zn = 0.5 * complex(np.sqrt(4 * xn**2 - yn**2), yn)
        zn_bar = zn.conjugate()
        jm_z = spherical_jn(m,zn)
        jm1_z = spherical_jn(m-1,zn)
        jm2_z = spherical_jn(m-2,zn)
        jm_z_bar = spherical_jn(m,zn_bar)
        jm1_z_bar = spherical_jn(m-1,zn_bar)
        jm1_jmbar = jm1_z * jm_z_bar
        z_jm1_jmbar = zn * jm1_z * jm_z_bar
        z_jm2_jmbar = zn * jm2_z * jm_z_bar
        z2_jm2_jmbar = zn**2 * jm2_z * jm_z_bar
        tmp = (1 - m - zn**2 * (m+1) / xn**2) * jm1_z * jm_z_bar

        fx = 2 * np.absolute(jm_z)**2 / np.pi
        df_dx = - 4 * (m+1) * np.absolute(jm_z)**2 / (np.pi * xn) + \
                4 * xn * jm1_jmbar.real / (np.pi * zn.real)
        df_dy = - 2 * z_jm1_jmbar.imag / (np.pi * zn.real)
        df_dydy = - yn * z_jm1_jmbar.imag / (2 * np.pi * zn.real**3) + \
                  xn**2 * np.absolute(jm1_z)**2 / (np.pi * zn.real**2) - \
                  (2 * z_jm1_jmbar.real + z2_jm2_jmbar.real) / (np.pi * zn.real**2)
        df_dxdy = - (2 * xn / (np.pi * zn.real**2)) * \
                  (- z_jm1_jmbar.imag / zn.real + tmp.imag + z_jm2_jmbar.imag + \
                   zn.imag*np.absolute(jm1_z)**2)
        fx = fx.real
        df_dx = df_dx.real
        df_dy = df_dy.real
        df_dydy = df_dydy.real
        df_dxdy = df_dxdy.real
    else:
        zn_p = 0.5 * (yn  + np.sqrt(yn**2 - 4 * xn**2))
        zn_m = 0.5 * (yn  - np.sqrt(yn**2 - 4 * xn**2))
        [Gm_p, dGm_dx_p, d2Gm_dx2_p] = Compute_Gm_derivatives(zn_p, m)
        [Gm_m, dGm_dx_m, d2Gm_dx2_m] = Compute_Gm_derivatives(zn_m, m)
        fx = 2 * Gm_p * Gm_m / np.pi
        df_dx = 4 * xn * (dGm_dx_m*Gm_p - dGm_dx_p*Gm_m) / (np.pi * (zn_p-zn_m))
        df_dy = 2 * (zn_p*dGm_dx_p*Gm_m - zn_m*dGm_dx_m*Gm_p) / (np.pi * (zn_p-zn_m))
        df_dydy = 2 * (zn_p**2*d2Gm_dx2_p*Gm_m - zn_m**2*d2Gm_dx2_m*Gm_p - \
                       2*xn**2*dGm_dx_p*dGm_dx_m) / (np.pi * (zn_p-zn_m)**2)
        df_dxdy = 4 * xn / (np.pi * (zn_p-zn_m)**2) * \
                  (dGm_dx_p*Gm_m - dGm_dx_m*Gm_p + yn*dGm_dx_p*dGm_dx_m - \
                   zn_p*d2Gm_dx2_p*Gm_m - zn_m*d2Gm_dx2_m*Gm_p) 
        
    return fx, df_dx, df_dy, df_dydy, df_dxdy


# Compute BigA and BigB
def Compute_A_B(omega_b, omega_p, Omega_ce, N0, theta, n_beam):
    P_loc = 1 - (omega_p/omega_b)**2
    Npar = N0 * np.cos(theta)
    Nperp = N0 * np.sin(theta)
    n0 = omega_b * np.sqrt(1 - Npar**2) / Omega_ce
    tmp = omega_b * (omega_p**2 - (omega_b**2-Omega_ce**2) * (1-N0**2)) / \
              (Omega_ce*omega_p**2)
    Small_a = (1 + P_loc*(Npar*tmp/(P_loc-Nperp**2))**2) * np.sin(theta)
    Small_b = abs(1 + P_loc / (P_loc-Nperp**2) * tmp**2) * abs(np.cos(theta))
    ey = np.sqrt(1 / (N0 * np.sqrt(Small_a**2 + Small_b**2)))
    normalized_freq = Omega_ce / omega_b
    R_loc = (P_loc - normalized_freq) / (1 - normalized_freq)
    L_loc = (P_loc + normalized_freq) / (1 + normalized_freq)
    S_loc = 0.5 * (R_loc + L_loc)
    T_loc = 0.5 * (R_loc - L_loc)
    ex = (S_loc - N0**2) * ey * complex(0,1) / T_loc
    ez = - Npar * Nperp * ex / (P_loc - Nperp**2)
    Axz = ex + Npar * Nperp * ez / (1 - Npar**2)
    xn = omega_b * Nperp * np.sqrt((n_beam/n0)**2 - 1) / Omega_ce
    yn = mu_loc * Npar * np.sqrt(((n_beam/n0)**2 - 1)/(1 - Npar**2))
    [fx, df_dx, df_dy, df_dydy, df_dxdy] = Compute_usefull_derivatives(xn, yn, n_beam)
    [gx, dg_dx, dg_dy, dg_dydy, dg_dxdy] = Compute_usefull_derivatives(xn, yn, n_beam+1)
    BigA = (np.absolute(Axz)**2 + ey**2) * fx + (Axz*complex(0,ey)).real * xn * df_dx / n_beam - \
           (xn/n_beam)**2 * (n_beam/(n_beam+1)) * ey**2 * (fx - df_dydy) + \
           (xn / (n_beam * np.sqrt(1-Npar**2)))**2 * np.absolute(ez)**2 * df_dydy - \
           xn * (2*(Axz*ez.conjugate()).real * df_dy+ (complex(0,ey)*ez).real * xn * df_dxdy/ n_beam) / \
           (n_beam * np.sqrt(1-Npar**2))
    BigB = (xn / n_beam)**2 * (2*n_beam + 3) / ((n_beam+1)*(n_beam+2)) * ey**2 * (gx - dg_dydy)
    return BigA, BigB


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

# Allocation of the empty arrays
vec_Power = np.zeros(Nr)
vec_Albajar = np.zeros(Nr)
vec_Power[Nr-1] = Power_in
vec_Albajar[Nr-1] = Power_in
vec_tau = np.zeros(Nr)
Dn = np.zeros((Nr,2*Nv,Nv))
tau_loc = 0.
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

        # Compute the theoretical optical thickness (Albajar)
        Npar_loc = N0_loc * np.cos(theta0_loc)
        Nperp_loc = N0_loc * np.sin(theta0_loc)
        mu_loc = light_speed**2 * mass / Te_loc
        harmonic0 = omega_b * np.sqrt(1 - Npar_loc**2) / Omega_ce_loc
        if (math.ceil(harmonic0) == harmonic):
            [Big_A_loc, Big_B_loc] = Compute_A_B(omega_b, omega_p_loc, Omega_ce_loc, N0_loc, theta0_loc, harmonic)
            Pn_loc = np.pi*math.factorial(2*harmonic+1) / (2**harmonic*math.factorial(harmonic))**2 * \
                    (harmonic * Omega_ce_loc / (omega_b * Nperp_loc))**2 * (Big_A_loc + Big_B_loc)
            Fn_loc = mu_loc**2.5 * Pn_loc * \
                    np.exp(mu_loc*(1 - harmonic/(harmonic0*np.sqrt(1 - Npar_loc**2))))
            alpha_loc = omega_p_loc**2 * np.sqrt(0.5*np.pi) * Fn_loc * \
                        np.sqrt((harmonic/harmonic0)**2 - 1) / \
                        (light_speed * Omega_ce_loc * harmonic0)
            tau_loc += alpha_loc * abs(np.sin(theta0_loc)) * dR

            
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
                        Dn[iR, ivpar, ivperp] = np.sqrt(np.pi) * charge**2 * N0_loc / \
                                                (2 * mass**2 * omega_b * sigma_loc * \
                                                 abs(Vpar[ivpar]) * vT_on_c_loc) * Theta2_n * E2_loc * \
                                                np.exp(-((theta_res - theta0_loc)/sigma_loc)**2)

                        Power_absorbed += Vperp[ivperp]**3 * Dn[iR, ivpar,ivperp] * \
                                          np.exp(- (Vpar[ivpar]**2 + Vperp[ivperp]**2)/2)
                
        # Normalisation of the Power absorbed
        Power_absorbed *= Ne_loc * mass * dVpar * dVperp * dR * R_loc * np.sqrt(2) * np.pi * W0
        # Normalisation of the resonant diffusion coefficient
        Dn[iR, :, :] = Dn[iR, :, :] / (Omega_ce_loc * Te_loc / mass) 

    # Fill the power vector
    vec_Power[iR] = vec_Power[iR+1] - Power_absorbed
    vec_Albajar[iR] = Power_in * np.exp(-tau_loc)
    if (Power_absorbed > 0.):
        print("iR", iR)


# Save arrays in prevision of their exploitation
np.save('vec_R.npy', vec_R)
np.save('vec_Ne.npy', vec_Ne)
np.save('vec_Te.npy', vec_Te)
np.save('Vpar.npy', Vpar)
np.save('Vperp.npy', Vperp)
np.save('vec_Power.npy', vec_Power)
np.save('vec_Albajar.npy', vec_Albajar)
np.save('Dn.npy', Dn)



