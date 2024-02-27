 #-*- coding: utf-8 -*-
"""
@author: Peter Donnel
"""

# Import useful libraries
import numpy as np
import math
from scipy.special import jn
from scipy.special import spherical_jn
import sys
np.set_printoptions(threshold=sys.maxsize)
import multiprocessing
from multiprocessing import Pool
import itertools

# Physical constants
charge = 1.602 * 10**(-19)   # elementary charge [C]
mass = 9.109 * 10**(-31)     # electron mass [kg]
light_speed = 299792458      # speed of light [m/s]
epsilon0 = 8.854 * 10**(-12) # vaccum permittivity [F/m]


# Plasma inputs
Ne0 = 1.0 * 10**(19)                  # maximum electron density [m^{-3}]
Te0 = 2.0 * 10**3 * 1.602 * 10**(-19) # maximum of electron temperature [J] (Warning: kB included)
B0 = 1.4                              # magnetic field on axis [T]
R0 = 1.0                              # major radius [m]
a0 = 0.25                             # minor radius [m]


# Beam inputs
harmonic = 2                          # harmonic of the cyclotron frequency
theta_in = np.pi/2 + 0.1              # Toroidal angle of injection
omega_b = 7.8 * 10**10 * 2 * np.pi    # beam pulsation [Hz]
W0 = 0.02                             # beam width [m]
Power_in = 1                          # power input of the beam [W]


# Numerical imput data
vmax = 4   # maximal velocity (normalized to the local thermal velocity)
Nv = 300   # number of grid points in velocity space (Nv for vperp, 2*Nv for vpar)
Nr = 400   # number of grid points for the major radius direction


# Generation of the grid
Vpar = np.linspace(-vmax,vmax,2*Nv)
Vperp = np.linspace(vmax/Nv,vmax,Nv)
vec_R = np.linspace(R0-a0,R0+a0,Nr)

dVpar =  Vpar[2] -Vpar[1]
dVperp = Vperp[2] -Vperp[1]
dR = vec_R[2] - vec_R[1]


# density and temperature profiles (assume a parabolic profiles)
vec_Ne = np.zeros(Nr)
vec_Te = np.zeros(Nr)
for iR in range(Nr):
    R_loc = vec_R[iR]
    vec_Ne[iR] = Ne0 * (1 - 0.9 * ((R_loc - R0) / a0)**2)
    vec_Te[iR] = Te0 * (1 - 0.9 * ((R_loc - R0) / a0)**2)


# Compute the refractive index
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
    

# Compute the resonant angle
def compute_theta_res(lambda_loc, P_loc, Omegace_on_omegab_loc):
    lambda2 = lambda_loc**2
    tmp1 = P_loc**2 * (2*lambda2 - P_loc) + \
           Omegace_on_omegab_loc**2 * (P_loc * (1 - lambda2)**2 - lambda2**2)
    tmp2 = Omegace_on_omegab_loc * (1 - P_loc) * lambda2 * \
           np.sqrt((Omegace_on_omegab_loc * (1 - lambda2))**2 + 4 * P_loc * lambda2)
    tmp3 =  P_loc * (P_loc**2 - Omegace_on_omegab_loc**2) + \
            lambda2 * Omegace_on_omegab_loc**2 * (P_loc - 1)
    x_lambda = (tmp1 + tmp2) / tmp3
    # Distinction between float and arrays
    if (isinstance(x_lambda, np.floating)):
        if (abs(x_lambda)>1):
            theta_res = 0.5*np.pi
        else:
            y_lambda = 0.5*np.arccos(x_lambda)
            if (lambda_loc>0):
                theta_res = y_lambda
            else:
                theta_res = np.pi - y_lambda
    else:
        x_lambda = np.where(abs(x_lambda)<1.0, x_lambda, 1.0)
        y_lambda = 0.5*np.arccos(x_lambda)
        theta_res = np.where(lambda_loc>0, y_lambda, np.pi - y_lambda) 

    return theta_res

# Compute |Theta_n|^2
def compute_Theta2_n(rho, theta, Ntheta, P_loc, omega_ce_loc, omega_b_loc, vpar, vperp):
    normalized_freq = omega_ce_loc / omega_b_loc
    R_loc = (P_loc - normalized_freq) / (1 - normalized_freq)
    L_loc = (P_loc + normalized_freq) / (1 + normalized_freq)
    S_loc = 0.5 * (R_loc + L_loc)
    T_loc = 0.5 * (R_loc - L_loc)

    tmp1 = (1 + T_loc / (S_loc - Ntheta**2)) * jn(harmonic+1,rho)
    tmp2 = (1 - T_loc / (S_loc - Ntheta**2)) * jn(harmonic-1,rho)
    tmp3 = - 2 * Ntheta**2 * np.cos(theta) * np.sin(theta) * np.divide.outer(vpar,vperp) * jn(harmonic,rho) / \
           (P_loc - (Ntheta*np.sin(theta))**2)
    tmp4 = 4 * (1 + (T_loc / (S_loc - Ntheta**2))**2 + \
                (Ntheta**2 * np.cos(theta) * np.sin(theta) / \
                 (P_loc - (Ntheta*np.sin(theta))**2))**2)
    Theta2 = np.where(abs(vperp) < 0.01, 0., (tmp1 + tmp2 + tmp3)**2 / tmp4)

    return Theta2


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
           xn * (2*(Axz*ez.conjugate()).real * df_dy+ (complex(0,ey)*ez).real*xn*df_dxdy/n_beam) / \
           (n_beam * np.sqrt(1-Npar**2))
    BigB = (xn / n_beam)**2 * (2*n_beam + 3) / ((n_beam+1)*(n_beam+2)) * ey**2 * (gx - dg_dydy)
    return BigA, BigB

def Compute_Dn_and_Pabs_wrapper(args):
    
    ivpar = args[0]
    ivperp = args[1]
    iR = args[2]
    vT_on_c = args[3]
    omega_b = args[4]
    Omega_ce = args[5]
    P_loc = args[6]
    theta0_loc = args[7]
    N0_loc = args[8]
    sigma = args[9]
    E2_loc = args[10]
    
    [Dn, Power_loc] = Compute_Dn_and_Pabs(ivpar,ivperp,iR,vT_on_c,omega_b,Omega_ce,P_loc,theta0_loc,N0_loc,sigma,E2_loc)

    return Dn, Power_loc

def Compute_Dn_and_Pabs(ivpar,ivperp,iR,vT_on_c,omega_b,Omega_ce,P_loc,theta0_loc,N0_loc,sigma,E2_loc):

   vpar = Vpar[ivpar]
   vperp = Vperp[ivperp]
   energy = vperp**2 + vpar**2
   lorentz = 1 / np.sqrt(1 - energy * vT_on_c**2)
   lambda_phys = (1 - harmonic * Omega_ce/omega_b / lorentz) / \
                 (vpar * vT_on_c)
   theta_res = compute_theta_res(lambda_phys,P_loc,Omega_ce/omega_b)
   rho = np.sin(theta0_loc) * N0_loc * omega_b *  np.transpose(lorentz) * vperp * \
         vT_on_c / Omega_ce

   Theta2_n = compute_Theta2_n(rho, theta0_loc, N0_loc, P_loc, Omega_ce, \
                               omega_b, vpar, vperp)
   Dn_loc = np.sqrt(np.pi) * charge**2 * N0_loc * E2_loc * Theta2_n / \
        (2 * mass**2 * omega_b * sigma * abs(vpar) * vT_on_c) * \
        np.exp(-((theta_res - theta0_loc)/sigma)**2)
   Power = vperp**3 * Dn_loc * np.exp(- energy/2)

   
   return Dn_loc, Power




def Compute_Dn_and_Pabs_vec(vpar,vperp,vT_on_c,omega_b,Omega_ce,P_loc,theta0_loc,N0_loc,sigma):

   energy = np.add.outer( vperp**2, vpar**2)
   lorentz = 1 / np.sqrt(1 - energy * vT_on_c**2)
   lambda_phys = (1 - harmonic * Omega_ce/omega_b / lorentz) / \
                 (vpar * vT_on_c)
   theta_res = compute_theta_res(lambda_phys,P_loc,Omega_ce/omega_b)
   rho = np.sin(theta0_loc) * N0_loc * omega_b *  np.transpose(lorentz) * vperp * \
         vT_on_c / Omega_ce

   Theta2_n = compute_Theta2_n(rho, theta0_loc, N0_loc, P_loc, Omega_ce, \
                               omega_b, vpar, vperp)
   Theta2_n = np.transpose(Theta2_n)
   Dn = np.sqrt(np.pi) * charge**2 * N0_loc * E2_loc * Theta2_n / \
        (2 * mass**2 * omega_b * sigma * \
         abs(vpar) * vT_on_c) * \
        np.exp(-((theta_res - theta0_loc)/sigma)**2)
   Dn = np.transpose(Dn)
   energy = np.transpose(energy)
   Power = vperp**3 * Dn * np.exp(- energy/2)

   return Dn, Power

def test(vpar, vperp):

    energy = vpar**2 + vperp**2

    return energy

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

# Computation of the resonant diffusion coefficient and
# the numerical and theoretical power deposition
tau_loc = 0.
for iR in range(Nr-2,-1,-1):
    Power_absorbed = 0.
    R_loc = vec_R[iR] 
    if R_loc < max(R_res_max,R_eff_max) and R_loc > R_eff_min:
        print("iR is being calculated", iR)
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
            [Big_A_loc, Big_B_loc] = Compute_A_B(omega_b, omega_p_loc, Omega_ce_loc, \
                                                 N0_loc, theta0_loc, harmonic)
            Pn_loc = np.pi*math.factorial(2*harmonic+1) / (2**harmonic*math.factorial(harmonic))**2 * \
                    (harmonic * Omega_ce_loc / (omega_b * Nperp_loc))**2 * (Big_A_loc + Big_B_loc)
            Fn_loc = mu_loc**2.5 * Pn_loc * \
                    np.exp(mu_loc*(1 - harmonic/(harmonic0*np.sqrt(1 - Npar_loc**2))))
            alpha_loc = omega_p_loc**2 * np.sqrt(0.5*np.pi) * Fn_loc * \
                        np.sqrt((harmonic/harmonic0)**2 - 1) / \
                        (light_speed * Omega_ce_loc * harmonic0)
            tau_loc += alpha_loc * abs(np.sin(theta0_loc)) * dR

            # Vectorized version of the code
#            [Dn_loc, Power] = Compute_Dn_and_Pabs_vec(Vpar,Vperp,vT_on_c_loc,omega_b,Omega_ce_loc,P_loc,theta0_loc,N0_loc,sigma_loc)            
#            Power_absorbed = np.sum(Power)

            # Parallelized version of the code
            Power_absorbed = 0.0
            args =[]
            for ivpar in range(len(Vpar)):
                for ivperp in range(len(Vperp)):
                    args.append((ivpar,ivperp,iR,vT_on_c_loc,omega_b,Omega_ce_loc,P_loc,theta0_loc,N0_loc,sigma_loc,E2_loc))
                    
            # protect the entry point
            if __name__ == '__main__':
                # create as many workers as possible (# of cores on the CPU)
                with Pool(processes=multiprocessing.cpu_count()) as pool:

                    Dn_pool, Power_pool = zip(*pool.map(Compute_Dn_and_Pabs_wrapper, args))

                    # We sum the Power_pool
                    Power_absorbed = sum(Power_pool)
                    #We write the result of Dn_pool at the right place
                    Dn_loc = np.zeros((2*Nv,Nv))
                    for i in range (len(Dn_pool)):
                        ivpar = args[i][0]
                        ivperp = args[i][1]
                        Dn_loc[ivpar,ivperp] = Dn_pool[i]

            # Normalisation of the resonant diffusion coefficient
            Dn[iR, :, :] = Dn_loc / (Omega_ce_loc * Te_loc / mass) 

            # Normalisation of the Power absorbed
            Power_absorbed *= Ne_loc * mass * dVpar * dVperp * dR * R_loc * np.sqrt(2) * np.pi * W0
            print(Power_absorbed)

    # Fill the power vector
    vec_Power[iR] = vec_Power[iR+1] - Power_absorbed
    vec_Albajar[iR] = Power_in * np.exp(-tau_loc)



# Save arrays in prevision of their exploitation
np.save('vec_R.npy', vec_R)
np.save('vec_Ne.npy', vec_Ne)
np.save('vec_Te.npy', vec_Te)
np.save('Vpar.npy', Vpar)
np.save('Vperp.npy', Vperp)
np.save('vec_Power.npy', vec_Power)
np.save('vec_Albajar.npy', vec_Albajar)
np.save('Dn.npy', Dn)



