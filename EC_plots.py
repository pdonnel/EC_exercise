 #-*- coding: utf-8 -*-
"""
@author: Peter Donnel
"""

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load arrays for their exploitation
vec_R = np.load('vec_R.npy')
vec_Ne = np.load('vec_Ne.npy')
vec_Te = np.load('vec_Te.npy')
Vpar = np.load('Vpar.npy')
Vperp = np.load('Vperp.npy')
vec_Power = np.load('vec_Power.npy')
Dn = np.load('Dn.npy')

# Plot the density, temperature and Power profiles
fig0 = plt.figure(0,figsize=(10, 10))
ax01 = fig0.add_subplot(311)
ax01.plot(vec_R, vec_Ne)
ax01.set_ylabel("$N_e$ ", fontsize = 20)
ax02 = fig0.add_subplot(312)
ax02.plot(vec_R, vec_Te / (1.602 * 10**(-19)))
ax02.set_ylabel("$T_e$ [eV]", fontsize = 20)
ax03 = fig0.add_subplot(313)
ax03.plot(vec_R, (vec_Power[-1] - vec_Power)/vec_Power[-1])
ax03.set_xlabel("$R - R_0$", fontsize = 20)
ax03.set_ylabel("Fraction absorbed", fontsize = 20)

plt.savefig("Radial_profiles.png")
fig0.show()
# Compute the position of maximum absorption

#fig0 = plt.figure(0,figsize=(12, 6))
#ax0 = fig0.add_subplot(111)
#plt.pcolor(Vpar, Vperp, np.transpose(Dn[Nr/2,:,:]) / (Omega_ce * Te / mass))
#ax0.set_xlabel("$v_{\parallel}$", fontsize = 20)
#ax0.set_ylabel("$v_{\perp}$", fontsize = 20)
#ax0.set_title("$\\frac{D_{n}}{v_{Te}^2 \Omega_{ce}}$", fontsize = 20) 
#ax0.set_aspect('equal','box')
#plt.colorbar()
#

