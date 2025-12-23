# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 21:24:52 2025

@author: aleja
"""

# Importing packages
import numpy as np
import matplotlib.pyplot as plt

"Opening the MARCS models"

# MARCS 5000
file = "C:\\Users\\aleja\\astrofisica\\atm_estelares\\Modelos_atm\\t5000.dat"

# Read the file as a list of lines
with open(file, "r") as f:
    lines = f.readlines()
    content = f.read()
    #print(content)

# Find where the numerical table begins
start = None
for i, line in enumerate(lines):
    if line.strip().startswith("k"):
        start = i + 1
        break
    
# Defining universal constants
k_B = 1.3806*10**(-16) # erg/K
m_H = 1.66*10**(-24) # g
Z_H = 2
Z_p = 1
Z_He = 1
Z_He_I = 2
Z_He_II = 1
chi_H = 13.6 # eV
chi_He = 24.6 # eV
chi_He_2 = 54.4 # eV

# Load only the numerical block
data = np.loadtxt(lines[start: start + 56])
k      = data[:,0]
lgTauR = data[:,1]
lgTau5 = data[:,2]
Depth  = data[:,3]
T      = data[:,4]  
Pe     = data[:,5]   
Pg     = data[:,6]   
Prad   = data[:,7]   
Pturb  = data[:,8]   

# Opening the MARCS 8000 model
file_8000 = "C:\\Users\\aleja\\astrofisica\\atm_estelares\\Modelos_atm\\t8000.dat"

# Read the file as a list of lines
with open(file_8000, "r") as f:
    lines_8000 = f.readlines()
    content = f.read()

# Find where the numerical table begins
start_8000 = None
for i, line in enumerate(lines_8000):
    if line.strip().startswith("k"):
        start_8000 = i + 1
        break

# Load only the numerical block
data_8000 = np.loadtxt(lines_8000[start_8000: start_8000 + 56])

k_8000      = data_8000[:,0]
lgTauR_8000 = data_8000[:,1]
lgTau5_8000 = data_8000[:,2]
Depth_8000  = data_8000[:,3]
T_8000      = data_8000[:,4]  
Pe_8000     = data_8000[:,5]   
Pg_8000     = data_8000[:,6]   
Prad_8000   = data_8000[:,7]   
Pturb_8000  = data_8000[:,8]   

# Gray body temperature
def T_gray(Teff, lgtau):
    return ((3/4)*(Teff**(4))*(10**(lgtau) + 2/3))**(1/4)

# Value of Teff
Teff_MARCS_5000 = 5000
Teff_MARCS_8000 = 8000
Teff_TLUSTY = 15000

# Opening the TLUSTY file
file_tlusty = "C:\\Users\\aleja\\astrofisica\\atm_estelares\\Modelos_atm\\BG15000g450v2_11.dat"

# Read the file as a list of lines
with open(file_tlusty, "r") as f:
    lines_15000 = f.readlines()
    #print(lines_15000)
    
# Load and store the data
data_15000 = np.loadtxt(lines_15000[1:])
k = data_15000[:, 0]
m = data_15000[:, 1]
tau_r = data_15000[:, 2]
kappa_r = data_15000[:, 3]
temperature = data_15000[:, 4]
n_e = data_15000[:, 5]
rho = data_15000[:, 6]

"Defninig the functions for the calculations of Pe and Pg for the TLUSTY model"
def Pe_15000(ne, T):
    return ne*k_B*T  

p_e_15000 = Pe_15000(n_e, temperature)

# Calculating Pg
def Pg_15000(rho, mu, T):
    return ((rho*k_B*T)/(mu*m_H))

# Mean molecular weight
# Solar Composition
X = 0.73826
Y = 0.24954
Z = 1.22e-2

def mu(Zh, Zhe, Zz, Ah, Ahe, Az, X, Y, Z):
    return ((1 + Zh)*X)/Ah + ((1 + Zhe)*Y)/Ahe + ((1 + Zz)*Z)/Az

# Full-ionization assumption
fully_ionized_mu = 1/mu(1, 2, 7, 1, 4, 16, X, Y, Z)

# Full-ionization gas pressure
p_g_15000_full = Pg_15000(rho, fully_ionized_mu, temperature)

# Partially-ionized (real case)

# Defining Saha equation
def saha(ne, Z1, Z2, T, chi):
    return (2.07*10e-16)*ne*(Z1/Z2)*(T)**(-3/2)*np.exp((chi*1.602e-12)/(k_B*T))

# Defining quotients of ions from Saha equation
hydrogen_fraction = saha(n_e, Z_H, Z_p, temperature, chi_H)
neutral_helium_fraction = saha(n_e, Z_He, Z_He_I, temperature, chi_He)
ionized_helium_fraction = saha(n_e, Z_He_I, Z_He_II, temperature, chi_He_2)
jumped_helium_fraction = (1/neutral_helium_fraction)*(1/ionized_helium_fraction)

# Defining the fractions of ionizations
x_H = (1/hydrogen_fraction)/(1 + (1/hydrogen_fraction))
x_He_II = (1/neutral_helium_fraction)/(1 + (1/neutral_helium_fraction) + jumped_helium_fraction)
x_He_III = (jumped_helium_fraction)/(1 + (1/neutral_helium_fraction) + jumped_helium_fraction)

# Avg. number of free electrons from partial ionization
z_h = x_H
z_he = 1*x_He_II + 2*x_He_III

# Partially ionized mean molecular weight
partially_ionized_mu = 1/mu(z_h, z_he, 7, 1, 4, 16, X, Y, Z)

# SAving data in table
table_tlusty = np.column_stack((tau_r, partially_ionized_mu, x_H, x_He_II, x_He_III))
np.savetxt("TLUSTY_15000_ionization_calculations.csv", table_tlusty, delimiter=",", header="logtaur, partially_ionized_mu, x_H, x_He_II, x_He_III", comments="Ionization calculations for the TLUSTY model", fmt="%.4e")

# Partially-ionized gas pressure
p_g_15000_partial = Pg_15000(rho, partially_ionized_mu, temperature)

"Calculation of the mass density for the MARCS models"
def mass_density(Pg, mu, T):
    return (Pg*mu*m_H)/(k_B*T)

def electron_density(Pe, T):
    return (Pe/(k_B*T))

# Electron density from MARCS_5000
elec_density = electron_density(Pe, T)

# Neutral assumption
neutral_mu = 1/mu(0, 0, 0, 1, 4, 16, X, Y, Z)

# Calculated mass density from neutral assumption
neutral_mass_density = mass_density(Pg, neutral_mu, T)

# Supposing partial ionization

# Defining quotients of ions from Saha equation
hydrogen_fraction_marcs = saha(elec_density, Z_H, Z_p, T, chi_H)
neutral_helium_fraction_marcs = saha(elec_density, Z_He, Z_He_I, T, chi_He)
ionized_helium_fraction_marcs = saha(elec_density, Z_He_I, Z_He_II, T, chi_He_2)
jumped_helium_fraction_marcs = (1/neutral_helium_fraction_marcs)*(1/ionized_helium_fraction_marcs)

# Defining the fractions of ionizations
x_H_marcs = (1/hydrogen_fraction_marcs)/(1 + (1/hydrogen_fraction_marcs))
x_He_II_marcs = (1/neutral_helium_fraction_marcs)/(1 + (1/neutral_helium_fraction_marcs) + jumped_helium_fraction_marcs)
x_He_III_marcs = (jumped_helium_fraction_marcs)/(1 + (1/neutral_helium_fraction_marcs) + jumped_helium_fraction_marcs)

# Avg. number of free electrons from partial ionization
z_h_marcs = x_H_marcs
z_he_marcs = 1*x_He_II_marcs + 2*x_He_III_marcs

# MARCS models partial ionization
partial_marcs_mu = 1/mu(z_h_marcs, z_he_marcs, 7, 1, 4, 16, X, Y, Z)

# SAving data in table
table_marcs_5000 = np.column_stack((lgTauR, partial_marcs_mu, x_H_marcs, x_He_II_marcs, x_He_III_marcs))
np.savetxt("MARCS_5000_ionization_calculations.csv", table_marcs_5000, delimiter=",", header="logtaur, partially_ionized_mu, x_H, x_He_II, x_He_III", comments="", fmt="%.4e")

# Calculated density from partial ionziation for MARCS models
partial_mass_density = mass_density(Pg, partial_marcs_mu, T)

# Electron density from MARCS_8000
elec_density_8000 = electron_density(Pe_8000, T_8000)

# Neutral assumption
neutral_mass_density_8000 = mass_density(Pg_8000, neutral_mu, T_8000)

# Partial ionization (MARCS_8000)
hydrogen_fraction_8000 = saha(elec_density_8000, Z_H, Z_p, T_8000, chi_H)
neutral_helium_fraction_8000 = saha(elec_density_8000, Z_He, Z_He_I, T_8000, chi_He)
ionized_helium_fraction_8000 = saha(elec_density_8000, Z_He_I, Z_He_II, T_8000, chi_He_2)
jumped_helium_fraction_8000 = (1/neutral_helium_fraction_8000)*(1/ionized_helium_fraction_8000)

x_H_8000 = (1/hydrogen_fraction_8000)/(1 + (1/hydrogen_fraction_8000))
x_He_II_8000 = (1/neutral_helium_fraction_8000)/(1 + (1/neutral_helium_fraction_8000) + jumped_helium_fraction_8000)
x_He_III_8000 = (jumped_helium_fraction_8000)/(1 + (1/neutral_helium_fraction_8000) + jumped_helium_fraction_8000)

z_h_8000 = x_H_8000
z_he_8000 = 1*x_He_II_8000 + 2*x_He_III_8000

partial_marcs_mu_8000 = 1/mu(z_h_8000, z_he_8000, 7, 1, 4, 16, X, Y, Z)

partial_mass_density_8000 = mass_density(Pg_8000, partial_marcs_mu_8000, T_8000)

# Saving table
table_marcs_8000 = np.column_stack((lgTauR_8000, partial_marcs_mu_8000, x_H_8000, x_He_II_8000, x_He_III_8000))
np.savetxt("MARCS_8000_ionization_calculations.csv", table_marcs_8000, delimiter=",", header="logtaur, partially_ionized_mu, x_H, x_He_II, x_He_III", comments="", fmt="%.4e")


"Plotting snippet of code"
plt.close()


# MARCS model
plt.figure(1, figsize=(12, 8))
plt.plot(lgTauR, T, color="black", label=r"$T_{\mathrm{eff}} = 5000$ [K]")
plt.plot(lgTauR_8000, T_8000, color="slategray", label=r"$T_{\mathrm{eff}} = 8000$ [K]")
plt.xlabel(r"$\log_{10}(\tau_{\mathrm{R}})$", fontsize=28)
plt.ylabel(r"$T \,$ [K]", fontsize=28)
plt.tick_params(direction="in", which="major", length=8, labelsize=24, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=24, top=True, right=True)
plt.legend(loc="upper left", prop={"size":20})
plt.title(r"$\mathbf{MARCS}$ model", fontsize=30)
plt.tight_layout()
plt.savefig("t.pdf", dpi=300)
plt.show()

plt.figure(2, figsize=(12, 8))
plt.plot(lgTauR, Pe, color="black", label=r"$T_{\mathrm{eff}} = 5000$ [K]")
plt.plot(lgTauR_8000, Pe_8000, color="slategray", label=r"$T_{\mathrm{eff}} = 8000$ [K]")
plt.xlabel(r"$\log_{10}(\tau_{\mathrm{R}})$", fontsize=28)
plt.ylabel(r"$P_{\mathrm{e}} \,$ [dyne cm$^{-2}$]", fontsize=28)
plt.tick_params(direction="in", which="major", length=8, labelsize=24, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=24, top=True, right=True)
plt.legend(loc="upper left", prop={"size":20})
plt.title(r"$\mathbf{MARCS}$ model", fontsize=30)
plt.tight_layout()
plt.savefig("pe.pdf", dpi=300)
plt.show()

plt.figure(3, figsize=(12, 8))
plt.plot(lgTauR, Pe/Pg, color="black", label=r"$T_{\mathrm{eff}} = 5000$ [K]")
plt.plot(lgTauR_8000, Pe_8000/Pg_8000, color="slategray", label=r"$T_{\mathrm{eff}} = 8000$ [K]")
plt.xlabel(r"$\log_{10}(\tau_{\mathrm{R}})$", fontsize=28)
plt.ylabel(r"$P_{\mathrm{e}}/P_{\mathrm{g}}$", fontsize=28)
plt.tick_params(direction="in", which="major", length=8, labelsize=24, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=24, top=True, right=True)
plt.legend(loc="upper left", prop={"size":20})
plt.title(r"$\mathbf{MARCS}$ model", fontsize=30)
plt.tight_layout()
plt.savefig("pe_pg.pdf", dpi=300)
plt.show()

plt.figure(4, figsize=(12, 8))
plt.plot(lgTauR, neutral_mass_density/10e-8, color="black", label="Neutral Ionization ($T_{\mathrm{eff}} = 5000$ [K])")
plt.plot(lgTauR, partial_mass_density/10e-8, color="black", label="Partial Ionization ($T_{\mathrm{eff}} = 5000$ [K])", linestyle="dashed")
plt.plot(lgTauR_8000, neutral_mass_density_8000/10e-8, color="slategray", label="Neutral Ionization ($T_{\mathrm{eff}} = 8000$ [K])")
plt.plot(lgTauR_8000, partial_mass_density_8000/10e-8, color="slategray", label="Partial Ionization ($T_{\mathrm{eff}} = 8000$ [K])", linestyle="dashed")
plt.xlabel(r"$\log_{10}(\tau_{\mathrm{R}})$", fontsize=28)
plt.ylabel(r"$\rho \times 10^{-8} \,$ [$\mathrm{g} \, \mathrm{cm}^{-3}$]", fontsize=28)
plt.tick_params(direction="in", which="major", length=8, labelsize=24, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=24, top=True, right=True)
plt.legend(loc="upper left", prop={"size":20})
plt.title(r"$\mathbf{MARCS}$ model", fontsize=30)
plt.tight_layout()
plt.savefig("rho.pdf", dpi=300)
plt.show()

plt.figure(5, figsize=(12, 8))
plt.plot(T_gray(Teff_MARCS_5000, lgTauR), T, color="black", label=r"$T_{\mathrm{eff}} = 5000$ [K]")
plt.plot(T_gray(Teff_MARCS_8000, lgTauR_8000), T_8000, color="slategray", label=r"$T_{\mathrm{eff}} = 8000$ [K]")
plt.xlabel(r"$T_{\mathrm{gray}} \,$ [K]", fontsize=28)
plt.ylabel(r"$T \,$ [K]", fontsize=28)
plt.tick_params(direction="in", which="major", length=8, labelsize=24, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=24, top=True, right=True)
plt.legend(loc="upper left", prop={"size":20})
plt.title(r"$\mathbf{MARCS}$ model", fontsize=30)
plt.tight_layout()
plt.savefig("tgray.pdf", dpi=300)
plt.show()
"""

"""
# TLUSTY model
plt.figure(6, figsize=(12, 8))
plt.plot(np.log10(tau_r), temperature, color="red")
plt.xlabel(r"$\log_{10}(\tau_{\mathrm{R}})$", fontsize=28)
plt.ylabel(r"$T \,$ [K]", fontsize=28)
plt.tick_params(direction="in", which="major", length=8, labelsize=24, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=24, top=True, right=True)
plt.legend(loc="upper left", prop={"size":20})
plt.title(r"$\mathbf{TLUSTY}$ model", fontsize=30)
plt.tight_layout()
plt.savefig("t_15000.pdf", dpi=300)
plt.show()

plt.figure(7, figsize=(12, 8))
plt.plot(np.log10(tau_r), p_e_15000, color="red")
plt.xlabel(r"$\log_{10}(\tau_{\mathrm{R}})$", fontsize=28)
plt.ylabel(r"$P_{\mathrm{e}}$ [dyne cm$^{-2}$]", fontsize=28)
plt.tick_params(direction="in", which="major", length=8, labelsize=24, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=24, top=True, right=True)
plt.ticklabel_format(axis='both', style='sci', scilimits=(1,7))
plt.legend(loc="upper left", prop={"size":20})
plt.title(r"$\mathbf{TLUSTY}$ model", fontsize=30)
plt.tight_layout()
plt.savefig("pe_15000.pdf", dpi=300)
plt.show()

plt.figure(8, figsize=(12, 8))
plt.plot(np.log10(tau_r), (p_e_15000/p_g_15000_full), color="red", label="Complete Ionization")
plt.plot(np.log10(tau_r), (p_e_15000/p_g_15000_partial), color="red", linestyle="dashed", label="Partial Ionization")
plt.xlabel(r"$\log_{10}(\tau_{\mathrm{R}})$", fontsize=28)
plt.ylabel(r"$P_{\mathrm{e}}/P_{\mathrm{g}}$", fontsize=28)
plt.tick_params(direction="in", which="major", length=8, labelsize=24, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=24, top=True, right=True)
plt.legend(loc="upper left", prop={"size":20})
plt.title(r"$\mathbf{TLUSTY}$ model", fontsize=30)
plt.tight_layout()
plt.savefig("pe_pg_15000.pdf", dpi=300)
plt.show()

plt.figure(9, figsize=(12, 8))
plt.plot(np.log10(tau_r), rho/10e-8, color="red")
plt.xlabel(r"$\log_{10}(\tau_{\mathrm{R}})$", fontsize=28)
plt.ylabel(r"$\rho \times 10^{-8} \,$ [$\mathrm{g} \, \mathrm{cm}^{-3}$]", fontsize=28)
plt.tick_params(direction="in", which="major", length=8, labelsize=24, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=24, top=True, right=True)
plt.legend(loc="upper left", prop={"size":20})
plt.title(r"$\mathbf{TLUSTY}$ model", fontsize=30)
plt.tight_layout()
plt.savefig("rho_15000.pdf", dpi=300)
plt.show()

plt.figure(10, figsize=(12, 8))
plt.plot(T_gray(Teff_TLUSTY, np.log10(tau_r)), temperature, color="red")
plt.xlabel(r"$T_{\mathrm{gray}} \,$ [K]", fontsize=28)
plt.ylabel(r"$T \,$ [K]", fontsize=28)
plt.tick_params(direction="in", which="major", length=8, labelsize=24, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=24, top=True, right=True)
plt.legend(loc="upper left", prop={"size":20})
plt.title(r"$\mathbf{TLUSTY}$ model", fontsize=30)
plt.tight_layout()
plt.savefig("tgray_15000.pdf", dpi=300)
plt.show()



# MARCS_8000 model
plt.figure(11)
plt.plot(lgTauR_8000, T_8000, color="silver")
plt.xlabel(r"$\log_{10}(\tau_{\mathrm{R}})$", fontsize=30)
plt.ylabel(r"$T \,$ [K]", fontsize=30)
plt.tick_params(direction="in", which="major", length=8, labelsize=26, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=26, top=True, right=True)
plt.legend(loc="upper left", prop={"size":20})
plt.title(r"$\mathbf{MARCS}$ model $(T_{\mathrm{eff}} = 8000$ [K])", fontsize=30)
plt.tight_layout()
plt.show()

plt.figure(12)
plt.plot(lgTauR_8000, Pe_8000, color="blue")
plt.xlabel(r"$\log_{10}(\tau_{\mathrm{R}})$", fontsize=30)
plt.ylabel(r"$P_{\mathrm{e}} \,$ [dyne cm$^{-2}$]", fontsize=30)
plt.tick_params(direction="in", which="major", length=8, labelsize=26, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=26, top=True, right=True)
plt.legend(loc="upper left", prop={"size":20})
plt.title(r"$\mathbf{MARCS}$ model $(T_{\mathrm{eff}} = 8000$ [K])", fontsize=30)
plt.tight_layout()
plt.show()

plt.figure(13)
plt.plot(lgTauR_8000, Pe_8000/Pg_8000, color="orchid")
plt.xlabel(r"$\log_{10}(\tau_{\mathrm{R}})$", fontsize=30)
plt.ylabel(r"$P_{\mathrm{e}}/P_{\mathrm{g}}$", fontsize=30)
plt.tick_params(direction="in", which="major", length=8, labelsize=26, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=26, top=True, right=True)
plt.legend(loc="upper left", prop={"size":20})
plt.title(r"$\mathbf{MARCS}$ model $(T_{\mathrm{eff}} = 8000$ [K])", fontsize=30)
plt.tight_layout()
plt.show()

plt.figure(14)
plt.plot(lgTauR_8000, neutral_mass_density_8000/10e-8, color="green", label="Neutral Ionization")
plt.plot(lgTauR_8000, partial_mass_density_8000/10e-8, color="green", label="Partial Ionization", linestyle="dashed")
plt.xlabel(r"$\log_{10}(\tau_{\mathrm{R}})$", fontsize=30)
plt.ylabel(r"$\rho \times 10^{-8} \,$ [$\mathrm{g} \, \mathrm{cm}^{-3}$]", fontsize=30)
plt.tick_params(direction="in", which="major", length=8, labelsize=26, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=26, top=True, right=True)
plt.legend(loc="upper left", prop={"size":20})
plt.title(r"$\mathbf{MARCS}$ model $(T_{\mathrm{eff}} = 8000$ [K])", fontsize=30)
plt.tight_layout()
plt.show()

plt.figure(15)
plt.plot(T_gray(Teff_MARCS_8000, lgTauR_8000), T_8000, color="gray")
plt.xlabel(r"$T_{\mathrm{gray}} \,$ [K]", fontsize=30)
plt.ylabel(r"$T \,$ [K]", fontsize=30)
plt.tick_params(direction="in", which="major", length=8, labelsize=26, top=True, right=True)
plt.tick_params(direction="in", which="minor", length=3, labelsize=26, top=True, right=True)
plt.legend(loc="upper left", prop={"size":20})
plt.title(r"$\mathbf{MARCS}$ model $(T_{\mathrm{eff}} = 8000$ [K])", fontsize=30)
plt.tight_layout()
plt.show()

