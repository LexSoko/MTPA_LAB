#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as ip
import scipy.signal as sig
from scipy.constants import N_A
from uncertainties import ufloat as uf
from uncertainties import unumpy as unp

def interpolate(table, reference, x, y):
    table.update(pd.DataFrame({
        x: reference[x],
        y: np.interp(reference[x], table[x], table[y])
    }))

def shrink(table, column, min, max):
    table.drop(table[table[column] < min].index, inplace=True)
    table.drop(table[table[column] > max].index, inplace=True)

def shift(table, column, delta):
    table.update(pd.DataFrame({column: table[column] + delta}))

spectra = "Versuch/MilacherSokolov"
#spectra = "Versuch/KnefzStark"
images = "Latex/Absorption_Spectroscopy/graphics"

config = [
    ("SHG Reference Spectrum", f"{spectra}/4-25Grad_justiert-100_avg.txt", f"{images}/iodine-reference.png", uf(0, 0), 0),
    ("Single pass, (23 ± 3) °C", f"{spectra}/4-25Grad_justiert_jod-100_avg.txt", f"{images}/absorbance-25.png", uf(94e-3, 6e-3), 0),
    ("Single pass, (35 ± 3) °C", f"{spectra}/4-42.4Grad_justiert_jod-100_avg.txt", f"{images}/absorbance-42.png", uf(94e-3, 6e-3), 0),
    ("Single pass, (53 ± 3) °C", f"{spectra}/4-60.1Grad_justiert_jod-100_avg.txt", f"{images}/absorbance-60.png", uf(94e-3, 6e-3), 0),
    ("Triple pass, (53 ± 3) °C", f"{spectra}/4-60.1Grad_justiert_jod-100_avg_triple.txt", f"{images}/absorbance-triple.png", uf(94e-3, 6e-3)*3, 0)
]

# config = [
#     ("SHG Reference Spectrum", f"{spectra}/Spectrum_ref.txt", f"{images}/iodine-reference.png", uf(0, 0)),
#     # ("Single pass, (23 ± 3) °C", f"{spectra}/4-25Grad_justiert_jod-100_avg.txt", f"{images}/absorbance-25.png", uf(90e-3, 5e-3)),
#     # ("Single pass, (35 ± 3) °C", f"{spectra}/4-42.4Grad_justiert_jod-100_avg.txt", f"{images}/absorbance-42.png", uf(90e-3, 5e-3)),
#     ("Single pass, (53 ± 3) °C", f"{spectra}/spektrum_55.txt", f"{images}/absorbance-60.png", uf(90e-3, 5e-3)),
#     ("Triple pass, (53 ± 3) °C", f"{spectra}/spektrum_3_55.txt", f"{images}/absorbance-triple.png", uf(90e-3, 5e-3)*3)
# ]

config = [(t, f, i, l, o, pd.read_csv(f, delim_whitespace=True, names=["lambda", "I"], skiprows=1)) for t, f, i, l, o in config]

for _, _, _, _, o, table in config:
    shift(table, "lambda", o)

iodine = config[1:]
reference = config[0]
reference_data = reference[-1]
interpolate(reference_data, iodine[0][-1], "lambda", "I")
shrink(reference_data, "lambda", 505, 525)

literature = pd.read_csv(f"{spectra}/absorption-literature.txt", delim_whitespace=True, names=["lambda", "sigma"])
interpolate(literature, iodine[0][-1], "lambda", "sigma")
shrink(literature, "lambda", 505, 525)

plt.figure(figsize=(9, 3.5))
plt.plot(reference[-1]["lambda"], reference[-1]["I"], label=r"$I_0$")
for title, _, image, _, _, data in iodine:
    plt.plot(data["lambda"], data["I"], label=r"$I_t$" + f" ({title})")

plt.xlabel(r"$\lambda$ / nm")
plt.ylabel(r"$I$ / 1")

plt.legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{images}/transmission.png", dpi=300)
# plt.show()
# exit()

epsilon = N_A * literature["sigma"] / np.log(10) / 1e3 # L / mol / cm

for title, _, image, _, _, data in iodine:
    plt.figure(figsize=(9, 3.5))
    for l in [508.79, 509.62, 510.53]:
        plt.axvline(l, color="orange", label="Peak match")
    absorbance = -np.log10(data["I"]/reference[-1]["I"])
    absorbance = sig.savgol_filter(absorbance, 15, 1)
    l1 = plt.plot(data["lambda"], absorbance, label=r"$A$" + f" ({title})")
    plt.xlabel(r"$\lambda$ / nm")
    plt.ylabel(r"$A$ / 1")
    twin1 = plt.gca().twinx()
    l2 = twin1.plot(literature["lambda"], epsilon, color="grey", linestyle="--", label=r"$\epsilon$")
    leg = l1 + l2
    labs = [l.get_label() for l in leg]
    plt.gca().legend(leg, labs, loc="upper right")
    twin1.set_ylabel(r"$\epsilon$ / L mol$^{-1}$ cm$^{-1}$")
    plt.tight_layout()
    plt.savefig(image, dpi=300)

low = 508
up = 511

shrink(literature, "lambda", low, up)
for t, f, i, l, o, d in config:
    shrink(d, "lambda", low, up)

epsilon = N_A * literature["sigma"] / np.log(10) / 1e4 # m² / mol
epsilon = unp.uarray(epsilon, 10) # TODO
print(epsilon)

# k, d = np.polyfit(literature["lambda"], epsilon, 1)
# epsilon_fit = k*literature["lambda"] + d
# epsilon_fit = unp.uarray(epsilon_fit, 0.005)
# plt.figure(figsize=(9, 4))
# plt.plot(literature["lambda"], epsilon)
# plt.errorbar(literature["lambda"], unp.nominal_values(epsilon_fit), marker="o", linestyle="", capsize=3, yerr=unp.std_devs(epsilon_fit))
# plt.tight_layout()
# plt.savefig(f"{images}/absorption.png", dpi=300)
# plt.show()
# exit()

for title, _, image, l, _, data in iodine:
    A = unp.uarray(-np.log10(data["I"]/reference[-1]["I"]), 0.1)
    c = A / (epsilon * l)

    print(title)
    print(np.mean(c) * 1e3) # mmol / m³




# A = epsilon * l * c
# epsilon = molar attenuation (aka absorptivity) (m² / mol)
# l = optical path length
# c = concentration (mol / m³)

# epsilon = N_A * sigma / (ln(10) * 10^3)