import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy import signal
import matplotlib.pyplot as plt
import os
import modules.useful as use
import modules.latextools as lt
import uncertainties as un
import uncertainties.umath as um
import math as m
from uncertainties import unumpy as unp
path_data = "C:\\Lab_Master\\MTPA\\MTPA_LAB\\Latex\\Absorption_Spectroscopy\\data\\"
path_graphics = "C:\\Lab_Master\\MTPA\\MTPA_LAB\\Latex\\Absorption_Spectroscopy\\graphics\\"


def plot_spectra(path, data ,ax ,label = 'data',x = 'Wavelength', y = 'Intensity',delimiter = '\t'):
    if type(data) == str:
        spectra = pd.read_csv(path+data,delimiter= delimiter)
    else:
        spectra = data
    I = spectra[y]
    Lambda = spectra[x]
    ax.plot(Lambda,I, label = label )
    ax.set_ylabel(y)
    ax.set_xlabel(x)
    ax.legend()


def quick_plot_files(File_split):
    for files in File_split:
        plt.style.use('bmh')
        fig , ax = plt.subplots(3, figsize=[16,9])
        for i,file in enumerate(files):
            plot_spectra(path_data,file,ax[i],label=file)
        #fig.savefig(path_graphics+file +".pdf")
        plt.show()
        fig.clf()
gaussian = lambda x,A, mu,sig: A*np.exp(-0.5 * ((x - mu) / sig) ** 2)
gaussian_u = lambda x,A, mu,sig: A*um.exp(-0.5 * ((x - mu) / sig) ** 2)
cauchy = lambda x,amp,mu,fhwm: amp *( (fhwm**2)/(fhwm**2 + 4*(x-mu)**2))
def total_gaussian(x, *params):
    total = np.zeros_like(x)
    for i in range(0, len(params), 3):
        mean, std_dev, amplitude = params[i:i+3]
        total += gaussian(x, mean, std_dev, amplitude)
    return total

def total_cauchy(x, *params):
    total = np.zeros_like(x)
    for i in range(0, len(params), 3):
        mean, std_dev, amplitude = params[i:i+3]
        total += cauchy(x, mean, std_dev, amplitude)
    return total
def add_functions(x,func,*params):
    total_func = un.ufloat(0,0)
    for i in range(0, len(params), 3):
        amplitude ,mean, std_dev = params[i:i+3]
        total_func += func(x, amplitude, mean, std_dev)
    return total_func

def fit_spectra(data,guesses=None):
    I = data["Intensity"]
    Lamda = data["Wavelength"]
    I = I/max(I)
    if guesses == None:
        guesses = []
        peaks, _ = find_peaks(I,width = 2,prominence=0.1)
        guessed_means = Lamda[peaks]
        for amplitude,mean in zip(I[peaks],guessed_means):
            guesses.extend([amplitude,mean,0.1])
        print(guesses)
    params, error = curve_fit(total_gaussian,Lamda,I,p0=guesses)
    params_cauchy, error_cauchy = (0,0)

    return params, error,  params_cauchy, error_cauchy
#import scipy.signal as signal
#b, a = signal.butter(3,0.1)
spectra_green = pd.read_csv(path_data+"2-only-green.txt",delimiter= "\t")
spectra_IR = pd.read_csv(path_data+"1-blende-offen.txt",delimiter= "\t")
spectra_both = pd.read_csv(path_data+"2-IR_and_green.txt",delimiter= "\t")
spectra_ir_shg = pd.read_csv(path_data+"2-only-IR.txt",delimiter= "\t")
#spectra_green["Intensity"] = signal.filtfilt(b, a, spectra_green["Intensity"])
guesses_IR=[
    0.8,1015.73,3.8,
    0.55,1021,1,
    0.1,1028,1,
    0.4,1035,1,
    0.6,1040,1,
    0.6,1048,3.8,
    0.6,1050,1
            ]
guesses_green=[
    0.5, 509, 0.1,
    0.5, 511, 0.1,
    1, 511.495601173, 0.1,
    0.486254133207369, 514.3, 0.1,
    0.1261848527791214, 517.9765395894, 0.1, 
    0.21993386868209733, 521.442521994, 0.1,
    0.1, 524, 0.1
    ]
#spectra_ir_shg["Intensity"] = signal.savgol_filter(np.abs(spectra_ir_shg["Intensity"] -175*0.174),15,1)
baseline = np.mean(spectra_ir_shg["Intensity"][int(0.6*len(spectra_ir_shg["Intensity"])):])
spectra_ir_shg["Intensity"] = np.abs(spectra_ir_shg["Intensity"] -baseline)

print()
#plt.plot(spectra_ir_shg["Wavelength"], spectra_ir_shg["Intensity"])
#plt.show()
param_IR_G ,errors_IR_G, _ , _  = fit_spectra(spectra_IR,guesses=guesses_IR)
param_green_G ,errors_green_G, _ , _ = fit_spectra(spectra_green,guesses=guesses_green)
param_IR_SHG ,errors_IR_SHG, _ , _  = fit_spectra(spectra_ir_shg,guesses=guesses_IR[:-6])
b = True
a = False
paramsshg = [param_green_G,param_IR_SHG]
errorsshg = [errors_green_G,errors_IR_SHG]
spectrashg = [spectra_green,spectra_ir_shg]
if a: 
    
    
    #print("mean ir", np.mean(split[1]))
    

    #param_auto_G ,errors_auto_G, _ , _ = fit_spectra(spectra_green)
    
    params_IR_un = unp.uarray(param_IR_G,np.sqrt(np.diag(errors_IR_G)))
    
    split = np.array(np.split(params_IR_un,7)).T
    print(f"IR {split} \n")
    I_ir =     np.array(spectra_IR["Intensity"])
    I_ir = I_ir/max(I_ir)
    Lambda_ir =np.array(spectra_IR["Wavelength"])
    Lambda_ir_u = unp.uarray(Lambda_ir,[np.abs(Lambda_ir[0]- Lambda_ir[1])]*len(Lambda_ir))
    
    
    fullspectra_ir_u = [add_functions(lam,gaussian_u,*params_IR_un) for lam in Lambda_ir_u]


    central_ir  = np.sum(fullspectra_ir_u*Lambda_ir_u/np.sum(fullspectra_ir_u))
    print(split)
    FHWM_lower = split[1][0]- np.sqrt(2*np.log(2))*split[2][0]
    FHWM_upper = split[1][-1] + np.sqrt(2*np.log(2))*split[2][-1]
    plt.figure(figsize=(12,4.666))
    j = 0
    for i in range(0, len(params_IR_un), 3):
            amplitude,mean, std_dev = params_IR_un[i:i+3]
            plt.plot(Lambda_ir,gaussian(Lambda_ir, amplitude.n, mean.n, std_dev.n),label = f"$G_{j}(\lambda_p = ({mean.n:.2f} \pm {mean.s:.2f})$" + " nm)")
            j += 1
    plt.plot(Lambda_ir,I_ir,"r-", label = "Data Fundamental Spectrum")
    use.un_plot_simple(Lambda_ir_u,fullspectra_ir_u,format= "b--", show_error="fill_between", label = "$G_{comb}(\lambda)$ Fit", unlabel= "$G_{comb}(\lambda)$ error")
    plt.vlines([central_ir.n],0,1,linestyles="--",label=f"$\lambda_c = ({central_ir.n:.2f} \pm {central_ir.s:.2f})$ nm")
    plt.vlines([FHWM_lower.n,FHWM_upper.n], 0,1,linestyles="--", label = "$FHWM_{total}$ = " + f"({(FHWM_upper - FHWM_lower).n:.2f} $\pm$ {(FHWM_upper - FHWM_lower).s:.2f}) nm")
    #plt.xticks(np.arange(min(Lambda_ir), max(Lambda_ir)+1, 1.0))
    plt.xlabel("$\lambda$ / nm")
    plt.ylabel("I / 1")
    plt.legend()
    lt.latextable(pd.DataFrame(split).T,"c")
    #plt.savefig(path_graphics+"IR_fitteddx.pdf",dpi=300)

if b:
    all_speck = []
    all_params = []
    #fig_conv , ax_conv = plt.subplots(2, figsize=(11,9))
    k = 0
    for p, e, speck in zip(paramsshg,errorsshg,spectrashg):
        
        p_un = unp.uarray(p,np.sqrt(np.diag(e)))
        fig_conv , ax_conv = plt.subplots(1, figsize=(9,4))
        ax_conv.set_ylim(0,1.5)
        #print(np.sum(spectra_green["Intensity"])/(np.sum(spectra_ir_shg["Intensity"])+np.sum(spectra_green["Intensity"])))
        if k == 0:
            split2 = np.array(np.split(p_un,7)).T
            I_g =      np.array(speck["Intensity"])
        if k == 1:
            split2 = np.array(np.split(p_un,5)).T
            ax_conv.set_ylim(0,1)
            I_g =      np.array(speck["Intensity"])
        all_params.append(split2)
        #print("mean grean", np.mean(split2[1]))
        #param_auto_G ,errors_auto_G, _ , _ = fit_spectra(spectra_green)
        
        
        
        
        I_g = I_g/max(I_g)

        Lambda = np.array(speck["Wavelength"])
        Lambda_u = unp.uarray(Lambda,[np.abs(Lambda[0]- Lambda[1])]*len(Lambda))
        
        fullspectra_u = [add_functions(lam,gaussian_u,*p_un) for lam in Lambda_u]
        all_speck.append(fullspectra_u)
        central_wl = np.sum(fullspectra_u*Lambda_u/np.sum(fullspectra_u))
        
        j = 0
        for i in range(0, len(p_un), 3):
                amplitude,mean, std_dev = p_un[i:i+3]
                ax_conv.plot(Lambda,gaussian(Lambda, amplitude.n, mean.n, std_dev.n), label = f"$G_{j}(\lambda_p = {mean.n:.2f} \pm {mean.s:.2f})$" + " nm")
                j += 1
        ax_conv.plot(Lambda,I_g,"r-",label = "Data SHG")
        ax_conv.vlines([central_wl.n], 0,1,linestyles="--", label=f"$\lambda_c$ = ({central_wl.n:.2f} $\pm$ {central_wl.s:.2f}) nm")
        use.un_plot(ax_conv,Lambda_u,fullspectra_u,x_label= "$\lambda$ / nm",y_label= "I / 1",format= "b--", show_error="fill_between", label = "$G_{comb}(\lambda)$ Fit", unlabel= "$G_{comb}(\lambda)$ error")
        #ax_conv[k].xaxis.set_ticks(np.arange(min(Lambda), max(Lambda)+1, 1))
        ax_conv.legend()
        k +=1
        #fig_conv.savefig(path_graphics+f"SHG_fitteddx{k}.pdf",dpi=300)
        #fig_conv.clf()

    for i in all_params:
        i = pd.DataFrame(i).T
        lt.latextable(i,"c")
    print(np.sum(max(spectra_green["Intensity"])*np.array(all_speck[0]))/(np.sum(max(spectra_green["Intensity"])*np.array(all_speck[0]))+ np.sum(max(spectra_ir_shg["Intensity"])*np.array(all_speck[1]))), "speck")
    a1 = un.ufloat(0,0)
    a2 = un.ufloat(0,0)
    for i in all_speck[0]:
        a1+=i
        
    for j in all_speck[1]:
        a2+=j
        #print(a2)
    print(a1/(a1+a2))