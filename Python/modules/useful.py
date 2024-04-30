import numpy as np
import uncertainties as un
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import unumpy

def un_separator(ufloat):
    if isinstance(ufloat, (list, np.ndarray, pd.DataFrame)):
        nominal = np.array([i.n for i in ufloat])
        uncertan = np.array([i.s for i in ufloat])
    elif isinstance(ufloat, un.UFloat):
        nominal = ufloat.n
        uncertan = ufloat.s
    return [nominal, uncertan]

def un_merger(nominal, uncertain):
    ufloat_array = []
    for i, j in zip(nominal,uncertain):
        ufloat_array.append(un.ufloat(i,j))
    return unumpy.uarray(ufloat_array)



def const_un(nominal: list, uncertainty: float):
    ufloat_array = [un.ufloat(i,uncertainty) for i in nominal]
    return ufloat_array



def un_plot(ax,x:list,y:list,x_label = "$x$", y_label ="$y$",title=" ",unlabel="",label=" " ,format= "b+", cap = 5 , show_error= "errorbar"):
    y, dy = un_separator(y)
    x , dx = un_separator(x)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if show_error == "errorbar":
        ax.errorbar(x,y,yerr= dy, xerr = dx, fmt=format , capsize=cap, label=label)
    if show_error == "fill_between":
        ax.fill_between(x,y-dy,y+dy,alpha=0.3,color= format[0],label=unlabel)
        ax.plot(x,y,format,label=label)
    
    



def un_plot_simple(x:list,y:list,format = "b", show_error = "errorbar", label = "", cap =  5, unlabel=""):
    y, dy = un_separator(y)
    x , dx = un_separator(x)
    if show_error == "errorbar":
        plt.errorbar(x,y,yerr= dy, xerr = dx, fmt=format , capsize=cap, label=label)
    if show_error == "fill_between":
        plt.fill_between(x,y-dy,y+dy,alpha=0.3,color=format[0],label=unlabel)
        plt.plot(x,y,format,label=label)
  
