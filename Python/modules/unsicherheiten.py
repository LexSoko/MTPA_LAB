import numpy as np
import statistics as st

def studentT(dataSeries:np.array,t=1):
    a = np.linspace(0,1,10)
    n = len(dataSeries)

    mean = st.mean(dataSeries)
    error = st.stdev(dataSeries) * t / np.sqrt(n)

    return error

def _test_studentT_():
    a1 = np.linspace(0,1,10)
    a2 = np.linspace(15,16*np.pi,10)
    a1_mean, a1_err = studentT(a1)
    a1_mean_v, a1_err_v = _studentT_vergleich_(a1)
    a2_mean, a2_err = studentT(a2)
    a2_mean_v, a2_err_v = _studentT_vergleich_(a2)
    if( a1_mean == a1_mean_v and a1_err == a1_err_v and a2_mean == a2_mean_v and a2_err == a2_err_v):
        print("success")
    else:
        print("fail")


def _studentT_vergleich_(dataSeries:np.array,t=1):
    summe = 0
    anzahl = 0
    for x in dataSeries:
        summe += x
        anzahl += 1
    arith_mittel = summe/anzahl

    summe_std_dev = 0
    for x in dataSeries:
        summe_std_dev += (x-arith_mittel)**2
    std_dev = np.sqrt(summe_std_dev / (anzahl-1))
    std = std_dev / np.sqrt(anzahl)
    error = std * t

    return arith_mittel, error