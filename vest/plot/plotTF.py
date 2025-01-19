from omas import *
import numpy as np
import sys
import os
import scipy.io
import matplotlib.pyplot as plt

def plotTF(TF):
    
    R0=TF['r0']
    myy=TF['b_field_tor_vacuum_r.data']/R0
    myx=TF['time']
    myy2=TF['coil.0.current.data']
    
    fig1=plt.figure(facecolor='white')
    plt.plot(myx,myy,label='b_field_tor')

    fig2=plt.figure(facecolor='white')
    plt.plot(myx,myy2,label='current')

    mystring="Shot: {} Run:{}".format(shot,run)
    plt.title(mystring)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    argv=sys.argv[1:]

    shot = int(argv[0])
    run = int(argv[1])

    filename='{}_{}.nc'.format(shot,run)
    ods = load_omas_nc(filename)

    TF=ods['tf']


    plotTF(TF)
