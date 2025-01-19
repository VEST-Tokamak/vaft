from omas import *
import numpy as np
import sys
import os
import scipy.io
import matplotlib.pyplot as plt
from omas.omas_structure import add_extra_structures

def plot_mag(MG):
    nbloop=len(MG['flux_loop'])
    nbprobe=len(MG['b_field_pol_probe'])
    print(nbloop,nbprobe)

    MGtime=MG['time']
    # flux_loop
    nbp=len(MG['flux_loop'])
    MGpsi=[]
    psi=[]
    Label=[]
    for i in range(nbp):
        psi.append(MG['flux_loop.{}.flux.reconstructed'.format(i)])
        MGpsi.append(MG['flux_loop.{}.flux.data'.format(i)])
        Label.append(MG['flux_loop.{}.name'.format(i)])

    nbp2=len(MG['b_field_pol_probe'])
    bz=[]
    MGbz=[]
    BLabel=[]
    for i in range(nbp2):
        bz.append(MG['b_field_pol_probe.{}.field.reconstructed'.format(i)])
        MGbz.append(MG['b_field_pol_probe.{}.field.data'.format(i)])
        BLabel.append(MG['b_field_pol_probe.{}.name'.format(i)])
    
    Color=['b','g','r','c','m','y','b','g','r','c','m','y']
        
    # Flux
    f1=plt.figure(facecolor='white')
    for i in range(6):
        plt.plot(MGtime,psi[i],color=Color[i])
        plt.plot(MGtime,MGpsi[i],color=Color[i],linestyle='dashed',label=Label[i])
    mystring="Shot: {} Run:{}".format(shot,run)
    plt.title(mystring)
    plt.legend()
    
    f2=plt.figure(facecolor='white')
    for i in range(6,11):
        plt.plot(MGtime,psi[i],color=Color[i])
        plt.plot(MGtime,MGpsi[i],color=Color[i],linestyle='dashed',label=Label[i])
    mystring="Shot: {} Run:{}".format(shot,run)
    plt.title(mystring)
    plt.legend()

    # Bz
    for j in range(12):
        fb1=plt.figure(facecolor='white')
        for i in range(5):
            plt.plot(MGtime,bz[i+j*5],color=Color[i])
            plt.plot(MGtime,MGbz[i+j*5],color=Color[i],linestyle='dashed',label=BLabel[i+j*5])
        mystring="Shot: {} Run:{}".format(shot,run)
        plt.title(mystring)
        plt.legend()

        fb1=plt.figure(facecolor='white')
    j=12
    for i in range(4):
        plt.plot(MGtime,bz[i+j*5],color=Color[i])
        plt.plot(MGtime,MGbz[i+j*5],color=Color[i],linestyle='dashed',label=BLabel[i+j*5])
    mystring="Shot: {} Run:{}".format(shot,run)
    plt.title(mystring)
    plt.legend()

    
    plt.show()



if __name__ == "__main__":
    argv=sys.argv[1:]

    shot = int(argv[0])
    run = int(argv[1])

    # OMAS extra_structures
    _extra_structures = {
        'magnetics': {
            "magnetics.b_field_pol_probe[:].field.reconstructed": {
                "coordinates": ["magnetics.b_field_pol_probe[:].field.time"],
                "documentation": "value calculated from the reconstructed magnetics",
                "data_type": "FLT_1D",
                "units": "T",
                "cocos_signal": "?",
            },
            "magnetics.flux_loop[:].flux.reconstructed": {
                "coordinates": ["magnetics.flux_loop[:].flux.time"],
                "documentation": "value calculated from the reconstructed magnetics",
                "data_type": "FLT_1D",
                "units": "Wb",
                "cocos_signal": "?",
            }
        }
    }
    add_extra_structures(_extra_structures)
    
    filename='{}_{}.nc'.format(shot,run)
    ods = load_omas_nc(filename)

    MG=ods['magnetics']


    plot_mag(MG)
