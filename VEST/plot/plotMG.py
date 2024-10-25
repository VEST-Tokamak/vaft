from omas import *
import numpy as np
import sys
import os
import scipy.io
import matplotlib.pyplot as plt

def plotMG(MG):
    nbloop=len(MG['flux_loop'])
    nbprobe=len(MG['b_field_pol_probe'])
    print(nbloop,nbprobe)
    
# plot Geometry
    xvar=np.zeros(nbloop)
    yvar=np.zeros(nbloop)
    for i in range(nbloop):
        xvar[i]=MG['flux_loop.{}.position.0.r'.format(i)]
        yvar[i]=MG['flux_loop.{}.position.0.z'.format(i)]
             
    xvar2=np.zeros(nbprobe)
    yvar2=np.zeros(nbprobe)
    for i in range(nbprobe):
        xvar2[i]=MG['b_field_pol_probe.{}.position.r'.format(i)]
        yvar2[i]=MG['b_field_pol_probe.{}.position.z'.format(i)]


    fig1=plt.figure(facecolor='white')
    myx=MG['time']
    myy=MG['diamagnetic_flux.0.data']
    plt.plot(myx,myy)

    
    fpf2=plt.figure(facecolor='white')
    myax=plt.axes()
    myax.set_aspect('equal')
    plt.scatter(xvar,yvar,lw=1,label='FL position')
    plt.scatter(xvar2,yvar2,lw=1,label='Probe position')

    mystring="Shot: {} Run:{}".format(shot,run)
    plt.title(mystring)
    plt.legend()

    Color=['b','g','r','c','m','y','b','g','r','c','m','y']

    if len(MG['flux_loop[0].flux.data'])>0:
        time=MG['time']

        pf1=MG['flux_loop[0].flux.data']
        pf2=MG['flux_loop[4].flux.data']
        pf3=MG['flux_loop[9].flux.data']
        fpf2=plt.figure(facecolor='white')
        plt.plot(time,pf1,lw=2,color=Color[0],label='Current {}'.format(MG['flux_loop[0].name']))
        plt.plot(time,pf2,lw=2,color=Color[1],label='Current {}'.format(MG['flux_loop[4].name']))
        plt.plot(time,pf3,lw=2,color=Color[2],label='Current {}'.format(MG['flux_loop[9].name']))
#        plt.axis([0.,0.03,-20.,20.])
        mystring="Shot: {} Run:{}".format(shot,run)
        plt.title(mystring)
        plt.legend()


    if len(MG['b_field_pol_probe[0].field.data'])>0:
        time=MG['time']

        pf1=MG['b_field_pol_probe[10].field.data']
        pf2=MG['b_field_pol_probe[30].field.data']
        pf3=MG['b_field_pol_probe[60].field.data']
        pf4=MG['b_field_pol_probe[63].field.data']
        fpf2=plt.figure(facecolor='white')
        plt.plot(time,pf1,lw=2,color=Color[0],label='Bz {}'.format(MG['b_field_pol_probe[10].name']))
        plt.plot(time,pf2,lw=2,color=Color[1],label='Bz {}'.format(MG['b_field_pol_probe[30].name']))
        plt.plot(time,pf3,lw=2,color=Color[2],label='Bz {}'.format(MG['b_field_pol_probe[60].name']))
        plt.plot(time,pf4,lw=2,color=Color[3],label='Bz {}'.format(MG['b_field_pol_probe[63].name']))
#        plt.axis([0.,0.03,-20.,20.])
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

    MG=ods['magnetics']


    plotMG(MG)
