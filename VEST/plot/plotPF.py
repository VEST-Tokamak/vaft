from omas import *
import numpy as np
import sys
import os
import scipy.io
import matplotlib.pyplot as plt

def plotV(PF,PFP):
    nbcoil=len(PF['coil'])
    nbloop=len(PFP['loop'])

# plot Geometry
    nbtot=0
    for i in range(nbcoil):
        nbelt=len(PF['coil.{}.element'.format(i)])
        nbtot=nbtot+nbelt
    xvar=np.zeros(nbtot)
    yvar=np.zeros(nbtot)
    cpt=0
    for i in range(nbcoil):
        nbelt=len(PF['coil.{}.element'.format(i)])
        for j in range(nbelt):
            xvar[cpt]=PF['coil.{}.element.{}.geometry.rectangle.r'.format(i,j)]
            yvar[cpt]=PF['coil.{}.element.{}.geometry.rectangle.z'.format(i,j)]
            cpt=cpt+1
            
    nbtot=4*nbloop
    xvar2=[]
    yvar2=[]
    for i in range(nbloop):
        nbelt2=len(PFP['loop.{}.element'.format(i)])
        for k in range(nbelt2):
            if PFP['loop.{}.element.{}.geometry.geometry_type'.format(i,k)]==1:
                for j in range(4):
                    xvar2.append(PFP['loop.{}.element.0.geometry.outline.r'.format(i)][j])
                    yvar2.append(PFP['loop.{}.element.0.geometry.outline.z'.format(i)][j])
            elif PFP['loop.{}.element.{}.geometry.geometry_type'.format(i,k)]==2:
                xvar2.append(PFP['loop.{}.element.0.geometry.rectangle.r'.format(i)])
                yvar2.append(PFP['loop.{}.element.0.geometry.rectangle.z'.format(i)])

    fpf2=plt.figure(facecolor='white')
    myax=plt.axes()
    myax.set_aspect('equal')
    plt.scatter(xvar,yvar,lw=1,label='Coil position')
    plt.scatter(xvar2,yvar2,lw=1,label='VV position')

    mystring="Shot: {} Run:{}".format(shot,run)
    plt.title(mystring)
    plt.legend()

    nbCoil=len(PF['coil'])
    Color=['b','g','r','c','m','y','b','g','r','c','m','y']

    if len(PF['coil[0].current.data'])>0:
        time=PF['time']
        if len(time) == 0:
            time=PF['coil[0].current.time']

        pf1=PF['coil[0].current.data']
        pf2=PF['coil[4].current.data']
        pf3=PF['coil[9].current.data']
        fpf2=plt.figure(facecolor='white')
        plt.plot(time,pf1,lw=2,color=Color[0],label='Current {}'.format(PF['coil[0].name']))
        plt.plot(time,pf2,lw=2,color=Color[1],label='Current {}'.format(PF['coil[4].name']))
        plt.plot(time,pf3,lw=2,color=Color[2],label='Current {}'.format(PF['coil[9].name']))
#        plt.axis([0.,0.03,-20.,20.])
        mystring="Shot: {} Run:{}".format(shot,run)
        plt.title(mystring)
        plt.legend()


    if len(PF['coil[0].voltage.data'])>0:
        time=PF['time']
        if len(time) == 0:
            time=PF['coil[0].voltage.time']

        pf1=PF['coil[0].voltage.data']
        pf2=PF['coil[4].voltage.data']
        pf3=PF['coil[9].voltage.data']
        fpf2=plt.figure(facecolor='white')
        plt.plot(time,pf1,lw=2,color=Color[0],label='Voltage {}'.format(PF['coil[0].name']))
        plt.plot(time,pf2,lw=2,color=Color[1],label='Voltage {}'.format(PF['coil[4].name']))
        plt.plot(time,pf3,lw=2,color=Color[2],label='Voltage {}'.format(PF['coil[9].name']))
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

    PF=ods['pf_active']
    PFP=ods['pf_passive']

    plotV(PF,PFP)
