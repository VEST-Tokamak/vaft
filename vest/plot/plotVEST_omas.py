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

    xvar2=[]
    yvar2=[]
    for i in range(nbloop):
        nbelt2=len(PFP['loop.{}.element'.format(i)])
        for k in range(nbelt2):
            if PFP['loop.{}.element.{}.geometry.geometry_type'.format(i,k)]==1:
                for j in range(4):
                    xvar2.append(PFP['loop.{}.element.{}.geometry.outline.r'.format(i,k)][j])
                    yvar2.append(PFP['loop.{}.element.{}.geometry.outline.z'.format(i,k)][j])
            elif PFP['loop.{}.element.{}.geometry.geometry_type'.format(i,k)]==2:
                xvar2.append(PFP['loop.{}.element.{}.geometry.rectangle.r'.format(i,k)])
                yvar2.append(PFP['loop.{}.element.{}.geometry.rectangle.z'.format(i,k)])

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

        fpf=plt.figure(facecolor='white')
        for j in range(nbCoil):
            pf=PF['coil.{}.current.data'.format(j)]
            plt.plot(time,pf,lw=2,color=Color[j],label='Current {}'.format(PF['coil.{}.name'.format(j)]))
#        plt.axis([0.,0.03,-20.,20.])
        mystring="Shot: {} Run:{}".format(shot,run)
        plt.title(mystring)
        plt.legend()

    try:
        print(len(PFP['loop.0.current']))
        ok=1
    except:
        ok=0

    if ok==1:
        time=PFP['time']
        if len(time) == 0:
            time=PF['time']

        pf1=PFP['loop[0].current']
        pf2=PFP['loop[5].current']
        pf3=PFP['loop[9].current']
        fpf3=plt.figure(facecolor='white')
        plt.plot(time,pf1,lw=2,color=Color[0],label='Current {}'.format(PFP['loop[0].name']))
        plt.plot(time,pf2,lw=2,color=Color[1],label='Current {}'.format(PFP['loop[5].name']))
        plt.plot(time,pf3,lw=2,color=Color[2],label='Current {}'.format(PFP['loop[9].name']))
#        plt.axis([0.,0.03,-20.,20.])
        mystring="Shot: {} Run:{}".format(shot,run)
        plt.title(mystring)
        plt.legend()

        nbloop=len(PFP['loop'])
        k=-1
        for i in range(len(time)):
            if time[i]> 0.31 and k ==-1:
                k=i
        tot=0.
        for i in range(nbloop):
            if PFP['loop.{}.name'.format(i)]=='W1':
                tot=tot+PFP['loop.{}.current'.format(i)][k]
                print(PFP['loop.{}.current'.format(i)][k])
        print(tot)
        
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
