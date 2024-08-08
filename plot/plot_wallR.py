from omas import *
import numpy as np
import sys
import os
import scipy.io
import matplotlib.pyplot as plt

def plotW(ods,shot,run):
    nbrun=len(run)
    Rin=[]
    Rout=[]
    Wnum=[0,50,100,150,200,750,800,850,900,949]
    nbnum=len(Wnum)
    Inum=[]

    times=[]
    for i in range(nbrun):
        ODS=ods[i]
        PFP=ODS['pf_passive']
        times.append(PFP['time'])
        nbloop=len(PFP['loop'])

        rin=[]
        rout=[]
        for j in range(nbloop):
            if PFP[f'loop.{j}.name']=='W1': # outboard
                rout.append(PFP[f'loop.{j}.resistance'])
            if PFP[f'loop.{j}.name']=='W11': # inboard
                rin.append(PFP[f'loop.{j}.resistance'])
        nbin=len(rin)
        nbout=len(rout)
            
        Rout.append(rout)
        Rin.append(rin)

        iwall=[]
        for j in range(nbnum):
            iwall.append(PFP[f'loop.{j}.current'])
        Inum.append(iwall)
        

    Color=['b','g','r','c','m','y','b','g','r','c','m','y','b','g','r','c','m','y','b','g','r','c','m','g','r','c','m','y','b','g','r','c','m','y','b','g','r','c','m','y','b','g','r','c','m']

    xin=np.arange(nbin)
    xout=np.arange(nbout)

    f1=plt.figure(facecolor='white')
    for i in range(nbrun):
        plt.plot(xin,Rin[i],lw=2,color=Color[i],label=f'{run[i]}')
    plt.title(f'{shot} - R inboard')
    plt.legend()
    f2=plt.figure(facecolor='white')
    for i in range(nbrun):
        plt.plot(xout,Rout[i],lw=2,color=Color[i],label=f'{run[i]}')
    plt.title(f'{shot} - R outboard')
    plt.legend()
    
    
    for j in range(nbnum):
        f=plt.figure(facecolor='white')
        for i in range(nbrun):
            time=times[i]
            nbt=len(time)
            yvar=Inum[i][j]
            #            if j == 0:
            #                print(yvar[int(nbt/2)],max(yvar))
            plt.plot(time,yvar,lw=2,color=Color[i],label=f'{run[i]}')
        if j<=4:
            board='outboard'
        else:
            board='inboard'
        plt.title(f'{shot} - Eddy current - {board}')
        plt.legend()

#    print(len(Inum),len(Inum[0]),nbt)

       
    plt.show()


if __name__ == "__main__":
    shot = 41516
    run = [1,2,3,4,5]
    nbrun=len(run)
    
    ods=[]
    for i in range(nbrun):
        filename='{}_{}.json'.format(shot,run[i])
        ods.append(load_omas_json(filename))

    plotW(ods,shot,run)
