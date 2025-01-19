from omas import *
from omfit_classes.omfit_eqdsk import OMFITgeqdsk
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.io import netcdf
    
def plotEQ(EQs,shots,runs):
    name=f'{shots[0]}_{runs[0]}'
    nbODS=len(EQs)
    fig = plt.figure(figsize=(10,8))
    fig.suptitle(name, fontsize=16)
#    ax = fig.add_subplot()
#    fig.subplots_adjust(top=0.85)
#    ax.axis([0, 10, 0, 10])
#    ax.text(1,2, 'Experimental data',color='blue', fontsize=15)
    
    ax1 = plt.subplot2grid((5, 4), (0, 0))
    ax2 = plt.subplot2grid((5, 4), (0, 1))
    ax3 = plt.subplot2grid((5, 4), (0, 2))
    ax4 = plt.subplot2grid((5, 4), (0, 3))
    ax5 = plt.subplot2grid((5, 4), (2, 0))
    ax6 = plt.subplot2grid((5, 4), (2, 1))
    ax7 = plt.subplot2grid((5, 4), (2, 2))
    ax8 = plt.subplot2grid((5, 4), (2, 3))
    ax9 = plt.subplot2grid((5, 4), (4, 0))
    ax10 = plt.subplot2grid((5, 4), (4, 1))
    ax11 = plt.subplot2grid((5, 4), (4, 2))
    ax11.axis('off')


    Color=['blue','orange','green','red','purple','pink','gray','olive','cyan']
    
    for j in range(nbODS):
        EQ=EQs[j]
        time=EQ['time']
        nbt=len(time)
        B_normal=[]
        B_pol=[]
        B_tor=[]
        b_field=[]
        R=[]
        a=[]
        elong=[]
        triang=[]
        Ip=[]
        li_3=[]
        for i in range(nbt):
            B_normal.append(EQ[f'time_slice.{i}.global_quantities.beta_normal'])
            B_pol.append(EQ[f'time_slice.{i}.global_quantities.beta_pol'])
            B_tor.append(EQ[f'time_slice.{i}.global_quantities.beta_tor'])
            b_field.append(EQ[f'time_slice.{i}.global_quantities.magnetic_axis.b_field_tor'])
            R.append(EQ[f'time_slice.{i}.global_quantities.magnetic_axis.r'])
            RR=EQ[f'time_slice.{i}.boundary.outline.r']
            a.append((max(RR)-min(RR))/2)
            elong.append(EQ[f'time_slice.{i}.profiles_1d.elongation'][-1])
            triang.append(EQ[f'time_slice.{i}.profiles_1d.triangularity_upper'][-1])
            Ip.append(EQ[f'time_slice.{i}.global_quantities.ip'])
            li_3.append(EQ[f'time_slice.{i}.global_quantities.li_3'])
    
        ax1.scatter(time, B_normal,color=Color[j])
        ax1.set_title('Beta_normal', fontsize=10)

        ax2.scatter(time, B_pol,color=Color[j])
        ax2.set_title('Beta_pol', fontsize=10)

        ax3.scatter(time, B_tor,color=Color[j])
        ax3.set_title('Beta_tor', fontsize=10)

        ax4.scatter(time, b_field,color=Color[j])
        ax4.set_title('b_field_tor', fontsize=10)

        ax5.scatter(time, R,color=Color[j])
        ax5.set_title('R', fontsize=9)

        ax6.scatter(time, a,color=Color[j])
        ax6.set_title('a', fontsize=9)

        ax7.scatter(time, elong,color=Color[j])
        ax7.set_title('elongation', fontsize=10)

        ax8.scatter(time, triang,color=Color[j])
        ax8.set_title('triangularity', fontsize=9)

        ax9.scatter(time, Ip,color=Color[j])
        ax9.set_title('Ip', fontsize=10)

        ax10.scatter(time, li_3,color=Color[j])
        ax10.set_title('li_3', fontsize=10)

        ax11.text(0,j*0.2,f'Shot: {shots[j]} - {runs[j]}',color=Color[j], fontsize=10)


    plt.savefig(f'C{name}.png')
    print(f'C{name}.png generated')
    
    #display plots
    plt.show()
        
        
if __name__ == "__main__":
    argv=sys.argv[1:]

    nbODS=int(len(argv)/2)
    shots=[]
    runs=[]
    for i in range(nbODS):
        shots.append(int(argv[2*i]))
        runs.append(int(argv[2*i+1]))

    EQs=[]
    for i in range(nbODS):
        filename=f'{shots[i]}_{runs[i]}.json'
        ods = load_omas_json(filename)
        EQs.append(ods['equilibrium'])

    
    plotEQ(EQs,shots,runs)
