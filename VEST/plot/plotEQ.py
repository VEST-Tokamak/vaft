# plot Psi, Br and Bz stored in a EQ ODS
# python plotEQ.py shot run time_index
# python plotEQ.py 37194 
from omas import *
import numpy as np
import sys
import os
import scipy.io
import matplotlib.pyplot as plt
from omas.omas_structure import add_extra_structures

def plotEQ(EQ,inc):
#    plt.scatter(xvar,yvar,lw=1,label='FL position')
#    plt.scatter(xvar2,yvar2,lw=1,label='Probe position')
    nbt=len(EQ['time_slice'])
    print(nbt)
    print(EQ['ids_properties.homogeneous_time'])
    if inc > nbt:
        inc=nbt
    temps=EQ['time'][inc-1]
    print('Time index:{}/{} - {} s'.format(inc,nbt,temps))
    xvar=EQ['time_slice.{}.profiles_2d.0.grid.dim1'.format(inc-1)]
    zvar=EQ['time_slice.{}.profiles_2d.0.grid.dim2'.format(inc-1)]
    psi=EQ['time_slice.{}.profiles_2d.0.psi'.format(inc-1)]
    br=EQ['time_slice.{}.profiles_2d.0.b_field_r'.format(inc-1)]
    bz=EQ['time_slice.{}.profiles_2d.0.b_field_z'.format(inc-1)]
    fpsi=plt.figure(facecolor='white')
    myax=plt.axes()
    myax.set_aspect('equal')
    #    print(len(xvar),len(zvar),len(psi),len(psi[0]))
    #    if len(psi) != len(zvar):
    psi=psi.T
    br=br.T
    bz=bz.T
    plt.contourf(xvar,zvar,psi)
    mystring="Shot: {} Run:{} Time:{}".format(shot,run,temps)
    plt.colorbar()
    plt.title(mystring)
    #    plt.legend()

    r=xvar
    z=zvar
    Nr=len(r)
    Nz=len(z)

    (r2,z2) = np.meshgrid(r,z)

    shape=(Nz,Nr)
    ndecay=np.zeros(shape)
    dBZ=np.zeros((Nr,Nz))
    dr=r[1]-r[0]
    for t in range(nbt):
        BZ=bz
        for i  in range(Nr-1):
            dBZ[i]=(BZ.T[i+1]-BZ.T[i])/dr
        ndecay = -r2/BZ * dBZ.T

    fbr=plt.figure(facecolor='white')
    myax=plt.axes()
    myax.set_aspect('equal')
    plt.contourf(xvar,zvar,br)
    mystring="Shot: {} Run:{} Time:{}".format(shot,run,temps)
    plt.colorbar()
    plt.title(mystring)

    fbz=plt.figure(facecolor='white')
    myax=plt.axes()
    myax.set_aspect('equal')
    plt.contourf(xvar,zvar,bz)
    mystring="Shot: {} Run:{} Time:{}".format(shot,run,temps)
    plt.colorbar()
    plt.title(mystring)

    
#    fn=plt.figure(facecolor='white')
#    myax=plt.axes()
#    myax.set_aspect('equal')
#    plt.contourf(xvar,zvar,ndecay)
#    mystring="Shot: {} Run:{}".format(shot,run)
#    plt.colorbar()
#    plt.title(mystring)


    
        
    plt.show()


    

if __name__ == "__main__":
    argv=sys.argv[1:]

    shot = int(argv[0])
    run = int(argv[1])
    inc= int(argv[2])
# new data (centroid) are createad in the equilibrium ODS when the ODS is generated from geqdsk files
    _extra_structures = {
        'equilibrium': {
            'equilibrium.time_slice.:.profiles_1d.centroid.r_max': {
                "full_path": "equilibrium/time_slices(itime)/profiles_1d/centroid.r_max(:)",
                "coordinates": ['equilibrium.time_slice[:].profiles_1d.psi'],
                "data_type": "FLT_1D",
                "description": "centroid r max",
                "units": 'm',
                "cocos_signal": '?'  # optional
            },
            'equilibrium.time_slice.:.profiles_1d.centroid.r_min': {
                "full_path": "equilibrium/time_slices(itime)/profiles_1d/centroid.r_min(:)",
                "coordinates": ['equilibrium.time_slice[:].profiles_1d.psi'],
                "data_type": "FLT_1D",
                "description": "centroid r min",
                "units": 'm',
                "cocos_signal": '?'  # optional
            },
            'equilibrium.time_slice.:.profiles_1d.centroid.r': {
                "full_path": "equilibrium/time_slices(itime)/profiles_1d/centroid.r(:)",
                "coordinates": ['equilibrium.time_slice[:].profiles_1d.psi'],
                "data_type": "FLT_1D",
                "description": "centroid r",
                "units": 'm',
                "cocos_signal": '?'  # optional
            },
            'equilibrium.time_slice.:.profiles_1d.centroid.z': {
                "full_path": "equilibrium/time_slices(itime)/profiles_1d/centroid.z(:)",
                "coordinates": ['equilibrium.time_slice[:].profiles_1d.psi'],
                "data_type": "FLT_1D",
                "description": "centroid z",
                "units": 'm',
                "cocos_signal": '?'  # optional
            }
        }
    }
    add_extra_structures(_extra_structures)
    
    filename='{}_{}.nc'.format(shot,run)
    ods = load_omas_nc(filename)

    EQ=ods['equilibrium']


    plotEQ(EQ,inc)
