from VEST_tools import vest_load,vest_loadn
import matplotlib.pyplot as plt

shot=37194
Coils=['PF1','PF2 and 3','PF4','PF5','PF6','PF7','PF8','PF9','PF10']
nbCoil=len(Coils)
# call vest_load

Currents=[]
Legend=[]
for i in range(nbCoil):
    (time,data)=vest_loadn(shot,'{} Current'.format(Coils[i]))
    if len(data) > 1:
        Currents.append(data)
        Legend.append(Coils[i])
        rtime=time

nbfig=len(Legend)
fig=plt.figure()
for i in range(nbfig):
    plt.plot(rtime,Currents[i],label=Legend[i])
plt.legend(fontsize=10,loc='upper left')
plt.xlabel('Time [s]')
plt.ylabel('Current [A]')
plt.title('VEST PF Current')
plt.show()
