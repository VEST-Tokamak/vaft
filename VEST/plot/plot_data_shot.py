from VEST_tools import vest_load,vest_loadn
import matplotlib.pyplot as plt

shot=[37194,37195,37196,37197,37198,37199]

Data=['PF1 Current','PF2 and 3 Current','PF4 Current','PF5 Current','PF6 Current','PF7 Current','PF8 Current','PF9 Current','PF10 Current','TF Current']
Xlabel='Time [s]'
Ylabel='Current [A]'


def Title(k):
    T = 'VEST PF Current, shot number :'
    T = T.lstrip()
    T = T.rstrip()
    T = T.ljust(31)
    T = T+str(shot[k])
    return (T)

nbData=len(Data)
nbshot=len(shot)

# call vest_load

Datas=[]
Times=[]
Legend=[]

for k in range(nbshot):

    for i in range(nbData):

        (time,data)=vest_loadn(shot[k],Data[i])

        if len(data) > 1:

            Datas.append(data)

            Legend.append(Data[i])

            Times.append(time)


    #print(time[4999],time[8999])

    nbfig=len(Legend)
    fig=plt.figure()

    for i in range(nbfig):

        plt.plot(Times[i],Datas[i],label=Legend[i])

        plt.legend(fontsize=10,loc='upper left')
        plt.xlabel(Xlabel)
        plt.ylabel(Ylabel)
        plt.title(Title(k))

    Datas=[]
    Times=[]
    Legend=[]

plt.show()