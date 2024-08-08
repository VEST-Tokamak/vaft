import mysql.connector
import numpy as np
import matplotlib.pyplot as plt
from VEST_tools import vest_load
import scipy.stats
import time as timer


# id for the DB
hostname='147.46.31.16'
username='vestuser'
password='nu31123'
database='VEST'

shot_nb=0 # shot number selected

# take input and convert it to an array of shot numbers
shots_input=input("shots list : ")
shots_input=shots_input.split()
shots_list=[int(i) for i in shots_input]


# temporary arrays for status.csv
status=np.array(([]))
new=np.array(([]))

# variable containing the type of shot
str_type='null'


def is_nan_interval(array, start, end): # checks if entered interval is full of zeros

    if array[start:end+1] == [0]*(1+end-start):
        return True
    else:
        return False


def shot_type_determ(shot): # function determining the type of shot (cf. str_types)

    str_types=['vac','ohm','hi','nb','ec1','ec2']
    models=[38416,37329,35337,33071,38698,36842] # shot numbers of the models
    r=[]
    p=[]
    m=0
    tot=[0,0,0,0,0,0]

    for i in range(257): # for each sensor, we are going to compare the selected shot to the models

        if max(tot) < 129: # code optimisation, if one model as more than half of the most correlated plots, we can stop

            timer.sleep(1) # first of many timers, used to lessen the load on the DB

            try: # getting the data form the DB, cf. vest_load
                    mydb=mysql.connector.connect(host=hostname,user=username,password=password,database=database)
                    mycursor=mydb.cursor()
                    com='SELECT shotDataWaveformTime, shotDataWaveformValue FROM shotDataWaveform_2 WHERE shotCode = {} AND shotDataFieldCode = {} ORDER BY shotDataWaveformTime ASC'.format(shot,i)
                    mycursor.execute(com)
                    myresult=mycursor.fetchall()
                    result=np.array(myresult)

                    if len(result) >0:
                        time=result.T[0]
                        data=result.T[1]
                    else:
                        continue

            except KeyError:
                print('Data name not found')
                pass

            for j in range(6): # for each model, we compare the plots to the selected shot using a correlation method from scipy

                timer.sleep(1)

                try:  # getting the data form the DB, cf. vest_load
                    mydb=mysql.connector.connect(host=hostname,user=username,password=password,database=database)
                    mycursor=mydb.cursor()
                    com='SELECT shotDataWaveformTime, shotDataWaveformValue FROM shotDataWaveform_2 WHERE shotCode = {} AND shotDataFieldCode = {} ORDER BY shotDataWaveformTime ASC'.format(models[j],i)
                    mycursor.execute(com)
                    myresult=mycursor.fetchall()
                    result_j=np.array(myresult)
                    if len(result_j) >0:
                        time=result_j.T[0]
                        data=result_j.T[1]
                    else:
                        continue

                except KeyError:
                    print('Data name not found for model ',j)
                    pass

                if is_nan_interval(result_j,6001,8501)==False and is_nan_interval(result,6001,8501)==False: # making sure both the model's and the sensor's data isn't all zeros in [0.24:0.34]s, dt=40µs
                    
                    (time,test)=vest_load(shot,i)
                    (time,model_j)=vest_load(models[j],i)

                    try:
                        temp=scipy.stats.linregress(test, model_j) # linear regression to compare the standard deviations upon
                        res=temp.stderr

                    except ValueError :
                        ('len(x) != len(y), pass...')
                        temp=res=0
                        pass

                    r.append(res)

            m=r.index(min(r)) # locating the most correlated model for a particular sensor
            tot[m]+=1
            print(tot)
            r=[]
            p=[]

    m=tot.index(max(tot)) # locating the most correlated model for the entire shot's data
    str_type=str_types[m] # associating the string corresponding the the type of shot
    print(str_type)

    return(str_type)

#-------------------------------------------------------------------------------------------------------------------------#

for j in range(len(shots_list)):
    timer.sleep(1)
    shot_nb=shots_list[j]
    print('Checking for shot nb. : ',shot_nb,'...')
    str_type=shot_type_determ(shot_nb)

    if j > 0: # create a new array to be inserted later in the excel sheet

        new=np.array(([]))

    for i in range(257): 
        timer.sleep(1)

        try:  # getting the data form the DB, cf. vest_load
            mydb=mysql.connector.connect(host=hostname,user=username,password=password,database=database)
            mycursor=mydb.cursor()
            com='SELECT shotDataWaveformTime, shotDataWaveformValue FROM shotDataWaveform_2 WHERE shotCode = {} AND shotDataFieldCode = {} ORDER BY shotDataWaveformTime ASC'.format(shot_nb,i)
            mycursor.execute(com)
            myresult=mycursor.fetchall()
            result=np.array(myresult)
            if len(result) >0:
                time=result.T[0]
                data=result.T[1]
            else:
                time=[0.]
                data=[0.]

        except KeyError:
            print('Data name not found')
            time=[0.]
            data=[0.]

        if is_nan_interval(result,6001,8501)==False: # verifying the status of the sensor for the shot (if all data in non zero) in [0.24:0.34]s, dt=40µs

            Av=np.array((['Available']))
            new=np.concatenate((new,Av), axis=0) # add the status to the going to be added 'new' column

            (time,data)=vest_load(shot_nb,i) # gets the data for the selected shot
                                             # (using vest_load directly because here there is no need to manipulate the "result" array)
            fig=plt.figure()
            plt.xlim(0.24,0.34) # time interval of the experiment
            str_j=str(shot_nb) # gets the shot number as a string
            str_i=str(i) # gets the sensor number as a string
            str_name="figures/"+str_type+'_'+str_j+'_'+str_i+".jpg" # ex name of the plot : ohm_36546_54.jpg
            plt.title(str_type+'_'+str_j+'_'+str_i)
            plt.plot(time,data)
            plt.savefig(str_name)
            plt.close()

        else:
            Nc=np.array((['Not connected']))
            new=np.concatenate((new,Nc), axis=0) # add the status to the going to be added 'new' column

    if j == 0:
        status=new  # for the first shot, we cannot insert a column to an empty array, also, at this point we want status=array as only 1 shot as been treated

    if j > 0:
        status=np.insert(np.reshape(status, (257, j)), j, new, axis=1) # add the new column to the excel sheet

np.savetxt('status.csv',status,delimiter=",",fmt='%s') # saving the array as a status.csv, containing the availability of the sensors for each choosen shot