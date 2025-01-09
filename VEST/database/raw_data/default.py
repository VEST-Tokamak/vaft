hostname = ''
username = ''
password = ''

import numpy as np
import mysql.connector
from mysql.connector.pooling import MySQLConnectionPool
import re
import time
import os

db_pool = None


def vest_getInfo():
    hostname = input("Enter the database hostname: ")
    username = input("Enter the database username: ")
    password = input("Enter the database password: ")
    database = 'VEST'

    file_path = './vest_database_info.txt'

    with open(file_path, 'w') as file:
        file.write(f"{hostname}\n")
        file.write(f"{username}\n")
        file.write(f"{password}\n")
        file.write(f"{database}\n")

def vest_configuration():
    global hostname, username, password, database 
    file_path = './vest_database_info.txt'

    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        hostname = lines[0].strip()
        username = lines[1].strip()
        password = lines[2].strip()
        database = lines[3].strip()

    else:
        vest_getInfo()



def vest_connection_pool():
    global db_pool
    if hostname == '':
        vest_configuration()
    db_pool = MySQLConnectionPool(pool_name="mypool",
                                  pool_size=4,  # 예를 들어 풀 크기를 10으로 설정
                                  host=hostname,
                                  database=database,
                                  user=username,
                                  password=password)

def vest_load_shotWaveform_2(mydb, shot, field ):
    # This fuction is to load the shot data from the VEST sql database table shotDataWaveform_2
    mycursor = mydb.cursor()
    com = 'SELECT shotDataWaveformTime, shotDataWaveformValue FROM shotDataWaveform_2 WHERE shotCode = {} AND shotDataFieldCode = {} ORDER BY shotDataWaveformTime ASC'.format(shot, field)
    mycursor.execute(com)
    myresult = mycursor.fetchall()
    result = np.array(myresult)
    if len(result) > 0:
        return (result.T[0], result.T[1])
    else:
        return ([0.], [0.])

def vest_load_shotWaveform_3(mydb, shot, field):
    # This function loads shot data from the VEST SQL database table shotDataWaveform_3.
    mycursor = mydb.cursor()
    com = 'SELECT shotDataWaveformTime, shotDataWaveformValue FROM shotDataWaveform_3 WHERE shotCode = {} AND shotDataFieldCode = {}'.format(shot, field)
    mycursor.execute(com)
    myresult = mycursor.fetchall()
    
    if len(myresult) != 1:
        print("Warning: shot: {}, field: {} has multiple signal rows. Please check error".format(shot, field))
        return (np.array([0.]), np.array([0.]))
    
    shotDataWaveformTime = myresult[0][0]
    shotDataWaveformValue = myresult[0][1]

    # Remove '[' and ']' characters
    shotDataWaveformTime = re.sub(r'[\[\]]', '', shotDataWaveformTime)
    shotDataWaveformValue = re.sub(r'[\[\]]', '', shotDataWaveformValue)

    # Split by ',' and convert to floats, then store in numpy arrays
    time = np.array([float(x) for x in shotDataWaveformTime.split(',')])
    data = np.array([float(x) for x in shotDataWaveformValue.split(',')])

    return (time, data)

def vest_load(shot, field, max_retries=3):
    #This function loads the shot data from the VEST sql database
    if hostname == '':
        vest_configuration()

    global db_pool
    retries = 0
    mycursor = None
    mydb = None
    while retries < max_retries:
        mydb = None
        try:
            # if global variable db_pool is set, get a connection from the pool
            if db_pool is not None:
                mydb = db_pool.get_connection()
            else:
                # otherwise, create a new connection
                mydb = mysql.connector.connect(
                    host=hostname, user=username, password=password, database=database)
            if shot> 29349 and shot <= 42190:
                return vest_load_shotWaveform_2(mydb, shot, field)
            elif shot > 42190:
                return vest_load_shotWaveform_3(mydb, shot, field)
            else:
                print("Shot number out of range")
                return None
        except mysql.connector.Error as err:
            print(f"Error connecting to MySQL: {err}")
            retries += 1
            time.sleep(1)
        finally:
            if mycursor:
                mycursor.close()
            if mydb and mydb.is_connected():
                mydb.close()
            # if db_pool exists, return the connection to the pool
        retries += 1

def vest_date(shot):
    # This function returns the date for the given shotCode.
    if hostname == '':
        vest_configuration()
    try:
        mydb = mysql.connector.connect(
            host=hostname, user=username, password=password, database=database)
        mycursor = mydb.cursor()

        # Query to get the date for the given shotCode
        com = "SELECT recordDateTime FROM shot WHERE shotNumber = {}".format(shot)
        
        mycursor.execute(com)
        myresult = mycursor.fetchone()
        
        if myresult is not None:
            # Extract the date from the result
            date = myresult[0].strftime('%Y-%m-%d')
        else:
            date = None

    except Exception as e:
        print('Error:', e)
        date = None

    return date

def vest_shots(date):
    # This function returns a list of 'shotCode's for the given date.
    if hostname == '':
        vest_configuration()
    try:
        mydb = mysql.connector.connect(
            host=hostname, user=username, password=password, database=database)
        mycursor = mydb.cursor()

        # Query to get all distinct shotCodes for the given date
        com = "SELECT DISTINCT shotNumber FROM shot WHERE DATE(recordDateTime) = '{}'".format(date)
        
        mycursor.execute(com)
        myresult = mycursor.fetchall()
        
        if len(myresult) > 0:
            # Extract the list of shotCodes from the result
            shot_codes = [int(item[0]) for item in myresult]
        else:
            shot_codes = []

    except Exception as e:
        print('Error:', e)
        shot_codes = []

    return shot_codes