import pandas as pd
import numpy as np
import glob,os
import csv


def fixation_detection(x, y, time, maxdist=25, mindur=50):
    """Detects fixations, defined as consecutive samples with an inter-sample
	distance of less than a set amount of pixels (disregarding missing data)

	arguments

	x		-	numpy array of x positions
	y		-	numpy array of y positions
	time		-	numpy array of EyeTribe timestamps

	keyword arguments

	missing	-	value to be used for missing data (default = 0.0)
	maxdist	-	maximal inter sample distance in pixels (default = 25)
	mindur	-	minimal duration of a fixation in milliseconds; detected
				fixation cadidates will be disregarded if they are below
				this duration (default = 100)

	returns
	Sfix, Efix
				Sfix	-	list of lists, each containing [starttime]
				Efix	-	list of lists, each containing [starttime, endtime, duration, endx, endy]
	"""


    # empty list to contain data
    Sfix = []
    Efix = []

    # loop through all coordinates
    si = 0
    fixstart = False
    for i in range(1, len(x)):
        # calculate Euclidean distance from the current fixation coordinate
        # to the next coordinate
        squared_distance = ((x[si] - x[i]) ** 2 + (y[si] - y[i]) ** 2)
        dist = 0.0
        if squared_distance > 0:
            dist = squared_distance ** 0.5
        # check if the next coordinate is below maximal distance
        if dist <= maxdist and not fixstart:
            # start a new fixation
            si = 0 + i
            fixstart = True
            Sfix.append([time[i]])
        elif dist > maxdist and fixstart:
            # end the current fixation
            fixstart = False
            # only store the fixation if the duration is ok
            if time[i - 1] - Sfix[-1][0] >= mindur:
                Efix.append([Sfix[-1][0], time[i - 1], time[i - 1] - Sfix[-1][0], x[si], y[si]])
            # delete the last fixation start if it was too short
            else:
                Sfix.pop(-1)
            si = 0 + i
        elif not fixstart:
            si += 1
    # add last fixation end (we can lose it if dist > maxdist is false for the last point)
    if len(Sfix) > len(Efix):
        Efix.append([Sfix[-1][0], time[len(x) - 1], time[len(x) - 1] - Sfix[-1][0], x[si], y[si]])
    return Sfix, Efix

path = r'D:\Pycharm\Project\plot_all_data\input_data\wakeboard10'
path_out = r'D:\Pycharm\Project\plot_all_data\output_fixation\wakeboard10_data'
file = glob.glob(os.path.join(path, "*.csv"))
input_df = []
# Produce datasets for input
for f1 in file:
    input_df.append(pd.read_csv(f1, usecols=[0,5,6], names=['time', 'x','y']))

num_input = len(input_df)
counter = np.arange(1,num_input+1)
#print(counter)

# Data type of input_df: (list); input_df[1]: raw data from observer 1 (dataframe)
for i in counter:
    reader_in = input_df[i-1] # input_df start from input_df[0], thus need to "-1" 
    df_nan = reader_in.fillna(0)  # replace NaN by 0
    reader = df_nan.astype(int)  # transfer data type
    x = reader['x']
    y = reader['y']
    time = reader['time']
    Sfix, Efix = fixation_detection(x, y, time, maxdist=25, mindur=50)
    name = ['stime', 'etime', 'dur','endx','endy']
    Efix_df = pd.DataFrame(columns=name, data=Efix)
    # print(type(Efix_df))
    # print(Efix_df)
    Efix_df.to_csv(os.path.join(path_out, 'wakeboard10_fix_'+ str(i)+'.csv'), encoding='gbk')


