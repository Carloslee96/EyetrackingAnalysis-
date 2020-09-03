import pandas as pd
import numpy as np
import glob,os
import csv

# Select point (x,y) satisfying
# group 2: 713>x>625 & 378>y>295
# person 3: 680>x>614 & 280>y>178
# person 13: 602>x>455 & 440>y>280
# wakeboard 10: 705>x>485 & 361>y>214
# car 10: 695>x>541 & 531>y>326

path = r'D:\Pycharm\Project\plot_all_data\output_fixation\car10_data'
file = glob.glob(os.path.join(path, "*.csv"))
input_df = []
# Produce datasets for input
for f1 in file:
    input_df.append(pd.read_csv(f1, usecols=[3,4,5]))

num_input = len(input_df)
counter = np.arange(0,num_input)

# Data type of input_df: list; input_df[1]: dataframe
index = []
for i in counter:
    reader_in = input_df[i]
    df_nan = input_df[i].fillna(0)  # replace NaN by 0
    reader = df_nan.astype(int)  # transfer data type
    # # group2
    # select = reader[(reader['endx'] >= 625) & (reader['endx']<713) &
    #                     (reader['endy'] >= 295) & (reader['endy']<378)]
    # # person3
    # select = reader[(reader['endx'] >= 614) & (reader['endx']<680) &
    #                     (reader['endy'] >= 178) & (reader['endy']<280)]
    # # person13
    # select = reader[(reader['endx'] >= 455) & (reader['endx']<602) &
    #                     (reader['endy'] >= 280) & (reader['endy']<440)]
    # # wakeboard10
    # select = reader[(reader['endx'] >= 485) & (reader['endx']<705) &
    #                     (reader['endy'] >= 214) & (reader['endy']<361)]
    # car10
    select = reader[(reader['endx'] >= 541) & (reader['endx']<695) &
                        (reader['endy'] >= 326) & (reader['endy']<531)]

    #print(select)
    total_time = (reader['etime'].iloc[-1] - reader['etime'].iloc[1]) / 1000
    dwelling_time = (select['etime'].iloc[-1]-select['etime'].iloc[1])/1000
    #print('The Dwelling Time is %f s' % dwelling_time)

    time_raw = reader_in['etime'].iloc[1]
    time_select = select['etime'].iloc[1]
    Reaction = (time_select - time_raw)
    #print('The Reaction time is %f ms' % Reaction)

    row_select = select.shape[0]
    row_raw = reader_in.shape[0]
    hit_num = row_select
    #print('The Number of fixation points in the ROI is %f' % hit_num)

    index_temp = [dwelling_time,Reaction,hit_num]
    index.append(index_temp)

with open('D:\Pycharm\Project\plot_all_data\output_index\ car10_14obs.csv', 'w') as file:
    csvwriter = csv.writer(file, lineterminator='\n')
    csvwriter.writerows(index)
