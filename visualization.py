# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import glob,os
import numpy as np
import math

path = r'D:\Pycharm\Project\plot_all_data\output_index'
file = glob.glob(os.path.join(path, "*.csv"))
input_df = []
for f1 in file:
    input_df.append(pd.read_csv(f1, names=['Dwelling Time', 'Reaction Time','Number of Fixation']))

# 1. Plot bar chart
DT_0_temp =input_df[0]
DT_0 = DT_0_temp['Reaction Time'].iloc[:4] # First 4 column of Dwelling Time in one image
DT_1_temp =input_df[1]
DT_1 = DT_1_temp['Reaction Time'].iloc[:4] # First 4 column of Dwelling Time in one image
DT_2_temp =input_df[2]
DT_2 = DT_2_temp['Reaction Time'].iloc[:4] # First 4 column of Dwelling Time in one image
DT_3_temp =input_df[3]
DT_3 = DT_3_temp['Reaction Time'].iloc[:4] # First 4 column of Dwelling Time in one image
DT_4_temp =input_df[4]
DT_4 = DT_4_temp['Reaction Time'].iloc[:4] # First 4 column of Dwelling Time in one image


name_list = ['Car10', 'Group2', 'Person3', 'Person13','Wakeboard10']

list0 = [DT_0[0],DT_1[0],DT_2[0],DT_3[0],DT_4[0]]
list1 = [DT_0[1],DT_1[1],DT_2[1],DT_3[1],DT_4[1]]
list2 = [DT_0[2],DT_1[2],DT_2[2],DT_3[2],DT_4[2]]
list3 = [DT_0[3],DT_1[3],DT_2[3],DT_3[3],DT_4[3]]

x = list(range(len(list0)))
total_width, n = 0.8, 4
width = total_width / n

plt.bar(x, list0, width=width, label='Observer1',  fc='y')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, list1, width=width, label='Observer2',tick_label=name_list, fc='r')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, list2, width=width, label='Observer3', fc='b')
for i in range(len(x)):
    x[i] = x[i] + width
plt.bar(x, list3, width=width, label='Observer4', fc='g')
plt.title('Reaction TIme (ms)')
plt.legend()
plt.show()


# 2. Boxplot
NF_0_temp =input_df[0]
NF_0 = NF_0_temp['Number of Fixation'].iloc[:5]
NF_1_temp =input_df[1]
NF_1 = NF_1_temp['Number of Fixation'].iloc[:5]
NF_2_temp =input_df[2]
NF_2 = NF_2_temp['Number of Fixation'].iloc[:5]
NF_3_temp =input_df[3]
NF_3 = NF_3_temp['Number of Fixation'].iloc[:5]
NF_4_temp =input_df[4]
NF_4 = NF_4_temp['Number of Fixation'].iloc[:5]
#print(NF_4)

box_0 = [NF_0[0],NF_1[0],NF_2[0],NF_3[0],NF_4[0]]
box_1 = [NF_0[1],NF_1[1],NF_2[1],NF_3[1],NF_4[1]]
box_2 = [NF_0[2],NF_1[2],NF_2[2],NF_3[2],NF_4[2]]
box_3 = [NF_0[3],NF_1[3],NF_2[3],NF_3[3],NF_4[3]]
#print(box_0)

plt.figure(figsize=(10, 5))  # figure size
plt.title('Boxplot of Number of Fixation Points in AOI', fontsize=20)  # title
labels = 'Observer1', 'Observer2', 'Observer3', 'Observer4'  # Label
plt.boxplot([box_0, box_1, box_2, box_3], labels=labels)  # grid=False
plt.show()




## 3. Snare in 'car' and 'person13'

DT_car10_temp =input_df[0] # select car10.csv from input_df
DT_car10 = DT_car10_temp['Number of Fixation'].iloc[:4] # First 4 column of Dwelling Time in one image
list_num_AOI = DT_car10.values.tolist()

DT_person13_temp =input_df[0] # select car10.csv from input_df
DT_person13 = DT_person13_temp['Number of Fixation'].iloc[:4] # First 4 column of Dwelling Time in one image
list_num_AOI = DT_person13.values.tolist()

#Extract fixation points in snare part
import pandas as pd
import numpy as np
import glob,os
import csv

path = r'D:\Pycharm\Project\plot_all_data\output_fixation\car10_data'
# path = r'D:\Pycharm\Project\plot_all_data\output_fixation\person13_data'
file = glob.glob(os.path.join(path, "*.csv"))
input_df = []
# Produce datasets for input
for f1 in file:
    input_df.append(pd.read_csv(f1, usecols=[2,4,5]))

num_input = len(input_df)
counter = np.arange(0,num_input)

# Data type of input_df: list; input_df[1]: dataframe
index = []
index_snare = []
for i in counter:
    reader_in = input_df[i]
    df_nan = input_df[i].fillna(0)  # replace NaN by 0
    reader = df_nan.astype(int)  # transfer data type
    # car10: snare x range (860,987), y range (408,496)
    #        AOI   x range (541,695), y range (326,541)
    # person13: snare x range (720,836), y range (181,320)
    #          AOI   x range (455,602), y range (280,440)


    # person13
    # select = reader[(reader['endx'] >= 720) & (reader['endx']<836) &
    #                 (reader['endy'] >= 181) & (reader['endy']<320)]
    # car10
    select = reader[(reader['endx'] >= 860) & (reader['endx']<987) &
                    (reader['endy'] >= 408) & (reader['endy']<496)]
    row_select = select.shape[0]
    row_raw = reader_in.shape[0]
    hit_num = row_select
    #print('The Number of fixation points in the AOI is %f' % hit_num)

    # for plot snare
    index_snare_temp = hit_num
    index_snare.append(index_snare_temp)
    # for writting into csv
    index_temp = [hit_num]
    index.append(index_temp)

with open('D:\Pycharm\Project\plot_all_data\output_snare\ car10_snare.csv', 'w') as file:
#with open('D:\Pycharm\Project\plot_all_data\output_snare\ person13_snare.csv', 'w') as file:
    csvwriter = csv.writer(file, lineterminator='\n')
    csvwriter.writerows(index)


list_num_snare = index_snare[:4]
print(list_num_AOI)
print(list_num_snare)

name = ['Observer1','Observer2','Observer3','Observer4']

plt.bar(range(len(list_num_AOI)), list_num_AOI, label='Number of fixation in AOI',fc = 'y')
plt.bar(range(len(list_num_AOI)), list_num_snare, bottom=list_num_AOI, label='Number of fixation in snare',tick_label = name,fc = 'r')
#plt.title('Number of fixation in AOI and Snare in image person13')
plt.title('Number of fixation in AOI and Snare in image car10')
plt.legend()
plt.show()



# 4. Euclidean Distance
#path = r'D:\Pycharm\Project\plot_all_data\output_fixation\car10_data'
#path = r'D:\Pycharm\Project\plot_all_data\output_fixation\group2_data'
#path = r'D:\Pycharm\Project\plot_all_data\output_fixation\person3_data'
#path = r'D:\Pycharm\Project\plot_all_data\output_fixation\person13_data'
path = r'D:\Pycharm\Project\plot_all_data\output_fixation\wakeboard10_data'

file = glob.glob(os.path.join(path, "*.csv"))
input_df = []
# Produce datasets for input
for f1 in file:
    input_df.append(pd.read_csv(f1, usecols=[2,4,5]))
# input_df[0-13] represents all fixation data
reader_in = input_df[0]
xa = reader_in['endx']
ya = reader_in['endy']
num_input = len(xa)
counter = np.arange(1,num_input)

Eucd = []
for i in counter:
    p1 = (xa[i],ya[i])
    # p2 = (618,452) # car10
    # p2 = (368,521) # person13
    # p2 = (669,337) # group2
    # p2 = (647,229) # person3
    p2 = (595,288) # wakeboard10

    Eucd_temp = math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)]))
    Eucd_temp = int(Eucd_temp)
    Eucd.append(Eucd_temp)

print(Eucd)

plt.plot(counter,Eucd,'o-',color = 'g',label="ob2")
plt.xlabel("Time")
plt.ylabel("Eucidean Distance")
plt.legend("wakeboard10")
plt.show()

