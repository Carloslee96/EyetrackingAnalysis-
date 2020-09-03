# EyetrackingAnalysis-
The projec aims to study Eye-tracking Analysis for Performance Assessment in Colonoscopy. The programming of the project is based on Pygaze Open eye-tracking source.

numpy 1.13.1
matplotlib 2.0.2

Object: Raw data of eyetracking points ('input_data\car10\car10_o1_720p')
1. Use 'fixation' to come out fixation points stored in 'output_fixation\ object_data.file'  
2. Use 'plot_heatmap' to come out  heatmap images stored in 'output_heatmap\ object.file'

Object: Fixation points ('output_fixation\car10_data\car10_fix_1')
1. Use 'plot_fixation' to come out  fixation images stored in 'output_fixation\ object.file'
2. Use 'plot_fixation' to come out  scanpath images stored in 'output_scanpath\ object.file'
3. Use 'index' to come out 'Dwelling Time' and 'Reaction Time' and  'Number of fixation points in the ROI' stored in  'output_index\ car10_14obs'

Raw data contains 46833 eye tracking points  
Fixation data contains 100-500 fixation points

[Hit]
Once changing the 'object' name and  'backgroumd image' name, it would come out all output images. 
'plot_heatmap'
'plot_fixation'
'plot_scanpth': select 1 out of 5 Fixation points
