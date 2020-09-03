import os
import argparse
import csv
import numpy
import matplotlib
from matplotlib import pyplot, image
import pandas as pd


def draw_display(dispsize, imagefile=None):
    """Returns a matplotlib.pyplot Figure and its axes, with a size of
    dispsize, a black background colour, and optionally with an image drawn
    onto it

    arguments

    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)

    returns
    fig, ax		-	matplotlib.pyplot Figure and its axes: field of zeros
                    with a size of dispsize, and an image drawn onto it
                    if an imagefile was passed
    """

    # construct screen (black background)
    screen = numpy.zeros((dispsize[1], dispsize[0], 3), dtype='float32')

    # if an image location has been passed, draw the image
    if imagefile != None:
        # check if the path to the image exists
        if not os.path.isfile(imagefile):
            raise Exception("ERROR in draw_display: imagefile not found at '%s'" % imagefile)
        # load image
        img = image.imread(imagefile)

        # width and height of the image
        w, h = len(img[0]), len(img)
        # x and y position of the image on the display
        x = int(dispsize[0] / 2 - w / 2)
        y = int(dispsize[1] / 2 - h / 2)

        # draw the image on the screen
        screen[y:y + h, x:x + w, :] += img
    # dots per inch
    dpi = 100.0
    # determine the figure size in inches
    figsize = (dispsize[0] / dpi, dispsize[1] / dpi)
    # create a figure
    fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)
    ax = pyplot.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plot display
    ax.axis([0, dispsize[0], 0, dispsize[1]])
    ax.imshow(screen.astype('uint8'))  # , origin='upper')
    # debug screen.astype('uint8') coz matplotlib expects it to range from 0 to

    return fig, ax


def draw_scanpath(gaze_data, dispsize, imagefile=None, alpha=0.5, savefilename=None):
    """Draws a scanpath: a series of arrows between numbered fixations,
    optionally drawn over an image

    arguments

    fixations		-	a list of fixation ending events from a single trial,
                    as produced by edfreader.read_edf, e.g.
                    edfdata[trialnr]['events']['Efix']
    saccades		-	a list of saccade ending events from a single trial,
                    as produced by edfreader.read_edf, e.g.
                    edfdata[trialnr]['events']['Esac']
    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    alpha		-	float between 0 and 1, indicating the transparancy of
                    the heatmap, where 0 is completely transparant and 1
                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                    heatmap
    """

    # image
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # FIXATIONS
    # parse fixations
    fix = parse_fixations(gaze_data)
    # draw fixations
    ax.scatter(fix['x'], fix['y'], s=500, color='#FFA500', marker='o', cmap='jet',
               alpha=1, edgecolors='none')
    # draw annotations (fixation numbers)
    for i in range(len(gaze_data)):
        ax.annotate(str(i + 1), (fix['x'][i], fix['y'][i]), size=15, color='#000000', alpha=1,
                    horizontalalignment='center', verticalalignment='center', multialignment='center')
        for j in range(len(gaze_data) - 1):
            ax.arrow(fix['x'][j], fix['y'][j], fix['x'][j + 1] - fix['x'][j], fix['y'][j + 1] - fix['y'][j], alpha=0.3,
                     fc='#FFFFFF', ec='#000000', fill=True,
                     shape='full', width=5, head_width=10, head_starts_at_zero=False, overhang=0)

    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)

    return fig


def parse_fixations(gaza_data):
    """Returns all relevant data from a list of fixation ending events

    arguments

    fixations		-	a list of fixation ending events from a single trial,
                    as produced by edfreader.read_edf, e.g.
                    edfdata[trialnr]['events']['Efix']

    returns

    fix		-	a dict with three keys: 'x', 'y', and 'dur' (each contain
                a numpy array) for the x and y coordinates and duration of
                each fixation
    """

    # empty arrays to contain fixation coordinates
    fix = {'dur': numpy.zeros(len(gaza_data)),
           'x': numpy.zeros(len(gaza_data)),
           'y': numpy.zeros(len(gaza_data))}
    # get all fixation coordinates

    for fixnr in range(len(gaza_data)):
        # stime, etime,
        dur, ex, ey = gaza_data[fixnr]
        fix['x'][fixnr] = ex
        fix['y'][fixnr] = ey
        fix['dur'][fixnr] = dur
    # print(fix)
    return fix


# Run function
background_image = 'D:\Pycharm\Project\plot_all_data\input_img\car10_000734.jpg'

dispsize = (1280, 720)
display_width = 1280
display_height = 720
alpha = 0.5
sd = None

# Counter number of files in input
path = 'D:\Pycharm\Project\plot_all_data\output_fixation\car10_data'
num_file = 0
for file in os.listdir(path):
    num_file = num_file+1
counter = numpy.arange(1, num_file+1)
print(counter)

for i in counter:
    input_path = os.path.join('D:\Pycharm\Project\plot_all_data\output_fixation\car10_data',
                              'car10_fix_' + str(i) + '.csv')
    output_name = os.path.join('D:\Pycharm\Project\plot_all_data\output_scanpath\car10', '0000' + str(i) + '.jpg')
    with open(input_path) as f:
        # debug Processing initial dataset

        # Select rows every 100 rows
        reader_in = pd.read_csv(f, index_col=False, usecols=[2, 4, 5])
        reader = reader_in.astype(int)  # transfer data type

        reader = reader[reader['endx'] != 0]  # delete original eye tracking point at (0,0)

        reader_scanpath = reader.iloc[::5, :]
        raw = reader_scanpath.values.tolist()

        gaza_data = []

        if len(raw[0]) is 2:
            gaze_data = list(map(lambda q: (int(q[0]), int(q[1]), 1), raw))
        else:
            gaze_data = list(map(lambda q: (int(q[0]), int(q[1]), int(q[2])), raw))

        # print(reader)

        draw_scanpath(gaze_data, dispsize, imagefile=background_image, alpha=alpha, savefilename=output_name)
