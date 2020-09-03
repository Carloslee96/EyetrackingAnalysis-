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

def gaussian(x, sx, y=None, sy=None):
    """Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution

    arguments
    x		-- width in pixels
    sx		-- width standard deviation

    keyword argments
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = x / 2
    yo = y / 2
    # matrix of zeros
    M = numpy.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = numpy.exp(
                -1.0 * (((float(i) - xo) ** 2 / (2 * sx * sx)) + ((float(j) - yo) ** 2 / (2 * sy * sy))))

    return M

# Define function for heat map
def draw_heatmap(gazepoints, dispsize, imagefile=None, alpha=0.5, savefilename=None, gaussianwh=200, gaussiansd=None):
    """Draws a heatmap of the provided fixations, optionally drawn over an
    image, and optionally allocating more weight to fixations with a higher
    duration.

    arguments

    gazepoints		-	a list of gazepoint tuples (x, y)
    
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

    # IMAGE
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # HEATMAP
    # Gaussian
    gwh = gaussianwh
    gsdwh = gwh / 6 if (gaussiansd is None) else gaussiansd
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = int(gwh / 2)
    heatmapsize = int(dispsize[1] + 2 * strt), int(dispsize[0] + 2 * strt)

    heatmap = numpy.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(gazepoints)):
        # get x and y coordinates
        x = int(strt + gazepoints[i][0]) - int(gwh / 2)
        y = int(strt + gazepoints[i][1]) - int(gwh / 2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh];
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x - dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y - dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * gazepoints[i][2]
            except:
                # fixation was probably outside of display
                pass

        else:
            # add Gaussian to the current heatmap

            heatmap[y:y + gwh, x:x + gwh] += gaus * gazepoints[i][2]
    # resize heatmap

    heatmap = heatmap[strt:dispsize[1] + strt, strt:dispsize[0] + strt]
    # remove zeros
    lowbound = numpy.mean(heatmap[heatmap > 0])
    heatmap[heatmap < lowbound] = numpy.NaN
    # draw heatmap on top of image
    ax.imshow(heatmap, cmap='jet', alpha=alpha)

    # FINISH PLOT
    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()
    # save the figure if a file name was provided
    if savefilename != None:
        fig.savefig(savefilename)

    return fig

# Define function for raw point map
def draw_raw(gazepoints, dispsize, imagefile=None, savefilename=None):
    """Draws the raw x and y data

    arguments

    x			-	a list of x coordinates of all samples that are to
                    be plotted
    y			-	a list of y coordinates of all samples that are to
                    be plotted
    dispsize		-	tuple or list indicating the size of the display,
                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                    is to be laid, or None for no image; NOTE: the image
                    may be smaller than the display size, the function
                    assumes that the image was presented at the centre of
                    the display (default = None)
    savefilename	-	full path to the file in which the heatmap should be
                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                    fixations
    """

    # image
    fig, ax = draw_display(dispsize, imagefile=imagefile)

    # x, y
    for i in range(0, len(gazepoints)):
        # get x and y coordinates
        x = int(gazepoints[i][0])
        y = int(gazepoints[i][1])
    # plot raw data in white points
        ax.plot(x, y, 'o', color='#FFFFFF')

    # invert the y axis, as (0,0) is top left on a display
    ax.invert_yaxis()


    # save the figure if a file name was provided
    if savefilename != None:
            fig.savefig(savefilename)

    return fig

# def draw_fixations(fixations, dispsize, imagefile=None, durationsize=True, durationcolour=True, alpha=0.5,
#                    savefilename=None):
#     """Draws circles on the fixation locations, optionally on top of an image,
#     with optional weigthing of the duration for circle size and colour
#
#     arguments
#
#     fixations		-	a list of fixation ending events from a single trial,
#                     as produced by edfreader.read_edf, e.g.
#                     edfdata[trialnr]['events']['Efix']
#     dispsize		-	tuple or list indicating the size of the display,
#                     e.g. (1024,768)
#
#     keyword arguments
#
#     imagefile		-	full path to an image file over which the heatmap
#                     is to be laid, or None for no image; NOTE: the image
#                     may be smaller than the display size, the function
#                     assumes that the image was presented at the centre of
#                     the display (default = None)
#     durationsize	-	Boolean indicating whether the fixation duration is
#                     to be taken into account as a weight for the circle
#                     size; longer duration = bigger (default = True)
#     durationcolour	-	Boolean indicating whether the fixation duration is
#                     to be taken into account as a weight for the circle
#                     colour; longer duration = hotter (default = True)
#     alpha		-	float between 0 and 1, indicating the transparancy of
#                     the heatmap, where 0 is completely transparant and 1
#                     is completely untransparant (default = 0.5)
#     savefilename	-	full path to the file in which the heatmap should be
#                     saved, or None to not save the file (default = None)
#
#     returns
#
#     fig			-	a matplotlib.pyplot Figure instance, containing the
#                     fixations
#     """
#
#     # FIXATIONS
#     fix = parse_fixations(fixations)
#
#     # IMAGE
#     fig, ax = draw_display(dispsize, imagefile=imagefile)
#
#     # CIRCLES
#     # duration weigths
#     if durationsize:
#         siz = 1 * (fix['dur'] / 15000.0)
#     else:
#         siz = 1 * numpy.median(fix['dur'] / 15000.0)
#     if durationcolour:
#         col = fix['dur']
#     else:
#         col = COLS['chameleon'][2]
#     # draw circles
#     ax.scatter(fix['x'], fix['y'], s=siz, c=col, marker='o', cmap='jet', alpha=alpha, edgecolors='#000000')
#
#     # FINISH PLOT
#     # invert the y axis, as (0,0) is top left on a display
#     ax.invert_yaxis()
#     # save the figure if a file name was provided
#     if savefilename != None:
#         fig.savefig(savefilename)
#
#     return fig
#
# def draw_scanpath(gaze_data, dispsize, imagefile=None, alpha=0.5, savefilename=None):
#     """Draws a scanpath: a series of arrows between numbered fixations,
#     optionally drawn over an image
#
#     arguments
#
#     fixations		-	a list of fixation ending events from a single trial,
#                     as produced by edfreader.read_edf, e.g.
#                     edfdata[trialnr]['events']['Efix']
#     saccades		-	a list of saccade ending events from a single trial,
#                     as produced by edfreader.read_edf, e.g.
#                     edfdata[trialnr]['events']['Esac']
#     dispsize		-	tuple or list indicating the size of the display,
#                     e.g. (1024,768)
#
#     keyword arguments
#
#     imagefile		-	full path to an image file over which the heatmap
#                     is to be laid, or None for no image; NOTE: the image
#                     may be smaller than the display size, the function
#                     assumes that the image was presented at the centre of
#                     the display (default = None)
#     alpha		-	float between 0 and 1, indicating the transparancy of
#                     the heatmap, where 0 is completely transparant and 1
#                     is completely untransparant (default = 0.5)
#     savefilename	-	full path to the file in which the heatmap should be
#                     saved, or None to not save the file (default = None)
#
#     returns
#
#     fig			-	a matplotlib.pyplot Figure instance, containing the
#                     heatmap
#     """
#
#     # image
#     fig, ax = draw_display(dispsize, imagefile=imagefile)
#
#     # FIXATIONS
#     # parse fixations
#     fix = parse_fixations(gaze_data)
#     # draw fixations
#     ax.scatter(fix['x'], fix['y'], s=500, color='#FFA500', marker='o', cmap='jet',
#                alpha=1, edgecolors='none')
#     # draw annotations (fixation numbers)
#     for i in range(len(gaze_data)):
#             ax.annotate(str(i + 1), (fix['x'][i], fix['y'][i]), size=15, color='#000000', alpha=1,
#                     horizontalalignment='center', verticalalignment='center', multialignment='center')
#             for j in range(len(gaze_data)-1):
#                 ax.arrow(fix['x'][j], fix['y'][j], fix['x'][j+1]-fix['x'][j], fix['y'][j+1] - fix['y'][j], alpha=0.3, fc='#FFFFFF', ec='#000000', fill=True,
#                             shape='full', width=5, head_width=10, head_starts_at_zero=False, overhang=0)
#
#     # invert the y axis, as (0,0) is top left on a display
#     ax.invert_yaxis()
#     # save the figure if a file name was provided
#     if savefilename != None:
#         fig.savefig(savefilename)
#
#     return fig
#


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
        #stime, etime,
        dur, ex, ey = gaza_data[fixnr]
        fix['x'][fixnr] = ex
        fix['y'][fixnr] = ey
        fix['dur'][fixnr] = dur
    #print(fix)
    return fix

# Test 2. Input dataset of EyeTrackUAV_Version_1_imgWITHdataset

# input_path = 'bike3_o1_720p.csv'
# heatmap_name = 'output_bike_heatmap.jpg'
# pointmap_name = 'output_bike_pointmap.jpg'
# fixation_name = 'output_bike_fixation.jpg'
# scanpath_name = 'output_bike_scanpath.jpg'
# background_image = 'bike000001.jpg'

input_path = 'D:\Pycharm\Project\plot_all_data\input_data\wakeboard10\wakeboard10_o1_720p.csv'
heatmap_name = 'output_boat_heatmap.jpg'
pointmap_name = 'output_boat_pointmap.jpg'
# fixation_name = 'output_boat_fixation.jpg'
# scanpath_name = 'output_boat_scanpath.jpg'
background_image = 'D:\Pycharm\Project\plot_all_data\input_img\wakeboard10_000356.jpg'

# input_path = 'car6_o1_720p.csv'
# heatmap_name = 'output_car_heatmap.jpg'
# pointmap_name = 'output_car_pointmap.jpg'
# fixation_name = 'output_car_fixation.jpg'
# scanpath_name = 'output_car_scanpath.jpg'
# background_image = 'car000001.jpg'

display_width = 1280
display_height = 720
alpha = 0.5

ngaussian = 200
sd = None
line_count = sum(1 for i in open(input_path))

with open(input_path) as f:
    # debug Processing initial dataset
    reader_in = pd.read_csv(f, index_col=False, usecols=[0,5,6])
    df_nan = reader_in.fillna(0)  # replace NaN by 0

    reader = df_nan.astype(int)  # transfer data type
    reader_2col = reader.iloc[:, 1:] # get 2-column fixations without time track
    raw_2c = reader_2col.values.tolist()  # 2-col fixations without time track

    # Select data every 1000 rows
    reader_scanpath = reader.iloc[::1000, :]
    reader_fixation = reader.iloc[::1, :] # bike, boat
    # reader_fixation = reader.iloc[::200, :] # car

    raw_3c = reader.values.tolist() # 3-col fixations with time track
    raw_3c_scanpath = reader_scanpath.values.tolist() # 3-column scanpath data with time track
    raw_3c_fixation = reader_fixation.values.tolist()  # 3-column scanpath data with time track

    # 1. Plot point map and heat map
    gaza_data_2 = []
    if len(raw_2c[0]) is 2:
        gaze_data_2 = list(map(lambda q: (int(q[0]), int(q[1]), 1), raw_2c))
    else:
        gaze_data_2 = list(map(lambda q: (int(q[0]), int(q[1]), int(q[2])), raw_2c))
    # print(raw_2c)
    # print(gaze_data_2)
    draw_heatmap(gaze_data_2, (display_width, display_height), alpha=alpha, savefilename=heatmap_name,
                 imagefile=background_image, gaussianwh=ngaussian, gaussiansd=sd)
    draw_raw(gaze_data_2, (display_width, display_height), imagefile=background_image, savefilename=pointmap_name)

    # # 2. Plot fixation map
    # gaza_data_3 = []
    # if len(raw_3c_fixation[0]) is 2:
    #     gaze_data_3 = list(map(lambda q: (int(q[0]), int(q[1]), 1), raw_3c_fixation))
    # else:
    #     gaze_data_3 = list(map(lambda q: (int(q[0]), int(q[1]), int(q[2])), raw_3c_fixation))
    # # print(raw_3c)
    # # print(gaze_data_3)
    # draw_fixations(gaze_data_3, (display_width, display_height), imagefile=background_image, durationsize=True, durationcolour=True, alpha=alpha,
    #                savefilename=fixation_name)
    #
    # # 3. Plot scanpath
    # gaza_data_scan = []
    # if len(raw_3c_scanpath[0]) is 2:
    #     gaze_data_scan = list(map(lambda q: (int(q[0]), int(q[1]), 1), raw_3c_scanpath))
    # else:
    #     gaze_data_scan = list(map(lambda q: (int(q[0]), int(q[1]), int(q[2])), raw_3c_scanpath))
    # # print(raw_3c_scanpath)
    # # print(len(gaze_data_scan))
    # draw_scanpath(gaze_data_scan, (display_width, display_height), imagefile=background_image, alpha=alpha, savefilename=scanpath_name)
