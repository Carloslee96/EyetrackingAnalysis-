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

# Run function
background_image = 'D:\Pycharm\Project\plot_all_data\input_img\wakeboard10_000356.jpg'
dispsize = (1280,720)
display_width = 1280
display_height = 720
alpha = 0.5
sd = None
ngaussian = 200

# Counter number of files in input
path ='D:\Pycharm\Project\plot_all_data\input_data\wakeboard10'
num_file = 0
for file in os.listdir(path):
    num_file = num_file+1
counter = numpy.arange(1,num_file+1)
print(counter)
for i in counter:
    input_path = os.path.join('D:\Pycharm\Project\plot_all_data\input_data\wakeboard10','wakeboard10_o'+str(i)+'_720p.csv')
    output_name = os.path.join('D:\Pycharm\Project\plot_all_data\output_heatmap\wakeboard10','0000'+str(i)+'.jpg')
    with open(input_path) as f:
        # debug Processing initial dataset
        reader_in = pd.read_csv(f, index_col=False, usecols=[0,5,6])
        df_nan = reader_in.fillna(0)  # replace NaN by 0

        reader = df_nan.astype(int)  # transfer data type
        reader_2col = reader.iloc[:, 1:] # get 2-column fixations without time track
        raw_2c = reader_2col.values.tolist()  # 2-col fixations without time track

        # 1. Plot point map and heat map
        gaza_data_2 = []
        if len(raw_2c[0]) is 2:
            gaze_data_2 = list(map(lambda q: (int(q[0]), int(q[1]), 1), raw_2c))
        else:
            gaze_data_2 = list(map(lambda q: (int(q[0]), int(q[1]), int(q[2])), raw_2c))
        # print(raw_2c)
        # print(gaze_data_2)
        draw_heatmap(gaze_data_2, (display_width, display_height), alpha=alpha, savefilename=output_name,
                     imagefile=background_image, gaussianwh=ngaussian, gaussiansd=sd)