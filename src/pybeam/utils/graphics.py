# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:47:49 2016

@author: phil
"""

import warnings
import logging

logger = logging.getLogger(__name__)

from matplotlib import pyplot as plt
import numpy as np
import pybeam.utils.arraymaths as ar

_useCV = True
_usePIL = True

try:
    import cv2
except:
    logger.warning("Cannot import opencv")
    _useCV = False
    #import scipy.misc as scim


class PMBexecption(RuntimeError):
    def __init__(self, arg):
        self.args = arg

try:
    from PIL import Image
except ImportError:
    logger.warning('Could not import PIL')
    _usePIL = False

_origin = 'upper'


def __drawcomplex(img, myExtent, colorbar, bad_phase_colour=(1, 0, 0)):
    ax = [None] * 4
    if img.ndim != 2:
        raise PMBexecption('Wrong number of dimensions for complex display')
    ax[0] = plt.subplot(2, 2, 1)
    plt.imshow(ar.ints(img), interpolation='nearest', origin=_origin, cmap='gray', extent=myExtent)
    if colorbar:
        plt.colorbar()
    plt.title('Intensity')

    ax[1] = plt.subplot(2, 2, 2)
    plt.imshow(np.real(img), interpolation='nearest', origin=_origin, cmap='gray', extent=myExtent)
    if colorbar:
        plt.colorbar()

    plt.title('Real')

    ax[2] = plt.subplot(2, 2, 3)
    plt.imshow(np.imag(img), interpolation='nearest', origin=_origin, cmap='gray', extent=myExtent)
    plt.title('Imaginary')
    if colorbar:
        plt.colorbar()

    ax[3] = plt.subplot(2, 2, 4)

    ints = ar.ints(img)
    indx2 = ints == 0
    ang = np.angle(img)
    if bad_phase_colour:
        colour_ang = np.dstack([ang, ang, ang])
        colour_ang = (colour_ang - colour_ang.min()) / (colour_ang.max() - colour_ang.min())
        colour_ang[indx2, 0] = bad_phase_colour[0]
        colour_ang[indx2, 1] = bad_phase_colour[1]
        colour_ang[indx2, 2] = bad_phase_colour[2]
        plt.imshow(colour_ang, interpolation='nearest', origin=_origin, extent=myExtent)
    else:
        plt.imshow(ang, interpolation='nearest', cmap='gray', origin=_origin, extent=myExtent)
    plt.title('Phase')
    if colorbar:
        plt.colorbar()
    return ax


def drawnow():
    plt.pause(0.05)


def __drawimg(img, myExtent, colorbar):
    if img.ndim == 3:  # color
        plt.imshow(img, interpolation='nearest', origin=_origin, extent=myExtent)
    elif img.ndim == 2:  # grayscale or complex
        plt.imshow(img, interpolation='nearest', origin=_origin, cmap='gray', extent=myExtent)
    else:
        raise PMBexecption("Wrong number of dimensions in input, input has " + str(img.ndim) + " dims")
    ax = plt.gca()
    if colorbar:
        plt.colorbar()
    return ax


def draw(img1, *args, extent=None, colorbar=False, newFigure=False, BGR=False, num=None, title=None,
         bad_phase_colour=(1, 0, 0)):
    """my custom image drawing function

    custom image drawing function to cope with different types such grey scale,
    colour or complex

    Parameters
    ------------
    img : numpy.ndarray
        First image to draw
    param2: numpy.ndarray,or :obj:'str', optional
        either a second image, to display in subplot or the figure suptitle
    param3: :obj:'str', optional
        figure suptitle if pararm2 is an image
    extent=[left,right,top,bottom], optional
        sets the extent of the image,
    BGR = False, optional, bool, if true image is BGR color,othewise RGB
    bad_phase_colour: tuple
        rgb colour tuple of the colour to use if phase is not defined on the complex plots. Set to None to not use."""
    # options={'Extent':None,'colorbar':False,'newFigure':False,'BGR':False}
    # options.update(kwargs)
    myExtent = extent
    # myExtent=options['Extent']
    # colorbar=options['colorbar']
    # newFigure=options['newFigure']
    # BGR=options['BGR']
    if num is None:
        if newFigure == True:
            fig = plt.figure()
        else:
            plt.clf()  # clear current figure
            fig = plt.gcf()
        try:
            fig.canvas.manager.window.raise_()  # bring figure to front
        except:
            # we are probably in an ipython embedded console
            warnings.warn("Are you running draw from embedded console? Try %matplotlib qt")
    else:
        fig = plt.figure(num=num)
        plt.clf()

    if BGR and img1.ndim == 3:  # convert BGR to RGB
        img = img1[:, :, ::-1]
    else:
        img = img1
    ax = [None]
    if np.iscomplexobj(img):
        ax = __drawcomplex(img, myExtent, colorbar, bad_phase_colour=bad_phase_colour)
        if title:
            plt.suptitle(title)
        return fig, ax

    if len(args) == 0:
        # only one input
        ax[0] = __drawimg(img, myExtent, colorbar)
        if title is not None:
            fig.suptitle(title)
        return fig, ax

    if isinstance(args[0], np.ndarray):  # did we pass two images
        if (args[0].ndim > 1):  # test for 2d or colour
            plt.subplot(1, 2, 1)
            ax[0] = __drawimg(img, myExtent, colorbar)
            plt.subplot(1, 2, 2)
            ax.append(__drawimg(args[0], myExtent, colorbar))
        else:
            # draw the only image then
            ax[0] = __drawimg(img, colorbar)
            # it must be axis data
            print('axis data not done yet')
    else:
        ax[0] = __drawimg(img, myExtent, colorbar)

    for var in args:
        if isinstance(var, str):
            plt.suptitle(var)
    if title is not None:
        fig.suptitle(title)

    return fig, ax


def _PIL_grimread(filename, scale=True, cast=np.float64, greyscale=True):
    if not _usePIL:
        raise Exception('PIL not imported')
        return None

    img = Image.open(filename, flatten=greyscale, mode='L')
    imgf = img.astype(cast)
    if imgf.dtype.kind == 'f' and scale == True:
        imgf /= float(np.iinfo(img.dtype).max)

    return imgf


def grimread(filename, scale=True, cast=np.float64, greyscale=True):
    """reads an image from file and greylevel and double

    filename str filename
    scale=True  scales o/p to 0-1 if o/p type is floating
    cast=np.float64 o/p type to cast to"""
    if _useCV == False:
        return _PIL_grimread(filename, scale, cast, greyscale)

    if greyscale:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise PMBexecption("Failed to open file: " + filename)
    else:
        img = cv2.imread(filename)
        if img is None:
            raise PMBexecption("Failed to open file: " + filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:  # bad news
        raise PMBexecption('Cannot load image: ' + filename)
    imgf = img.astype(cast)
    if imgf.dtype.kind == 'f' and scale == True:
        imgf /= float(np.iinfo(img.dtype).max)

    return imgf


def imread(filename, scale=True, greyscale=False):
    """returns a scaled float64 RGB colour image

    Parameters
    -------------
    filename str filename
    scale=True bool if true scale image 255->1
    greyscale=False bool if true return greyscale image
    """
    if scale == False:
        ncast = np.uint8
    else:
        ncast = np.float64
    return grimread(filename, scale=scale, greyscale=greyscale, cast=ncast)


def close_all():
    plt.close("all")


lastkey = 0


def __keypress(event):
    global lastkey
    lk = event.key  # a string
    if len(lk) == 1:
        lastkey = ord(lk)
    elif lk == 'escape':
        lastkey = 27
    else:
        lastkey = 0


def waitkey(n=0):
    global lastkey
    if n == 0:
        plt.pause(0.01)
    else:
        plt.pause(n)
    lk = lastkey
    lastkey = 0
    return lk


def vdraw(h, img1, BGR=True):
    """Video drawing
    """
    plt.ion()
    if BGR and img1.ndim == 3:  # convert BGR to RGB
        img = img1[:, :, ::-1]
    if h is None:
        h = plt.figure()
        h.canvas.mpl_connect('key_press_event', __keypress)
        __drawimg(img, None, False)
    else:
        ax = h.get_axes()[0].get_images()[0]  # there should be only one!
        ax.set_data(img)

    drawnow()
    return h


def imshow(title, img, scale=True, BGR=False):
    """Uses opencv imshow to display RGB images, can be scaled, BGR=True displays BGR format"""
    if _useCV == False:
        raise PMBexecption('No opencv')

    if img.dtype == np.uint8:
        dimg = img
    if img.dtype == np.uint16:
        if scale == True:
            dimg = cv2.normalize(img, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        else:
            dimg = img
        # dimg=img.view('uint8')[:,::2]  #view and skip MSB
    if img.dtype == np.float:
        if scale == True:
            dimg = cv2.normalize(img, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        else:
            dimg = img
    if img.dtype == np.bool:
        dimg = img.astype(np.uint8) * 255

    if BGR and img.ndim == 3:  # convert BGR to RGB
        dimg = dimg[:, :, ::-1]

    cv2.imshow(title, dimg)


def figure2opencv(figure):
    figure.canvas.draw()
    img = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


if __name__ == "__main__":
    import os

    print(os.getcwd())
    img = imread('../images/Lenna.png', scale=True)
    print(img)
    fig, ax = draw(img, newFigure=True)
    import matplotlib

    x1 = [0, .001]
    y1 = [0, .001]
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    line1, = ax.plot(x1, y1, 'r-')
    if _useCV:
        img = figure2opencv(fig)
        cv2.imshow("plot", img)
        cv2.waitKey()
        cv2.destroyAllWindows()
#    import PMB.common.FastVideo as fvr
#    h=None
#
#    with fvr.FVR("/home/phil/Dropbox/Matlab/Videos/PVtest.mp4") as fv:
#        for frame in fv.read():
#             sframe=frame.astype(np.uint16)*40
#             h=imshow("test",sframe,scale=False)
#             res=cv2.waitKey(10)
#             if res==27:
#                 break
#         #print(res)
