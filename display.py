import PMB.imutils.graphics as gr
import abc
import matplotlib.pyplot as plt
import numpy as np
import logging
logger = logging.getLogger(__name__)
#class display(abc.ABC):  # TODO is this needed?
#    def __init__(self):
#        pass

def displayInt(beam, cross_section=False, title=None, nthroot=1, colorbar=True, **kwargs):
    """
    Plots the beam intensity
    :param beam: complex beam
    :param cross_section: if true, show the intensity as an image, a cross section 2D plot through the centre + phase
    :param title: title
    :param nthroot: normally 1, otherwise use intensity^(1/nthroot)
    :param colorbar: show colour bar
    :param kwargs: kwargs passed to plt.figure
    :return: (fig, ax) handles, ax is list if cross_section=True
    """
    fig = plt.figure(**kwargs)
    fig.clf()

    if nthroot != 1:
        ints = np.power(beam.intensity, 1 / nthroot)
    else:
        ints = beam.intensity
    if not cross_section:
        # ax1=fig.add_subplot(1,1,1)

        fig, ax = gr.draw(ints, extent=[-beam.width / 2, beam.width / 2, -beam.width / 2, beam.width / 2],
                          title=title, colorbar=colorbar)
    else:
        ax = [None] * 3
        ax[0] = fig.add_subplot(2, 2, 1)
        ax[0].imshow(ints, extent=[-beam.width / 2, beam.width / 2, -beam.width / 2, beam.width / 2])
        ax[1] = fig.add_subplot(2, 2, 2)
        bi = beam.intensity
        r = bi.shape[0] // 2
        ax[1].plot(np.linspace(-beam.width / 2, beam.width / 2, bi.shape[0]), bi[r, :])
        ax[1].set_title('Int')
        ax[2] = fig.add_subplot(2, 2, 4)
        ax[2].plot(np.linspace(-beam.width / 2, beam.width / 2, bi.shape[0]), np.unwrap(beam.phase[r, :]))
        ax[2].set_title('Phase')
        fig.suptitle(title)
    return fig, ax
    # ax1.set_xticklabels(np.linspace(-beam.width/2,beam.width/2,5))

def displayInt3D(beam, title=None, nthroot=1, colorbar=True, **kwargs):
    fig = plt.figure(**kwargs)
    fig.clf()

    if nthroot != 1:
        ints = np.power(beam.intensity, 1 / nthroot)
    else:
        ints = beam.intensity

    # Make data.
    x = np.linspace(-beam.width / 2, beam.width / 2, beam.num)
    #Y = np.arange(-beam.width / 2, beam.width / 2, 0.25)
    X, Y = np.meshgrid(x,x)
    #fig, ax = gr.draw(ints, Extent=[-beam.width / 2, beam.width / 2, -beam.width / 2, beam.width / 2],
    #                      title=title, colorbar=colorbar)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,beam.intensity)
    return fig, ax



def display(beam, title=None, num=None, **kwargs):
    return gr.draw(beam.field, num=num, extent=[-beam.width / 2, beam.width / 2, -beam.width / 2,
                                                beam.width / 2])  # , title=title,**kwargs)


