
import pybeam.utils.graphics as gr
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from .element import Element

def displayInt(beam, cross_section=False, title=None, nthroot=1, colorbar=True, **kwargs):
    """
    Plots the Beam intensity
    :param beam: complex Beam
    :param cross_section: if true, show the intensity as an image, a cross section 2D plot through the centre + _phase
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
        ax[2].plot(np.linspace(-beam.width / 2, beam.width / 2, bi.shape[0]), np.unwrap(beam._phase[r, :]))
        ax[2].set_title('Phase')
        fig.suptitle(title)
    return fig, ax
    # ax1.set_xticklabels(np.linspace(-Beam.width/2,Beam.width/2,5))

def displayInt3D(beam, title=None, nthroot=1, colorbar=True, **kwargs):
    """
    Surface plot of a beams intensity
    :param beam:
    :param title:
    :param nthroot:
    :param colorbar:
    :param kwargs:
    :return:
    """
    fig = plt.figure(**kwargs)
    fig.clf()

    if nthroot != 1:
        ints = np.power(beam.intensity, 1 / nthroot)
    else:
        ints = beam.intensity

    # Make data.
    x = np.linspace(-beam.width / 2, beam.width / 2, beam.num)
    #Y = np.arange(-Beam.width / 2, Beam.width / 2, 0.25)
    X, Y = np.meshgrid(x,x)
    #fig, ax = gr.draw(ints, Extent=[-Beam.width / 2, Beam.width / 2, -Beam.width / 2, Beam.width / 2],
    #                      title=title, colorbar=colorbar)
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,beam.intensity)
    return fig, ax



def display(beam, title=None, num=None, **kwargs):
    """
    Diplays _amplitude, real, imaginary and _phase components of a beam, or element or numpy array
    :param beam:
    :param title:
    :param num:
    :param kwargs:
    :return:
    """
    if not num:
        f=plt.figure()
        num=f.number
        logger.debug(f'Created new figure number {num}')

    if isinstance(beam,Element):
        return gr.draw(beam.complex_transmission, title=title,**kwargs)
    elif isinstance(beam, np.ndarray):
        return gr.draw(beam, title=title, **kwargs)
    else:
        return gr.draw(beam.field, num=num, title=title, extent=[-beam.width / 2, beam.width / 2, -beam.width / 2,
                                                beam.width / 2] ,**kwargs)

def drawnow():
    gr.drawnow()


