import numpy as np
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

class Element(ABC):
    def __init__(self, complex_transmission):
        self.complex_transmission = complex_transmission

    def apply(self, beam):
        """
        Applies the optical Element to a given beam
        :param beam:
        :return: a new beam
        """
        b = beam.clone_parameters()
        b.field = beam.field * self.complex_transmission
        return b

    def __add__(self, element2):
        """Adds two complex masks"""
        print('adding')
        if isinstance(element2, Element):
            f = self.complex_transmission + element2.complex_transmission
        else:
            raise TypeError('Unknown Element type')
        return Element(f)


class ArbitraryMask(Element):
    def __init__(self, complex_field):
        super().__init__(complex_field)


class circ_mask(Element):
    def __init__(self, beam, radius=1e-3, offx=0, offy=0, amp_val=0):
        amp = np.ones((beam.num, beam.num))
        xv = np.linspace(-beam.width / 2 - offx / 2, beam.width / 2 - offx / 2, num=beam.num)
        xy = np.linspace(-beam.width / 2 - offy / 2, beam.width / 2 - offy / 2, num=beam.num)
        xx, yy = np.meshgrid(xv, xy)
        self.R = np.sqrt(xx ** 2 + yy ** 2)
        amp[self.R <= radius] = amp_val
        super().__init__(amp)


class circ_aperture(Element):
    def __init__(self, beam, radius=1e-3, offx=0, offy=0, amp_val=0):
        amp = np.ones((beam.num, beam.num))
        xv = np.linspace(-beam.width / 2 - offx / 2, beam.width / 2 - offx / 2, num=beam.num)
        xy = np.linspace(-beam.width / 2 - offy / 2, beam.width / 2 - offy / 2, num=beam.num)
        xx, yy = np.meshgrid(xv, xy)
        self.R = np.sqrt(xx ** 2 + yy ** 2)
        amp[self.R >= radius] = amp_val
        super().__init__(amp)


class bessel_CGH(Element):
    def __init__(self, beam, kr, mask_radius=None, zmax=None):
        """Generates as Bessel function with complex transmission (ie amplitude can go negative)
        Parameters
        ----------
        beam - beam class to base Element on
        kr - float -radial frequency
        mask_radius - float (optional) apply a radial circ_aperture if set with this radius
        zmax - float - (optional) ignore kr and calculte max beam range. Use beam.width or mask_radius if set.
        """
        x = np.linspace(-beam.width / 2, beam.width / 2, num=beam.num)
        xv, yv = np.meshgrid(x, x)
        r = np.sqrt(xv ** 2 + yv ** 2)
        # F=CircAperture(Bw*4,0,0,F)
        Bw = beam.width / 2
        k = 2 * np.pi / beam.wavelength
        if zmax is not None:
            if mask_radius is not None:
                Bw = mask_radius

            kr = 2.4048 / Bw

        J = ss.j0(kr * r)
        if mask_radius is not None:
            c = circ_aperture(beam, radius=mask_radius)
            J = J * c.complex_transmission
        logger.info('Bessel beam kr {:.3f}, max range {:.3f}m'.format(kr, Bw * k / kr))
        super().__init__(J)


class square_mask(Element):
    def __init__(self, beam, size=1e-4, offx=0, offy=0, amp_val=0, ysize=None):
        if ysize is None:
            ysize = size

        num = beam.num
        amp = np.zeros((num, num))
        xv = np.linspace(-beam.width / 2 - offx / 2, beam.width / 2 - offx / 2, num=num)
        xy = np.linspace(-beam.width / 2 - offy / 2, beam.width / 2 - offy / 2, num=num)
        xx, yy = np.meshgrid(xv, xy)
        amp = (abs(xx) <= size) * (abs(yy) <= ysize)
        amp = np.invert(amp).astype(np.float)
        amp[amp == 0] = amp_val
        # gr.draw(amp,num=2,colorbar=True)
        # amp=amp*amp_val
        super().__init__(amp)


class square_apperture(Element):
    def __init__(self, beam, size=1e-4, offx=0, offy=0, amp_val=0, ysize=None):
        if ysize is None:
            ysize = size

        num = beam.num
        amp = np.zeros((num, num))
        xv = np.linspace(-beam.width / 2 - offx / 2, beam.width / 2 - offx / 2, num=num)
        xy = np.linspace(-beam.width / 2 - offy / 2, beam.width / 2 - offy / 2, num=num)
        xx, yy = np.meshgrid(xv, xy)
        amp = ((abs(xx) <= size) * (abs(yy) <= ysize)).astype(np.float)
        # amp = np.invert(amp).astype(np.float)
        amp[amp == 0] = amp_val
        # gr.draw(amp,num=2,colorbar=True)
        # amp=amp*amp_val
        super().__init__(amp)


class phase_wedge(Element):
    def __init__(self, beam, xsweep=0, ysweep=0, piston=0):
        num = beam.num
        # amp=np.zeros((num,num))
        xv = np.linspace(-xsweep / 2, xsweep / 2, num=num)
        xy = np.linspace(-ysweep / 2, ysweep / 2, num=num)
        xx, yy = np.meshgrid(xv, xy)
        phase = xx + yy + piston
        amp = np.exp(1j * phase)
        super().__init__(amp)


class phase_lens(Element):
    def __init__(self, beam, focal_length, offx=0, offy=0, radius=None, circular_mask = True):
        """
        Generates a phase lens
        :param beam: beam object
        :param focal_length: focal length
        :param offx: x offset, default 0
        :param offy: y offset, default 0
        :param radius: radius, defaults to beam width /2
        :param circular_mask: if true (default) apply circular mask of given radius, if false radius is used to define
        the length of the square edge
        """
        num = beam.num
        if radius is None:
            radius = beam.width / 2

        amp = np.zeros((num, num))
        xv = np.linspace(-beam.width / 2 - offx / 2, beam.width / 2 - offx / 2, num=num)
        xy = np.linspace(-beam.width / 2 - offy / 2, beam.width / 2 - offy / 2, num=num)
        xx, yy = np.meshgrid(xv, xy)
        R = np.sqrt(xx ** 2 + yy ** 2)
        if circular_mask:
            amp = R <= radius
        else:
            ampx = np.where( abs(xx)<=radius,1,0)
            ampy = np.where(abs(yy) <= radius, 1, 0)
            amp = ampx*ampy
        #  rad=math.sqrt(f**2-(beam.width/2)**2)
        phase = R ** 2 * np.pi / (beam.wavelength * focal_length)
        super().__init__(amp * np.exp(1j * phase))


class axicon(Element):
    def __init__(self, beam, phi, n1=1.5, offx=0, offy=0):
        pass
