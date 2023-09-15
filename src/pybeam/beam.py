#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:25:55 2018

@author: phil
"""


import numpy as np
from copy import deepcopy
from loguru import logger


class Beam:
    def __init__(self, amplitude=None, phase=None, input_field=None, wavelength=500e-9, width=1e-3, num=512,
                  units='m', name='Beam'):
        """The basic Beam class. It holds the complex amplitude and physical parameters such as width and wavelength

        :param amplitude - numpy array (real only) _amplitude
        :param phase - numpy arrays (real only). _phase will be set to zero if _amplitude is set but _phase is not.
        :param input_field  - numpy array (complex data) sets a complex _amplitude field. _amplitude and _phase are ignored if this is set.
        :param wavelength - float. The wavelength of the Beam.
        :param width - float. The physical width of the array data
        :param num  - int. The number of rows of pixels in the array (ignored if _amplitude/_phase/field are set).
        :param units - string. The physical name of the width unit. For reference only, it has not effect.
        :param name - string. The name of the Beam. For reference only, it has not effect.
        """
        self.units = units
        self.name = name
        self._phase = phase
        self._amplitude = amplitude
        # copy the arg list but exclude the numpy arrays
        self._arglist = [k for k in locals().keys() if k not in ['amplitude', 'phase', 'input_field', 'self', 'num']]
        if input_field is not None:
            self.field = input_field
        else:    
            self._amplitude = amplitude
            if phase is not None:
                self._phase = phase
                if amplitude is None:
                    self._amplitude = np.ones(phase.shape)
            elif amplitude is not None:
                self._phase = np.zeros(amplitude.shape)
        self.wavelength = wavelength
        self.width = width
        self._num = num
        
    def __iadd__(self, beam2):
        """
        Adds beam2 complex field to the Beam.
        :param beam2:  either a Beam class or a scalar or numpy array of same size as Beam
        :return: added beams
        """
        if isinstance(beam2, Beam):
            # TODO test physical parameters match,
            self.field = self.field+beam2.field
        else:
            self.field = self.field+beam2
        return self

    def __sub__(self, beam2):
        """
        Subtracts beam2 complex field to the Beam.
        :param beam2:  either a Beam class or a scalar or numpy array of same size as Beam
        :return: added beams
        """
        # TODO test physical parameters match,
        if isinstance(beam2, Beam):
            f = self.field - beam2.field
        else:
            f = self.field - beam2
        return Beam(input_field=f)

    def __add__(self, beam2):
        """
        Adds beam2 complex field to the Beam.
        :param beam2:  either a Beam class or a scalar or numpy array of same size as Beam
        :return: added beams
        """
        # TODO test physical parameters match,
        if isinstance(beam2, Beam):
            f = self.field+beam2.field
        else:
            f = self.field+beam2
        return Beam(input_field=f)

    def __mul__(self, b2):
        """
        Multiples beam2 complex field to the Beam.
        :param b2:  either a Beam class or a scalar or numpy array of same size as Beam
        :return: multiplied beams
        """
        # TODO test physical parameters match,
        if isinstance(b2, Beam):
            f = self.field*b2.field
        else:
            f = self.field*b2
        return Beam(input_field=f)

    def __imul__(self, b2):
        """
        Multiples beam2 complex field to the Beam.
        :param b2:  either a Beam class or a scalar or numpy array of same size as Beam
        :return: multiplied beams
        """
        # TODO test physical parameters match, I don't think the inplace mul is needed, test this
        self.field = self.field*b2
        return self

    def copy(self):
        """
        :return: deepcopy of Beam
        """
        return deepcopy(self)

    def split(self, ratio=0.5):
        """splits the _amplitude by the amount of ratio
        :param ratio: split the beams by this ratio, defaults to 0.5
        :return: (a,b) two new Beam objects with amplitudes split by `ratio` and `1-ratio` respectively
        """
        a = self.copy(self)
        b = self.copy(self)
        a *= ratio
        b *= (1-ratio)
        return a, b

    def clone_parameters(self):
        """Creates a new Beam with the same parameters - but no _amplitude/_phase data"""
        self._arg_dict = {k: self.__dict__[k] for k in self._arglist}
        return Beam(**self._arg_dict)

    @property
    def num(self):
        """Returns size of data array, either from size of _amplitude array if set or the num parameter"""
        if self._phase is not None:
            return self._phase.shape[0]
        else:
            return self._num

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value

    @property
    def amplitude(self):
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value):
        self._amplitude = value

    @property
    def field(self):
        """
        Complex field
        :return: numpy complex field
        """
        return self._amplitude * np.exp(1j * self._phase)

    @field.setter
    def field(self, val):
        """
        Complex field
        :param val: numpy complex field
        :return: nothing
        """
        self._amplitude = np.absolute(val)
        self._phase = np.angle(val)

    @property
    def intensity(self):
        """
        Intensity
        :return: numpy array
        """
        return self._amplitude ** 2

    def add_buffer(self, number):
        """
        adds zero buffering to Beam, also updates physical width
        :param number: new width of Beam
        :return: nothing
        """
        self.width = self.width/self.num * number
        logger.debug(f"New Beam width = {self.width}")
        z = np.zeros((number, number), dtype=complex)
        offset = (number - self.num)//2
        end = (self.num + number)//2
        z[offset:end, offset:end]=self.field
        self.field = z

    def clip_beam(self, new_width):
        """
        Physically clip the Beam with a rectangle mask, reduces physical size and number of pixels
        :param new_width: new physical size
        :return: nothing
        """
        trim=self.width-new_width
        new_num = int(self.num*new_width//self.width)
        pixel_per_mm = self.num/self.width
        trim_pix = int(trim*pixel_per_mm/2)
        logger.debug(f'new width {new_width} old width {self.width} physical trim {trim} new num {new_num} '
                     f'trim pixels {trim_pix} old num {self.num}')
        self.field = self.field[trim_pix:new_num+trim_pix,trim_pix:new_num+trim_pix]
        logger.debug(f'New field size {self.field.shape}')


class SquareBeam(Beam):
    """
    Create a square Beam
    """
    def __init__(self, num=512, size=1e-4, width=1e-3, offx=0, offy=0, amp_val=1, **kwargs):
        """
        Create square Beam
        :param num: num of pixel wide
        :param size: physical size of array
        :param width: physical width of square Beam
        :param offx: physical offset in x
        :param offy: physical offset in y
        :param amp_val: Beam _amplitude, defaults to 1
        :param kwargs: passed to Beam class
        """
        self._size = size
        xv = np.linspace(-width/2-offy/2, width/2-offy/2, num=num)
        xy = np.linspace(-width/2-offx/2, width/2-offx/2, num=num)
        xx, yy = np.meshgrid(xv, xy)
        amp = (abs(xx) <= size/2) * (abs(yy) <= size/2)
        amp = amp*amp_val
        #amp[np.abs(xx)<=size & np.abs(yy)<=size]=amp_val
        super().__init__(amplitude=amp, width=width, **kwargs)

    @property
    def size(self):
        return self._size


class TophatBeam(Beam):
    def __init__(self, num=512, radius=1e-4, width=1e-3, offx=0, offy=0, amp_val=1, **kwargs):
        """
        Tophat beam
        """
        amp = np.zeros((num, num))
        xv = np.linspace(-width/2-offx/2,width/2-offx/2,num=num)
        xy = np.linspace(-width/2-offy/2,width/2-offy/2,num=num)
        xx, yy = np.meshgrid(xv, xy)
        self.R = np.sqrt(xx**2+yy**2)
        amp[self.R <= radius] = amp_val
        super().__init__(amplitude=amp, width=width, **kwargs)
        

class GaussianHermiteBeam(Beam):
    def __init__(self,w0,m,n,num=512, wavelength=500e-9,width=1e-3, amp_val=1):
        raise NotImplementedError()
        amp=GaussianBeam.makeGaussian(num, fwhm=fwhm) #TODO fix this
        self._fwhm=fwhm
        super().__init__(amplitude=amp,wavelength=wavelength,width=width)


class GaussianBeam(Beam):
    def __init__(self,num=512,fwhm=0.5e-3, wavelength=500e-9,width=1e-3):
        fwhm_p = fwhm/width*num
        amp=GaussianBeam._makeGaussian(num, fwhm=fwhm_p)
        self._fwhm=fwhm
        super().__init__(amplitude=amp,wavelength=wavelength,width=width)

    @property
    def fwhm(self):
        return self._fwhm
        
    @staticmethod
    def _makeGaussian(size, fwhm = 3, center=None):
        """ Make a square gaussian kernel.
    
        size is the length of a side of the square
        fwhm is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
    
        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]
    
        if center is None:
            x0 = y0 = size // 2
        else:
            x0 = center[0]
            y0 = center[1]
    
        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
