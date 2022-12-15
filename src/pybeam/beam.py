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
        """ basic Beam container class,
        Parameters
        ----------
        amplitude - numpy array (real only) amplitude
        phase - numpy arrays (real only). phase will be set to zero if amplitude is set but phase is not. 
        input_field  - numpy array (complex data) sets a complex amplitude field. amplitude and phase are ignored if this is set. 
        wavelength - float. The wavelength of the Beam.
        width - float. The physical width of the array data 
        num  - int. The number of rows of pixels in the array (ignored if amplitude/phase/field are set).
        units - string. The physical name of the width unit. For reference only, it has not effect.
        name - string. The name of the Beam. For reference only, it has not effect.
        """
        self.units = units
        self.name = name
        self.phase = phase
        self.amplitude = amplitude
        # copy the arg list but exclude the numpy arrays
        self._arglist = [k for k in locals().keys() if k not in ['amplitude', 'phase', 'input_field', 'self', 'num']]
        if input_field is not None:
            self.field = input_field
        else:    
            self.amplitude = amplitude
            if phase is not None:
                self.phase = phase
            elif amplitude is not None:
                self.phase = np.zeros(amplitude.shape)
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
        """splits the amplitude by the amount of ratio
        :param ratio: split the beams by this ratio, defaults to 0.5
        :return: (a,b) two new Beam objects with amplitudes split by `ratio` and `1-ratio` respectively
        """
        a = self.copy(self)
        b = self.copy(self)
        a *= ratio
        b *= (1-ratio)
        return a, b

    def clone_parameters(self):
        """Creates a new Beam with the same parameters - but no amplitude/phase data"""
        self._arg_dict = {k: self.__dict__[k] for k in self._arglist}
        return Beam(**self._arg_dict)

    @property
    def num(self):
        """Returns size of data array, either from size of amplitude array if set or the num parameter""" 
        if self.phase is not None:
            return self.phase.shape[0]
        else:
            return self._num

    @property
    def field(self):
        """
        Complex field
        :return: numpy complex field
        """
        return self.amplitude*np.exp(1j*self.phase)

    @field.setter
    def field(self, val):
        """
        Complex field
        :param val: numpy complex field
        :return: nothing
        """
        self.amplitude = np.absolute(val)
        self.phase = np.angle(val)

    @property
    def intensity(self):
        """
        Intensity
        :return: numpy array
        """
        return self.amplitude**2

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
        :param amp_val: Beam amplitude, defaults to 1
        :param kwargs: passed to Beam class
        """
        self._size = size
        xv = np.linspace(-width/2-offx/2, width/2-offx/2, num=num)
        xy = np.linspace(-width/2-offy/2, width/2-offy/2, num=num)
        xx, yy = np.meshgrid(xv, xy)
        amp = (abs(xx) <= size) * (abs(yy) <= size)
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
