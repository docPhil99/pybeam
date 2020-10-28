#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:25:55 2018

@author: phil
"""


import numpy as np
from copy import deepcopy
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import PMB.imutils.graphics as gr
import math
import scipy.special as ss



class beam():
    def  __init__(self,amplitude=None,phase=None,input_field=None,wavelength=500e-9,width=1e-3,num=512,units='m',name='beam'):
        """ basic beam container class, 
        Parameters
        ----------
        amplitude - numpy array (real only) amplitude
        phase - numpy arrays (real only). phase will be set to zero if amplitude is set but phase is not. 
        input_field  - numpy array (complex data) sets a complex amplitude field. amplitude and phase are ignored if this is set. 
        wavelength - float. The wavelength of the beam.
        width - float. The physical width of the array data 
        num  - int. The the number of rows of pixels in the array (ignored if amplitude/phase/field are set).
        units - string. The phyiscal name of the width unit. For reference only, it has not effect.
        name - string. The name of the beam. For reference only, it has not effect.
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
        self.wavelength= wavelength
        self.width = width
        self._num = num
        
    def __iadd__(self,beam2):
        """
        Adds beam2 complex field to the beam.
        :param beam2:  either a beam class or a scalar or numpy array of same size as beam
        :return: added beams
        """
        if isinstance(beam2, beam):
            # TODO test physical parameters match,
            self.field = self.field+beam2.field
        else:
            self.field = self.field+beam2
        return self

    def __add__(self,beam2):
        """
        Adds beam2 complex field to the beam.
        :param beam2:  either a beam class or a scalar or numpy array of same size as beam
        :return: added beams
        """
        # TODO test physical parameters match,
        if isinstance(beam2, beam):
            f = self.field+beam2.field
        else:
            f = self.field+beam2
        return beam(input_field=f)

    def __mul__(self, b2):
        """
        Multiples beam2 complex field to the beam.
        :param beam2:  either a beam class or a scalar or numpy array of same size as beam
        :return: multiplied beams
        """
        # TODO test physical parameters match,
        if isinstance(b2, beam):
            f = self.field*b2.field
        else:
            f = self.field*b2
        return beam(input_field=f)

    def __imul__(self, b2):
        """
        Multiples beam2 complex field to the beam.
        :param beam2:  either a beam class or a scalar or numpy array of same size as beam
        :return: multiplied beams
        """
        # TODO test physical parameters match, I don't think the inplace mul is needed, test this
        self.field = self.field*b2
        return self

    def copy(self):
        """
        :return: deepcopy of beam
        """
        return deepcopy(self)

    def split(self, ratio=0.5):
        """splits the amplitude by the amount of ratio
        Returns
        --------
        (a,b) new beam objects with ratio and 1-ratio repestively
        """
        a = self.copy(self)
        b = self.copy(self)
        a *= ratio
        b *= (1-ratio)
        return a, b

    def clone_parameters(self):
        """Creates a new beam with the same parameters - but no amplitude/phase data"""
        self._argdict = {k: self.__dict__[k] for k in self._arglist}
        return beam(**self._argdict)

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

    def add_buffer(self,number):
        """
        adds zero buffering to beam, also update physical width
        :param number: new width of beam
        :return: nothing
        """
        self.width = self.width/self.num * number
        logger.debug(f"New beam width = {self.width}")
        z = np.zeros((number,number),dtype=complex)
        offset = (number - self.num )//2
        end = (self.num + number)//2
        z[offset:end, offset:end]=self.field
        self.field = z

    def clip_beam(self, new_width):
        """
        Physically clip the beam with a rectangle mask, reduces physical size and number of pixels
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



class square_beam(beam):
    """
    Create a square beam
    """
    def __init__(self, num=512, size=1e-4, width=1e-3, offx=0, offy=0, amp_val=1, **kwargs):
        """
        Create square beam
        :param num: num of pixel wide
        :param size: physical size of array
        :param width: physical width of square beam
        :param offx: physical offset in x
        :param offy: physical offset in y
        :param amp_val: beam amplitude, defaults to 1
        :param kwargs: passed to beam class
        """
        self._size=size
        #amp=np.zeros((num,num))
        xv=np.linspace(-width/2-offx/2,width/2-offx/2,num=num)
        xy=np.linspace(-width/2-offy/2,width/2-offy/2,num=num)
        xx,yy=np.meshgrid(xv,xy)
        amp=(abs(xx)<=size) * (abs(yy)<=size)
        amp=amp*amp_val
        #amp[np.abs(xx)<=size & np.abs(yy)<=size]=amp_val
        super().__init__(amplitude=amp,width=width,**kwargs)

    @property
    def size(self):
        return self._size

class tophat_beam(beam):
    def __init__(self,num=512,radius=1e-4,width=1e-3,offx=0,offy=0,amp_val=1,**kwargs):
        #print(**kwargs)
        amp=np.zeros((num,num))
        xv=np.linspace(-width/2-offx/2,width/2-offx/2,num=num)
        xy=np.linspace(-width/2-offy/2,width/2-offy/2,num=num)
        xx,yy=np.meshgrid(xv,xy)
        self.R=np.sqrt(xx**2+yy**2)
        amp[self.R<=radius]=amp_val
        super().__init__(amplitude=amp,width=width,**kwargs)
        

class GaussianHermite_beam(beam):
    def __init__(self,w0,m,n,num=512, wavelength=500e-9,width=1e-3,amp_val=1):
        amp=Gaussian_beam.makeGaussian(num,fwhm=fwhm) #TODO fix this
        self._fwhm=fwhm
        super().__init__(amplitude=amp,wavelength=wavelength,width=width)


class Gaussian_beam(beam):
    def __init__(self,num=512,fwhm=0.5e-3, wavelength=500e-9,width=1e-3):
        fwhm_p = fwhm/width*num
        amp=Gaussian_beam._makeGaussian(num,fwhm=fwhm_p)
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


if __name__=='__main__':
    from element import square_mask
    from display import display
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(ch)
    b=beam(amplitude=np.ones((100,100)),width=3e-3)
    sq=square_mask(b,size=0.2e-3,offx=0.1e-3,amp_val=0.1,ysize=.1e-3 )
    #sq=bessel_CGH(b,2.405/b.width*2,mask_radius=b.width/2)
    c=sq.apply(b)
    c.add_buffer(201)
    display(c,colorbar=True,num=1)
    raise Exception('stop')
    pw=phase_wedge(b,xsweep=2*np.pi,ysweep=20*np.pi,piston=1)
    d=pw.apply(b)
    display(d,colorbar=True,num=3)
    #combine beams
    e=b+d
    displayInt(e,colorbar=True,num=4)
    lens=phase_lens(b,f=3e-2)
    l=lens.apply(b)
    display(l,colorbar=True,num=5)

