#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 11:44:44 2018

@author: phil
"""
from abc import ABC
import numpy as np
import logging
logger = logging.getLogger(__name__)



from .beam import Beam,SquareBeam,TophatBeam

import logging
logger = logging.getLogger(__name__)

class propergator(ABC):
    
    @staticmethod
    def propFF(beam: Beam, z: float = 1) -> Beam:
        """
        Fraunhofer propagation
        :param beam:
        :param z: distance to propagate in physical units
        :return: Beam
        """
        M=beam.num
        dx1=beam.width/M
        k=2*np.pi/beam.wavelength
        #output size
        L2=beam.wavelength*z/dx1
        #dx2=Beam.wavelength*z_distance/Beam.width
        x2=np.linspace(-L2/2,L2/2,M)
        XX,YY=np.meshgrid(x2,x2)
        c=1/(1j*beam.wavelength*z)*np.exp(1j*k/(2*z)*(XX**2+YY**2))
        u2=c*np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(beam.field)))*dx1**2
        b=beam.clone_parameters()
        b.width=L2
        b.field=u2
        return b
        
    
    
    @staticmethod
    def info(beam: Beam, z_distance: float, object_width: float) -> None:
        """Suggests which type of Fresnel transform to use
        
        Parameters
        ----------
        beam - Beam
        z_distance - distance to propagate
        object_width - width of object
        Returns
        ---------
        Nothing, prints to stdout
        """
        
        critical_sample= beam.wavelength * z_distance / beam.width
        delta_x = beam.width / beam.amplitude.shape[0]
        guardband_ratio= beam.width / object_width
        Fresnel_number= object_width ** 2 / (beam.wavelength * z_distance)
        
        print('Guardband ratio {:.2f}\nCritical space domain sample size {:f}\nActual sample size {:f}\n'.format(guardband_ratio,critical_sample,delta_x))
        if delta_x > critical_sample:
            print('Relatively short distance, TF preferred')
        
        else:
            print('Relatively long distance, IR preferred')
        if abs(delta_x - critical_sample)/critical_sample < 0.05:
            print('Although, critical sample condition met (within 5%). So you can use either IR or TF')
        print('Fresnel number {:f}\nIdeally Fresnel number should be less then 1, but it *maybe* ok higher if the function is smooth'.format(Fresnel_number))

    @staticmethod
    def propIR(beam: Beam, z_distance: float) -> Beam:
        """
        propagate beam with impulse response method
        :param beam:
        :param z_distance:
        :return: propagated beam
        """
        M=beam.amplitude.shape[0]
        #print('M ',M)
        dx= beam.width / M  #sample inteval
        #print('dx ',dx)
        k= 2 * np.pi / beam.wavelength
        #print('k ',k)
        x=np.arange(-beam.width / 2, beam.width / 2, dx)
        #print(x)
        X,Y=np.meshgrid(x,x)
        #create impulse
        h= 1 / (1j * beam.wavelength * z_distance) * np.exp(1j * k / (2 * z_distance) * (X ** 2 + Y ** 2))
        #plt.imshow(np.angle(h))
        H=np.fft.fft2(np.fft.fftshift(h))*dx**2
        U1=np.fft.fft2(np.fft.fftshift(beam.field))
        U2=H*U1
        u2=np.fft.ifftshift(np.fft.ifft2(U2))
        return Beam(input_field=u2, wavelength=beam.wavelength, width=beam.width)

    @staticmethod
    def propTF(beam: Beam, z_distance: float) -> Beam:
        """
        propagate beam with transfer function method
        :param beam:
        :param z_distance:
        :return: propagated beam
        """
        M=beam.amplitude.shape[0]
        print('M ',M)
        dx= beam.width / M  #sample inteval
        print('dx ',dx)
        #k=2*np.pi/beam.wavelength
        #fx=np.arange(-1/(2*dx),1/(2*dx)+1/beam.width,1/beam.width)
        fx=np.linspace(-1/(2*dx),1/(2*dx),M)
        #print(fx)
        FX,FY=np.meshgrid(fx,fx)
        H=np.exp(-1j * np.pi * beam.wavelength * z_distance * (FX ** 2 + FY ** 2))
        H=np.fft.fftshift(H)
        U1=np.fft.fft2(np.fft.fftshift(beam.field))
        U2=H*U1
        U2=np.fft.ifftshift(np.fft.ifft2(U2))
        return Beam(input_field=U2, wavelength=beam.wavelength, width=beam.width)

    @staticmethod
    def prop2step(b1,b2,z):
        """
        TODO
        :param b1:
        :param b2:
        :param z:
        :return:
        """
        k=2*np.pi/b1.wavelength
        #source plane
        x1=np.linspace(-b1.width/2,b1.width/2,b1.num)
        X,Y=np.meshgrid(x1,x1)
        u=b1.field*np.exp(1j*k/(2*z*b1.width)*(b1.width-b2.width)*(X**2+Y**2))
        u=np.fft.fft2(np.fft.fftshift(u))
        
        #dummy plane
        dx1=b1.width/b1.num
        fx1=np.linspace(-1/(2*dx1),1/(2*dx1),b1.num)
        fx1=np.fft.fftshift(fx1)
        FX1,FY1=np.meshgrid(fx1,fx1)
        u=np.exp(-1j*np.pi*b1.wavelength*z*b1.width/b2.width*(FX1**2+FY1**2))*u
        u=np.fft.ifftshift(np.fft.ifft2(u))
        #ob plane
        dx2=b2.width/b1.num
        x2=np.linspace(-b2.width/2,b2.width/2,b1.num)
        X,Y=np.meshgrid(x2,x2)
        u2=(b1.width/b2.width)*u*np.exp(-1j*k/(2*z*b2.width)*(b1.width-b2.width)*(X**2+Y**2))
        u2=u2*dx1**2/dx2**2
        b3=b1.clone_parameters(b1)
        b3.width=b2.width
        b3.field=u2
        return b3
