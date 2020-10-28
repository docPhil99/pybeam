#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 11:44:44 2018

@author: phil
"""

import numpy as np
#from copy import copy
import logging
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
#import PMB.imutils.graphics as gr

from beam import beam,square_beam,tophat_beam
from element import circ_mask, phase_lens
from display import displayInt, display
import logging
logger = logging.getLogger(__name__)

class propergator():
    
    @staticmethod
    def propFF(beam,z=1):
        M=beam.num
        dx1=beam.width/M
        k=2*np.pi/beam.wavelength
        #output size
        L2=beam.wavelength*z/dx1
        #dx2=beam.wavelength*z/beam.width
        x2=np.linspace(-L2/2,L2/2,M)
        XX,YY=np.meshgrid(x2,x2)
        c=1/(1j*beam.wavelength*z)*np.exp(1j*k/(2*z)*(XX**2+YY**2))
        u2=c*np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(beam.field)))*dx1**2
        b=beam.clone_parameters()
        b.width=L2
        b.field=u2
        return b
        
    
    
    @staticmethod
    def info(b,z,D):
        """Suggests which type of Fresnel transform to use
        
        Parameters
        ----------
        b - beam
        z - distance to propagate
        D - width of object
        Returns
        ---------
        Nothing, prints to stdout
        """
        
        critical_sample=b.wavelength*z/b.width
        delta_x=b.width/b.amplitude.shape[0]
        guardband_ratio=b.width/D
        Fresnel_number=D**2/(b.wavelength*z)
        
        print('Guardband ratio {:.2f}\nCritical space domain sample size {:f}\nActual sample size {:f}\n'.format(guardband_ratio,critical_sample,delta_x))
        if delta_x > critical_sample:
            print('Relatively short distance, TF preferred')
        
        else:
            print('Relatively long distance, IR preferred')
        if abs(delta_x - critical_sample)/critical_sample < 0.05:
            print('Although, critical sample condition met (within 5%). So you can use either IR or TF')
        print('Fresnel number {:f}\nIdeally Fresnel number should be less then 1, but it *maybe* ok higher if the function is smooth'.format(Fresnel_number))
    @staticmethod
    def propIR(b,z):
        M=b.amplitude.shape[0]
        #print('M ',M)
        dx=b.width/M  #sample inteval
        #print('dx ',dx)
        k=2*np.pi/b.wavelength
        #print('k ',k)
        x=np.arange(-b.width/2,b.width/2,dx)
        #print(x)
        X,Y=np.meshgrid(x,x)
        #create impulse
        h=1/(1j*b.wavelength*z)*np.exp(1j*k/(2*z)*(X**2+Y**2))
        #plt.imshow(np.angle(h))
        H=np.fft.fft2(np.fft.fftshift(h))*dx**2
        U1=np.fft.fft2(np.fft.fftshift(b.field))
        U2=H*U1
        u2=np.fft.ifftshift(np.fft.ifft2(U2))
        return beam(input_field=u2,wavelength=b.wavelength,width=b.width)
    @staticmethod
    def propTF(b,z):
        M=b.amplitude.shape[0]
        print('M ',M)
        dx=b.width/M  #sample inteval
        print('dx ',dx)
        #k=2*np.pi/b.wavelength
        #fx=np.arange(-1/(2*dx),1/(2*dx)+1/b.width,1/b.width)
        fx=np.linspace(-1/(2*dx),1/(2*dx),M)
        #print(fx)
        FX,FY=np.meshgrid(fx,fx)
        H=np.exp(-1j*np.pi*b.wavelength*z*(FX**2+FY**2))
        H=np.fft.fftshift(H)
        U1=np.fft.fft2(np.fft.fftshift(b.field))
        U2=H*U1
        U2=np.fft.ifftshift(np.fft.ifft2(U2))
        return beam(input_field=U2,wavelength=b.wavelength,width=b.width)

    @staticmethod
    def prop2step(b1,b2,z):
        
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
if __name__=='__main__':
    
    b=tophat_beam(num=512,radius=1.6e-3,width=10e-3)
    mask=circ_mask(b,radius=1.5e-3)
    b2=mask.apply(b)
    f=500e-3
    propergator.info(b2,f,1.5e-3)
    b3=propergator.propIR(b2,f)
    lens=phase_lens(b3,focal_length=f)
    b4=lens.apply(b3)
    delta_z=(1.5-0.0001)/11
    z=0.001
    for i in range(1,11):
        b4=propergator.propTF(b4,delta_z);
        
        plt.subplot(2,5,i)
        s='z= %3.1f m' % z
        plt.title(s)
        
        plt.imshow(b4.intensity,cmap='jet');plt.axis('off')
        
        z=z+delta_z
    plt.show()
    #raise Exception('stop')
    
    
    #test FF
    t=square_beam(offy=0,offx=0,size=0.022,width=0.5,num=250)
    z=2000
    propergator.info(t,z,t.size)
    t2=propergator.propFF(t,z)
    fig,ax=displayInt(t2,num=1,title='propFF',nthroot=3,cross_section=True)
    
    #show  analytic version
    x=np.linspace(-t.width/2,t.width/2,t.num)
    I2=(4*t.size**2/(t.wavelength*z))**2*np.sinc(2*t.size/(t.wavelength*z)*x)**2
    ax=fig.add_subplot(2,2,3)
    ax.plot(x,I2)
    #raise Exception('stop')
    
    b=beam(amplitude=np.ones((3,3)),phase=np.ones((3,3))*np.pi)
    #g=Gaussian_beam()
    t=square_beam(offy=0,offx=0,size=0.051,width=0.5)
    propergator.info(t,1000,00.051)
    
    
    
    
    
    t2=propergator.propIR(t,2000)
    t3=propergator.propTF(t,2000)
    displayInt(t2,cross_section=True,num=1,title='propIR')
    display(t2,num=2,title='propIR')
    
    displayInt(t3,cross_section=True,num=3,title='propTF')
    display(t3,num=4,title='propIR')
    
    fig,axs=plt.subplots(num=10,nrows=4,ncols=2)
    
    
    
    zlist=[1000,2000,4000,20000]
    for ind,z in enumerate(zlist):
        tir=propergator.propIR(t,z)
        ttr=propergator.propTF(t,z)
        bi=tir.intensity
        r=bi.shape[0]//2
        axs[ind,0].plot(np.linspace(-tir.width/2,tir.width/2,bi.shape[0]),bi[r,:])
        bi=ttr.intensity
        r=bi.shape[0]//2
        axs[ind,1].plot(np.linspace(-tir.width/2,tir.width/2,bi.shape[0]),bi[r,:])
        axs[ind,1].set_title('z={} TF'.format(z))
        axs[ind,0].set_title('z={} IR'.format(z))
    
    #b1=beam(amplitude=np.ones((2,2)),phase=np.ones((2,2))*np.pi/2)
    #b2=beam(amplitude=np.ones((2,2)))
    #print((b1+b2).amplitude)