# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 10:15:48 2016

@author: phil
"""

# import scipy
import numpy as np


def ints(a):
    """return mod squared result"""
    return np.real(a * np.conjugate(a))


def max2(a):
    """max of N-D array"""
    if type(a) is list:
        a = np.array(a)
    return a.max(), np.unravel_index(a.argmax(), a.shape)


def min2(a):
    """min of N-D array"""
    if type(a) is list:
        a = np.array(a)
    return a.min(), np.unravel_index(a.argmin(), a.shape)


def mean2(a):
    """mean of n-d array

    Only calls numpy mean, its really not needed"""
    return np.mean(a)


def sfft2(a):
    """2d fft with shifting"""
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(a)))


def isfft2(a):
    """2d ifft with shifting"""
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(a)))
