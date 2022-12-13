
from pybeam import SquareMask, PhaseWedge, PhaseLens
from pybeam import display, drawnow, displayInt, Beam

import numpy as np
from loguru import logger

if __name__ == '__main__':

    b = Beam(amplitude=np.ones((100, 100)), width=3e-3)
    sq = SquareMask(b, size=0.2e-3, offx=0.1e-3, amp_val=0.1, ysize=.1e-3)
    #sq=BesselCGH(beam,2.405/beam.width*2,mask_radius=beam.width/2)
    c = sq.apply(b)
    c.add_buffer(201)
    display(c,colorbar=True,num=1)


    pw = PhaseWedge(b,xsweep=2*np.pi,ysweep=20*np.pi,piston=1)
    d = pw.apply(b)
    display(d,colorbar=True,num=3)
    #combine beams
    e = b+d
    displayInt(e,colorbar=True,num=4)
    lens = PhaseLens(b,focal_length=3e-2)
    l = lens.apply(b)
    display(l,colorbar=True,num=5)
    drawnow()
