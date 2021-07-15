import numpy as np
from abc import ABC, abstractmethod
import logging
logger = logging.getLogger(__name__)

class Detector(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def apply(self):
        pass


class PointDetector(Detector):
    def __init__(self, beam, radius=1e-3, offx=0, offy=0):
        """
        Single circular point detector
        :param beam: input beam
        :param radius: radius of detector
        :param offx: position offset
        :param offy: position offset
        """
        xv = np.linspace(-beam.width / 2 - offx / 2, beam.width / 2 - offx / 2, num=beam.num)
        xy = np.linspace(-beam.width / 2 - offy / 2, beam.width / 2 - offy / 2, num=beam.num)
        xx, yy = np.meshgrid(xv, xy)
        R = np.sqrt(xx ** 2 + yy ** 2)
        amp = np.ones((beam.num, beam.num))
        amp[R > radius] = 0
        self.detector_area = amp

    def apply(self, beam):
        cap = beam.intensity * self.detector_area
        ints = np.max(cap)
        power = np.sum(cap)
        return ints, power
