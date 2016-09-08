import logging
import numpy as np

from spectral import *
import spectral.io.envi as envi

from msi.io.reader import Reader
from msi.msi import Msi


class EnviReader(Reader):

    def __init__(self):
        pass

    def read(self, fileToRead):
        """ read the envi image."""

        image = envi.open(fileToRead+'.hdr',
                          fileToRead+'.raw')

        # NOTE: This does not work for ordered dictionaries
        w= [float(s) for s in image.metadata.pop('wavelength')]
        image.metadata['wavelengths'] = np.array(w)
        msi = Msi(image[:,:,:image.nbands],image.metadata)
        return msi
