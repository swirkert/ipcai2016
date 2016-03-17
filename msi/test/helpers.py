# -*- coding: utf-8 -*-
"""

ipcai2016

Copyright (c) German Cancer Research Center,
Computer Assisted Interventions.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

See LICENSE for details

"""

"""
Created on Thu Aug 13 09:53:52 2015

@author: wirkert
"""

import numpy as np
from msi.msi import Msi


def getFakeMsi():

        # build a fake multispectral image with 5 dimensions.
    image = np.concatenate((np.ones((5, 5, 1)),
                            np.ones((5, 5, 1)) * 2,
                            np.ones((5, 5, 1)) * 3,
                            np.ones((5, 5, 1)) * 4,
                            np.ones((5, 5, 1)) * 5),
                           axis=-1)
    msi = Msi(image)

    msi.set_wavelengths(np.array([5, 4, 3, 2, 1]))

    return msi
