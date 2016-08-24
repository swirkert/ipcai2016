
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
'''
Created on Oct 19, 2015

@author: wirkert
'''

import numpy as np

from mc.usuag import get_haemoglobin_extinction_coefficients


class LinearSaO2Unmixing(object):
    '''
    classdocs
    '''

    def __init__(self, wavelengths, fwhm):
        # oxygenated haemoglobin extinction coefficients
        eHbO2_map , eHb_map = \
            get_haemoglobin_extinction_coefficients()

        eHbO2 = []
        eHb = []
        for w in wavelengths:
            # adapt absorption spectra to waveband specified by fwhm
            waveband = np.linspace(w - fwhm/2., w + fwhm/2., 100)
            eHbO2_w = np.sum(eHbO2_map(waveband)) / len(waveband)
            eHb_w = np.sum(eHb_map(waveband)) / len(waveband)
            # add it to our list
            eHbO2.append(eHbO2_w)
            eHb.append(eHb_w)

        eHbO2 = np.array(eHbO2)
        eHb = np.array(eHb)

        nr_total_wavelengths = len(wavelengths)
        # to account for scattering losses we allow a constant offset
        scattering = np.ones(nr_total_wavelengths)
        # put eHbO2, eHb and scattering term in one measurement matrix
        self.H = np.vstack((eHbO2, eHb, scattering)).T
        self.lsq_solution_matrix = np.dot(np.linalg.inv(np.dot(self.H.T,
                                                               self.H)),
                                          self.H.T)

    def fit(self, X, y, weights=None):
        """only implemented to fit to the standard sklearn framework."""
        pass

    def predict(self, X):
        """predict like in sklearn:

        Parameters:
            X: nrsamples x nr_features matrix of samples to predict for
            regression

        Returns:
            y: array of shape [nr_samples] with values for predicted
            oxygenation """
        # do least squares estimation
        oxy_test, deoxy, s = np.dot(self.lsq_solution_matrix, X.T)
        # calculate oxygenation = oxygenated blood / total blood
        saO2 = oxy_test / (oxy_test + deoxy)

        self.last_solution = oxy_test, deoxy, s

        return np.clip(saO2, 0., 1.)
