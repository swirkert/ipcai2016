
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
Created on Oct 26, 2015

@author: wirkert
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer


def preprocess2(df, nr_samples=None, snr=None,
                magnification=None, bands_to_sortout=None, y_parameters=None):

    if y_parameters is None:
        y_parameters = ["sao2", "vhb"]

    # first set 0 reflectances to nan
    df["reflectances"] = df["reflectances"].replace(to_replace=0.,
                                                    value=np.nan)
    # remove nan
    df = df.dropna(axis=0)

    # extract nr_samples samples from data
    if nr_samples is not None:
        df = df.sample(nr_samples)

    # get reflectance and oxygenation
    X = df.reflectances
    if bands_to_sortout is not None and bands_to_sortout.size > 0:
        X.drop(X.columns[bands_to_sortout], axis=1, inplace=True)
        snr = np.delete(snr, bands_to_sortout)
    X = X.values
    y = df.layer0[y_parameters]

    # do data magnification
    if magnification is not None:
        X_temp = X
        y_temp = y
        for i in range(magnification - 1):
            X = np.vstack((X, X_temp))
            y = pd.concat([y, y_temp])

    # add noise to reflectances
    X = add_snr(X, snr)

    X = np.clip(X, 0.00001, 1.)
    # do normalizations
    X = normalize(X)
    return X, y


def add_snr(X, snr):
    camera_noise = 0.
    if snr is not None:
        sigmas = X / snr
        noises = np.random.normal(loc=0., scale=1, size=X.shape)
        camera_noise = sigmas*noises
    return X + camera_noise


def preprocess(batch, nr_samples=None, snr=None, movement_noise_sigma=None,
               magnification=None, bands_to_sortout=None):
    X, y = preprocess2(batch, nr_samples, snr, movement_noise_sigma,
                       magnification, bands_to_sortout)

    return X, y["sao2"]


def normalize(X):
    # normalize reflectances
    normalizer = Normalizer(norm='l1')
    X = normalizer.transform(X)
    # reflectances to absorption
    absorptions = -np.log(X)
    X = absorptions
    # get rid of sorted out bands
    normalizer = Normalizer(norm='l2')
    X = normalizer.transform(X)
    return X
