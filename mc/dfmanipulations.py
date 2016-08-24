
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

from scipy.interpolate import interp1d
import sklearn.preprocessing
import pandas as pd
import numpy as np


def fold_by_sliding_average(df, window_size):
    """take a batch and apply a sliding average with given window size to
    the reflectances.
    window_size is elements to the left and to the right.
    There will be some boundary effect on the edges."""
    # next line does the folding.
    df.reflectances = pd.rolling_mean(df.reflectances.T, window_size,
                                      center=True).T
    # let's get rid of NaN columns which are created at the boundaries
    df.dropna(axis="columns", inplace=True)
    return df


def fold_by_fwhm(df, fwhm):
    """
    fold reflectances given in dataframe by fwhm. assumes rectangular window.
    :param df: created simulation batch
    :param fwhm: full width at half maximum [m]
    :return:
    """
    # build an interpolator using the inormation provided by the dataframes
    # reflectance column
    wavelengths = df.reflectances.columns.astype(float)
    interpolator = interp1d(wavelengths,
                            df.reflectances.as_matrix(), assume_sorted=False,
                            bounds_error=False)
    new_reflectances = []
    for w in wavelengths:
        waveband = np.linspace(w - fwhm/2., w + fwhm/2., 100)
        new_r = np.sum(interpolator(waveband), axis=1) / len(waveband)
        new_reflectances.append(new_r)
    new_reflectances = np.array(new_reflectances).T
    switch_reflectances(df, wavelengths, new_reflectances)
    # let's get rid of NaN columns which are created at the boundaries
    df.dropna(axis="columns", inplace=True)


def switch_reflectances(df, new_wavelengths, new_reflectances):
    """
    TODO: document and test

    :param df:
    :param new_wavelengths:
    :param new_reflectances:
    :return:
    """
    df.drop(df["reflectances"].columns, axis=1, level=1, inplace=True)
    for i, nw in enumerate(new_wavelengths):
        df["reflectances", nw] = new_reflectances[:, i]
    return df


def normalize_reflectances(df):
    r = df["reflectances"]
    r = sklearn.preprocessing.normalize(r, norm="l1")
    df["reflectances"] = r
    return df


def interpolate_wavelengths(df, new_wavelengths):
    """ interpolate image data to fit new_wavelengths. Current implementation
    performs simple linear interpolation. Neither existing nor new _wavelengths
    need to be sorted. """
    # build an interpolator using the inormation provided by the dataframes
    # reflectance column
    interpolator = interp1d(df.reflectances.columns.astype(float),
                            df.reflectances.as_matrix(), assume_sorted=False,
                            bounds_error=False)
    # use this to create new reflectances
    new_reflectances = interpolator(new_wavelengths)
    # build a new dataframe out of this information and set the original df
    # to the new information. This seems hacky, can't it be done easier?
    switch_reflectances(df, new_wavelengths, new_reflectances)
    return df


def to_df(msi):
    """
    TODO: document, write and test
    :param msi:
    :return:
    """
    pass