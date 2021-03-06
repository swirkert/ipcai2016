
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
Created on Sep 10, 2015

@author: wirkert
'''



import os

import pandas as pd
import numpy as np
import luigi
from scipy.interpolate import interp1d
from sklearn.preprocessing import normalize

import commons
import mc.dfmanipulations as dfmani
from msi.io.spectrometerreader import SpectrometerReader

sc = commons.ScriptCommons()


class SpectrometerFile(luigi.Task):
    input_file = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.input_file)


class FilterTransmission(luigi.Task):
    input_file = luigi.Parameter()

    def requires(self):
        return SpectrometerFile(self.input_file)

    def output(self):
        return luigi.LocalTarget(os.path.join(sc.get_full_dir("INTERMEDIATES_FOLDER"),
            "processed_transmission" + os.path.split(self.input_file)[1]))

    def run(self):
        reader = SpectrometerReader()
        filter_transmission = reader.read(self.input().path)
        # filter high and low _wavelengths
        wavelengths = filter_transmission.get_wavelengths()
        fi_image = filter_transmission.get_image()
        #fi_image[wavelengths < 450 * 10 ** -9] = 0.0
        #fi_image[wavelengths > 720 * 10 ** -9] = 0.0
        # filter elements farther away than +- 30nm
        file_name = os.path.split(self.input_file)[1]
        name_to_float = float(os.path.splitext(file_name)[0])
        #fi_image[wavelengths < (name_to_float - 30) * 10 ** -9] = 0.0
        #fi_image[wavelengths > (name_to_float + 30) * 10 ** -9] = 0.0
        # elements < 0 are set to 0.
        fi_image[fi_image < 0.0] = 0.0

        # write it to a dataframe
        df = pd.DataFrame()
        df["wavelengths"] = wavelengths
        df["reflectances"] = fi_image
        df.to_csv(self.output().path, index=False)


class JoinBatches(luigi.Task):
    df_prefix = luigi.Parameter()
    eval_name = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(sc.get_full_dir("INTERMEDIATES_FOLDER"),
                                              _make_file_name_prefix(self.eval_name, self.df_prefix) +
                                              "all" + ".txt"))

    def run(self):
        # get all files in the search path
        batch_folder = os.path.join(sc.get_full_dir("MC_DATA_FOLDER"), self.df_prefix)
        batches = os.listdir(batch_folder)

        # first get full path for each batch
        batches = [os.path.join(batch_folder, f) for f in batches]
        # now check if it is actually a file
        batches = [f for f in batches if os.path.isfile(f)]

        # load these files
        dfs = [pd.read_csv(f, header=[0, 1]) for f in batches]
        # now join them to one batch
        joined_df = pd.concat(dfs, ignore_index=True)
        # write it
        joined_df.to_csv(self.output().path, index=False)


class CameraBatch(luigi.Task):
    """takes a batch of reference data and converts it to the spectra
    processed by a camera with the specified wavelengths assuming a 10nm FWHM"""
    df_prefix = luigi.Parameter()
    eval_name = luigi.Parameter()

    def requires(self):
        return JoinBatches(self.df_prefix, self.eval_name)

    def output(self):
        return luigi.LocalTarget(os.path.join(sc.get_full_dir("INTERMEDIATES_FOLDER"),
                                              _make_file_name_prefix(self.eval_name, self.df_prefix) +
                                              "_all_virtual_camera.txt"))

    def run(self):
        # load dataframe
        df = pd.read_csv(self.input().path, header=[0, 1])
        # camera batch creation:
        dfmani.fold_by_fwhm(df, 10*10**-9)
        dfmani.interpolate_wavelengths(df, sc.other["RECORDED_WAVELENGTHS"])
        # write it
        df.to_csv(self.output().path, index=False)


class SpectrometerBatch(luigi.Task):
    """takes a batch of reference data and converts it to the spectra
    processed by a camera with the specified wavelengths assuming a 10nm FWHM"""
    df_prefix = luigi.Parameter()
    boxcar_smoothing = luigi.IntParameter() # how big is the boxcar to smooth
    # (in 2nm). So 8nm boxcars would be 4
    eval_name = luigi.Parameter()

    def requires(self):
        return JoinBatches(self.df_prefix)

    def output(self):
        return luigi.LocalTarget(os.path.join(sc.get_full_dir("INTERMEDIATES_FOLDER"),
                                              _make_file_name_prefix(self.eval_name, self.df_prefix) +
                                              str(self.boxcar_smoothing) + "boxcar"
                                              "_all_spectrometer.txt"))

    def run(self):
        # load dataframe
        df = pd.read_csv(self.input().path, header=[0, 1])
        # camera batch creation:
        dfmani.fold_by_sliding_average(df, self.boxcar_smoothing)
        dfmani.interpolate_wavelengths(df, sc.other["RECORDED_WAVELENGTHS"])
        # write it
        df.to_csv(self.output().path, index=False)


class SpectroCamBatch(luigi.Task):
    """
    Same as CameraBatch in purpose but adapts the batch to our very specific
    SpectroCam set-up.
    """
    df_prefix = luigi.Parameter()
    eval_name = luigi.Parameter()

    def requires(self):
        # all wavelengths must have been measured for transmission and stored in
        # wavelength.txt files (e.g. 470.txt)
        filenames = ((sc.other["RECORDED_WAVELENGTHS"] * 10**9).astype(int)).astype(str)
        filenames = map(lambda name: FilterTransmission(os.path.join(sc.get_full_dir("FILTER_TRANSMISSIONS"),
                                                                     name +
                                                                     ".txt")),
                        filenames)

        return JoinBatches(self.df_prefix, self.eval_name), filenames

    def output(self):
        return luigi.LocalTarget(os.path.join(sc.get_full_dir("INTERMEDIATES_FOLDER"),
                                              _make_file_name_prefix(self.eval_name, self.df_prefix) +
                                              "_all_spectrocam.txt"))

    def run(self):
        # load dataframe
        df = pd.read_csv(self.input()[0].path, header=[0, 1])
        # camera batch creation:
        new_reflectances = []
        for band in self.input()[1]:
            df_filter = pd.read_csv(band.path)
            interpolator = interp1d(df_filter["wavelengths"],
                                    df_filter["reflectances"],
                                    assume_sorted=False, bounds_error=False)
            # use this to create new reflectances
            interpolated_filter = interpolator(df["reflectances"].
                                               columns.astype(float))
            # normalize band response
            normalize(interpolated_filter.reshape(1, -1), norm='l1', copy=False)
            folded_reflectance = np.dot(df["reflectances"].values,
                                        interpolated_filter)
            new_reflectances.append(folded_reflectance)
        new_reflectances = np.array(new_reflectances).T
        dfmani.switch_reflectances(df, sc.other["RECORDED_WAVELENGTHS"],
                                   new_reflectances)
        # write it
        df.to_csv(self.output().path, index=False)


def _make_file_name_prefix(eval_name, mc_data_name):
    return eval_name + "_" + mc_data_name + "_"