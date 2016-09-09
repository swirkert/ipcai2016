

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:09:18 2015

@author: wirkert
"""

import os
import logging
import datetime
import numpy as np
import pandas as pd
from scipy import interpolate

import matplotlib.pylab as plt
import seaborn as sns

from mc.camera import ImagingSystem, transform_color
import get_camera_calibration_info as cc
import get_spectrometer_measurements as sm
import get_camera_reflectances as cr

import commons

sc = commons.ScriptCommons()


# the principle folder where the spectrometer measurements reside in
TOP = "/media/wirkert/data/Data/2016_09_08_Ximea"
EXPERIMENT = "color_targets"
CAMERA = "ximea"

color_tiles_loc = "Color_tiles_exposure_adapted"
#color_tiles_loc = "Color_tiles_exposure_8000us"
pixel_location = (80, 228)
window_size = (50, 65)

# create a folder for the results if necessary
sc.set_root(TOP)

sc.add_dir("INTERMEDIATES_FOLDER",
           os.path.join(sc.get_dir("INTERMEDIATES_FOLDER"), EXPERIMENT))

sc.add_dir("COLOR_TARGET_RESULTS",
           os.path.join(sc.get_dir("RESULTS_FOLDER"), EXPERIMENT, color_tiles_loc + "_px_" + str(pixel_location)) + "_ws_" + str(window_size))


def plot_imaging_system(F, w):
    # bring to same scale
    F = F / F.max().max()
    w = w / w.max()

    # convert wavelengths from m to nm for nicer plotting
    wavelengths_nm = w.index * 10**9
    w.index = wavelengths_nm
    F.columns = wavelengths_nm

    F_matrix = F.values
    index_w = (wavelengths_nm > 400) & (wavelengths_nm < 700)
    F_matrix = F_matrix[:, index_w]
    selected_w = wavelengths_nm[index_w]
    nr_bands = F_matrix.shape[0]

    fig, axarr = plt.subplots(nr_bands, 1, sharex=True, sharey=True)

    for i in range(nr_bands):
        axarr[i].plot(selected_w, F_matrix[i, :])
        # some tightening up the plot
        axarr[i].get_yaxis().set_visible(False)
        axarr[i].set_xlabel("wavelengths [nm]")

    #save
    out_path = os.path.join(sc.get_full_dir("COLOR_TARGET_RESULTS"),
                            "transform.png")
    plt.savefig(out_path, dpi=250, mode="png", bbox_inches='tight')


def plot_compare(S, C, F, w, d):
    wavelengths = sc.other["RECORDED_WAVELENGTHS"]
    # initialize information about imaging system
    imaging_system = ImagingSystem(wavelengths, F.values, w=w.values, d=d.values)
    # now transform the spectrometer measurements to the cam space
    transformed_colors = transform_color(imaging_system, S.values)
    # store results in pandas dataframe
    spectro_meas_cam_space = pd.DataFrame(data=transformed_colors)

    out_path = os.path.join(sc.get_full_dir("COLOR_TARGET_RESULTS"),
                            "compare.png")

    def plot_for_values(spectro, cam_meas, name):

        #normalize by l1
        def norm(values):
            return values / np.sum(values)
        spectro = spectro.apply(norm, 1)
        cam_meas = cam_meas.apply(norm, 1)
        # otherwise problems with datatype (object v int):
        cam_meas.columns = spectro.columns

        relative_errors = np.abs((1 - spectro / cam_meas) * 100.)
        relative_errors.reset_index(inplace=True)
        mean_errors = np.mean(relative_errors.values, axis=1)
        for i, index in enumerate(relative_errors["index"]):
            relative_errors["index"][i] = str(index) + " -- error: " + str(np.round(mean_errors[i], decimals=2)) + "%"
        relative_errors = pd.melt(relative_errors,
                                  id_vars="index", var_name ="camera_band",
                                  value_name="abs relative error [%]")
        grid = sns.FacetGrid(relative_errors, col="index", col_wrap=6, ylim=(0, 25))
        grid.map(plt.bar, "camera_band", "abs relative error [%]")

        plt.savefig(out_path + "relative.png" + name, dpi=250)

        cam_meas.reset_index(inplace=True)
        cam_meas["device"] = CAMERA
        spectro.reset_index(inplace=True)
        spectro["device"] = "Spectrometer"

        df_cam = pd.concat((cam_meas, spectro), ignore_index=True)
        df_cam = pd.melt(df_cam, id_vars=["index", "device"],
                     var_name="camera_band", value_name=name + " response")
        grid = sns.FacetGrid(df_cam, col="index", hue="device", col_wrap=6, ylim=(0., .15))
        grid.map(plt.plot, "camera_band", name + " response", marker="o", ms=4)
        grid.add_legend()

        plt.savefig(out_path + name, dpi=250)

    plot_for_values(spectro_meas_cam_space, C, "")


def plot_raw_camera(C_uncalibrated):
    df = C_uncalibrated.T

    df["wavelengths [nm]"] = df.index.astype(float) * 10**9

    df = pd.melt(df, id_vars=["wavelengths [nm]"],
                 var_name=EXPERIMENT, value_name="raw camera measurments")
    grid = sns.FacetGrid(df, col=EXPERIMENT, col_wrap=6)
    grid.map(plt.plot, "wavelengths [nm]", "raw camera measurments",
             marker="o", ms=4)
    grid.add_legend()

    out_path = os.path.join(sc.get_full_dir("COLOR_TARGET_RESULTS"),
                    "raw_camera.png")
    plt.savefig(out_path, dpi=250, mode="png")


def plot_raw_spectrometer(S, white, dark):
    S_t = S.T
    S_t["dark"] = dark
    S_t["white"] = white

    S_t["wavelengths [nm]"] = S_t.index.astype(float) * 10**9
    w = S_t["wavelengths [nm]"].values
    index_w = (w > 400) & (w < 700)

    S_t = S_t.iloc[index_w, :]

    df = pd.melt(S_t, id_vars=["wavelengths [nm]"],
                 var_name=EXPERIMENT, value_name="raw reflectances")
    grid = sns.FacetGrid(df, col=EXPERIMENT, col_wrap=6)
    grid.map(plt.plot, "wavelengths [nm]", "raw reflectances",
             marker="o", ms=4)

    grid.add_legend()

    out_path = os.path.join(sc.get_full_dir("COLOR_TARGET_RESULTS"),
                            "raw_spectrometer.png")
    plt.savefig(out_path, dpi=250, mode="png")


if __name__ == '__main__':

    # create folders where data shall be put
    sc.create_folders()

    logging.basicConfig(filename=os.path.join(sc.get_full_dir("LOG_FOLDER"),
                                 "color_board_data" +
                                 str(datetime.datetime.now()) +
                                 '.log'), level=logging.INFO)

    F_folder = os.path.join(TOP, "Ximea_software", "xiSpec-calibration-data",
                            "CMV2K-SSM4x4-470_620-9.2.4.11.xml")
    S_folder = os.path.join(TOP, "data", "spectrometer", color_tiles_loc)
    C_folder = os.path.join(TOP, "data", "Ximea_recordings", color_tiles_loc)

    F = cc.get_camera_calibration_info_df(F_folder)
    S = sm.get_all_spectrometer_measurements_as_df(S_folder)
    C_np = cr.get_camera_reflectances(C_folder, suffix='.bsq', pixel_location=pixel_location, size=window_size)
    C = pd.DataFrame(C_np)
    C_uncalibrated = pd.DataFrame(cr.get_camera_reflectances_uncorrected(C_folder, suffix='.bsq', pixel_location=pixel_location, size=window_size))
    dark = sm.get_spectrometer_measurement_df(os.path.join(TOP, "data", "spectrometer", "dark.txt"))
    white = sm.get_spectrometer_measurement_df(os.path.join(TOP, "data", "spectrometer", "white.txt"))

    # interpolate the F values to fit the wavelengths recorded
    # by the spectrometer
    f = interpolate.interp1d(F.columns, F.values, bounds_error=False, fill_value=0.)
    F_new_np = f(S.columns)
    F_new = pd.DataFrame(data=F_new_np, columns=S.columns)

    # use the wavelengths recorded by the spectrometer as basis for the eval
    sc.other["RECORDED_WAVELENGTHS"] = S.columns

    # specify tasks
    plot_spectro_measurements = plot_raw_spectrometer(S, white, dark)
    plot_imaging_system = plot_imaging_system(F_new, white)
    plot_mapping = plot_compare(S, C, F_new,
                                white, dark)
    plot_raw_camera = plot_raw_camera(C_uncalibrated)