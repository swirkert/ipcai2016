

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

import matplotlib.pylab as plt
import seaborn as sns

from mc.camera import ImagingSystem, transform_color
import get_camera_calibration_info as cc
import get_spectrometer_measurements as sm
import get_camera_reflectances as cr
import get_corrected_filterresponses as cfr

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


def plot_imaging_system(F, title):
    # convert wavelengths from m to nm for nicer plotting
    wavelengths_nm = F.columns * 10**9

    F_matrix = F.values
    index_w = (wavelengths_nm > 400) & (wavelengths_nm < 700)
    F_matrix = F_matrix[:, index_w]
    selected_w = wavelengths_nm[index_w]
    nr_bands = F_matrix.shape[0]

    fig, axarr = plt.subplots(nr_bands, 1, sharex=True, sharey=True)

    for i in range(nr_bands):
        axarr[i].plot(selected_w, F_matrix[i, :])
        # some tidying up the plot
        axarr[i].get_yaxis().set_visible(False)
        axarr[i].set_xlabel("wavelengths [nm]")

    #save
    out_path = os.path.join(sc.get_full_dir("COLOR_TARGET_RESULTS"), title)
    plt.savefig(out_path, dpi=250, mode="png", bbox_inches='tight')


def plot_compare(S, C, F, w, d):
    wavelengths = S.columns
    # initialize information about imaging system
    imaging_system = ImagingSystem(wavelengths, F.values, w=w.values, d=d.values)
    # now transform the spectrometer measurements to the cam space
    transformed_colors = transform_color(imaging_system, S.values)
    # store results in pandas dataframe
    spectro_meas_cam_space = pd.DataFrame(data=transformed_colors)

    out_path = sc.get_full_dir("COLOR_TARGET_RESULTS")
    _plot_for_values(spectro_meas_cam_space, C, out_path, "compare.png")


def plot_compare_adapted(S, C, F, w, d, pc):
    # F in the original recorded wavelengths.
    wavelengths = S.columns
    # the spectrometer measuremnts will be transformed to C
    transformed_colors = np.zeros_like(C.values)
    # for each color tile
    for i in range(S.shape[0]):
        # remove the tile from the data to be fitted
        S_i = S.drop(S.index[i])
        C_i = C.drop(C.index[i])
        # adapt F (basically this is cross validation)
        F_i = cfr.get_corrected_filter_responses_df(S_i, C_i, F, w, d, pc)

        F_new = cfr.to_wav(F_i.columns, F_i, S.columns)

        # initialize information about imaging system
        imaging_system = ImagingSystem(wavelengths, F_new, w=w.values, d=d.values)
        # now transform the spectrometer measurements to the cam space
        transformed_colors_i = transform_color(imaging_system, S.values[i,:])
        transformed_colors[i, :] = transformed_colors_i
    # store results in pandas dataframe
    spectro_meas_cam_space = pd.DataFrame(data=transformed_colors)

    out_path = sc.get_full_dir("COLOR_TARGET_RESULTS")
    _plot_for_values(spectro_meas_cam_space, C, out_path, "compare_adapted.png")


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


def _plot_for_values(spectro, cam_meas, path, name):
    """
    Actual plotting method for comparison of spectrometer and camera measurements

    :param spectro: spectrometer measurements transformed to camera space
    :param cam_meas: flatfield corrected camera measurements
    :param path: the base path
    :param name: file name
    :return:
    """
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
    plt.savefig(os.path.join(path, name + "relativ.png"), dpi=250)

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
    plt.savefig(os.path.join(path, name), dpi=250)


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
    pc = cfr.get_principle_components(F_folder)

    # interpolate the F values to fit the wavelengths recorded
    # by the spectrometer
    F_new = pd.DataFrame(cfr.to_wav_df(F.columns, F, S.columns), columns=S.columns)
    # calculated the adapted filter responses
    F_adapted = cfr.get_corrected_filter_responses_df(S, C, F, white, dark, pc)

    # specify tasks
    plot_raw_spectrometer(S, white, dark)
    plot_imaging_system(F, "transform.png")
    plot_imaging_system(F_adapted, "transform_adapted.png")
    plot_compare(S, C, F_new, white, dark)
    plot_compare_adapted(S, C, F, white, dark, pc)
    plot_raw_camera(C_uncalibrated)