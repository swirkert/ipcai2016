
#
"""
Created on 9/12/16

@author: Anant Vemuri

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
import get_camera_calibration_info as gcci
import get_spectrometer_measurements as gsm
import get_camera_reflectances as gcr
import commons


TOP = "/media/avemuri/E130-Projekte/Biophotonics/Data/2016_09_08_Ximea"
SINGLE_TILE_LOC = ["green_multiple_images","red_multipleimages_8Kus",
                   "red_multipleimages_11Kus", "red_multipleimages_14Kus",
                   "red_multipleimages_17Kus", "red_multipleimages_20Kus",
                   "red_avgframes1_exp_adapted", "green_avgframes1_exp_adapted"]
NOTES_SINGLE_TILE_LOC = ["Multiple Green Images","Multiple Red Images at 8Kus Exposure",
                         "Multiple Red Images at 11Kus Exposure",
                         "Multiple Red Images at 14Kus Exposure",
                         "Multiple Red Images at 17Kus Exposure",
                         "Multiple Red Images at 20Kus Exposure",
                         "Multiple Red Images Exposure Adapted 1 Image Averaged",
                         "Multiple Green Images Exposure Adapted 1 Image Averaged"]
BASE_RESULTS_FOLDER = "/media/avemuri/DEV/Data/CameraEvaluation/Ximea"
EXPERIMENT = "color_targets/pixel_noise"# + datetime.date.strftime("%Y_%m_%d_%H_%M_%S")
CAMERA = "ximea"



pixel_location = (80, 228)
window_size = (0,0)#(50, 65)

if __name__ == '__main__':

    i = 0;
    for iSINGLE_TILE_LOC in SINGLE_TILE_LOC:
        C_folder = os.path.join(TOP, "data", "Ximea_recordings", iSINGLE_TILE_LOC)

        C_np = gcr.get_camera_reflectances(C_folder, suffix='.bsq',
                                           pixel_location=pixel_location,
                                           size=window_size)

        C_np_uc = gcr.get_camera_reflectances_uncorrected(C_folder, suffix='.bsq',
                                           pixel_location=pixel_location,
                                           size=window_size)

        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].boxplot(C_np)
        axarr[0].set_title(NOTES_SINGLE_TILE_LOC[i])
        axarr[1].boxplot(C_np_uc)

        axarr[0].set_xlabel('Wavelengths')
        axarr[0].set_ylabel('Std Dev \n Normalized reflectance')
        axarr[1].set_ylabel('Std Dev \n Unnormalized')
        #axarr[0].set_ylim(0,0.03)
        #axarr[1].set_ylim(0, 0.03)



        out_path = os.path.join(BASE_RESULTS_FOLDER, EXPERIMENT, )
        commons.create_folder_if_necessary(out_path)
        out_path = os.path.join(out_path, iSINGLE_TILE_LOC + "_Noise.png")

        plt.savefig(out_path, dpi=250, mode="png")
        i += 1

    #print C_np