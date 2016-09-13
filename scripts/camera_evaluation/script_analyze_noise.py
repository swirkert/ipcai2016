
#
"""
Created on 9/12/16

@author: Anant Vemuri

"""

import os
import numpy as np

import matplotlib.pylab as plt
import get_camera_reflectances as gcr
import commons


TOP = "/media/avemuri/E130-Projekte/Biophotonics/Data/2016_09_08_Ximea"
SINGLE_TILE_LOC = ["green_multiple_images","red_multipleimages_8Kus",
                   "red_multipleimages_11Kus", "red_multipleimages_14Kus",
                   "red_multipleimages_17Kus", "red_multipleimages_20Kus"]
NOTES_SINGLE_TILE_LOC = ["Multiple Green Images","Multiple Red Images at 8Kus Exposure",
                         "Multiple Red Images at 11Kus Exposure",
                         "Multiple Red Images at 14Kus Exposure",
                         "Multiple Red Images at 17Kus Exposure",
                         "Multiple Red Images at 20Kus Exposure"]
BASE_RESULTS_FOLDER = "/media/avemuri/DEV/Data/CameraEvaluation/Ximea"
EXPERIMENT = "color_targets/pixel_noise"# + datetime.date.strftime("%Y_%m_%d_%H_%M_%S")
CAMERA = "ximea"



pixel_location = (80, 228)
window_size = (1,1)#(50, 65)

if __name__ == '__main__':

    i = 0
    for iSINGLE_TILE_LOC in SINGLE_TILE_LOC:
        C_folder = os.path.join(TOP, "data", "Ximea_recordings", iSINGLE_TILE_LOC)

        C_np_uc = gcr.get_camera_reflectances_uncorrected(C_folder, suffix='.bsq',
                                           pixel_location=pixel_location,
                                           size=window_size)

        plt.figure()
        plt.boxplot((C_np_uc-np.mean(C_np_uc, axis=0))/np.mean(C_np_uc, axis=0))

        plt.xlabel('Wavelengths')
        plt.ylabel('Deviation about Mean \n Unnormalized reflectance')

        out_path = os.path.join(BASE_RESULTS_FOLDER, EXPERIMENT)
        commons.create_folder_if_necessary(out_path)
        out_path = os.path.join(out_path, iSINGLE_TILE_LOC + "_Noise.png")

        plt.savefig(out_path, dpi=250, mode="png")
        i += 1

    #print C_np