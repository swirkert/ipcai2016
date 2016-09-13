
#
"""
Created on 9/13/16 

@author: Anant Vemuri

"""

import os
import numpy as np

import matplotlib.pylab as plt
import get_camera_reflectances as gcr
import commons



TOP = "/media/avemuri/E130-Projekte/Biophotonics/Data/2016_09_08_Ximea"
SINGLE_TILE_LOC = ["red_avgframes1_exp_adapted", "green_avgframes1_exp_adapted"]
NOTES_SINGLE_TILE_LOC = ["Multiple Red Images Exposure Adapted 1 Image Averaged",
                         "Multiple Green Images Exposure Adapted 1 Image Averaged"]
# labels_red = ['8', '10', '12', '14', '16',
#               '18','19','20','21','22',
#               '23','24','25']

labels_red = [8, 10, 12, 14, 16, 18,19,20,21,22, 23,24,25]

labels_green = [8, 10, 20, 30, 40, 50,60,70,80]


BASE_RESULTS_FOLDER = "/media/avemuri/DEV/Data/CameraEvaluation/Ximea"
EXPERIMENT = "color_targets/camera_linearity"
# + datetime.date.strftime("%Y_%m_%d_%H_%M_%S")
CAMERA = "ximea"



pixel_location = (80, 228)
window_size = (50, 65)

if __name__ == '__main__':
    i = 0


    # RED TILE
    iSINGLE_TILE_LOC = SINGLE_TILE_LOC[0]
    C_folder = os.path.join(TOP, "data", "Ximea_recordings", iSINGLE_TILE_LOC)
    C_np_uc = gcr.get_camera_reflectances_uncorrected(C_folder, suffix='.bsq',
                                                          pixel_location=pixel_location,
                                                          size=window_size)
    plt.figure()
    plt.plot(labels_red, C_np_uc, marker='o', linewidth=1.5)
    plt.legend(shadow=True, fancybox=True)
    plt.title('Reflectances for red tile')
    plt.xlabel('Exposure Time (x10^3 useconds)')
    plt.ylabel('Unnormalized reflectance')


    out_path = os.path.join(BASE_RESULTS_FOLDER, EXPERIMENT)
    commons.create_folder_if_necessary(out_path)
    out_path = os.path.join(out_path, iSINGLE_TILE_LOC + "_Linearity.png")
    plt.savefig(out_path, dpi=250, mode="png")

    # GREEN TILE
    iSINGLE_TILE_LOC = SINGLE_TILE_LOC[1]
    C_folder = os.path.join(TOP, "data", "Ximea_recordings", iSINGLE_TILE_LOC)
    C_np_uc = gcr.get_camera_reflectances_uncorrected(C_folder, suffix='.bsq',
                                                      pixel_location=pixel_location,
                                                      size=window_size)
    plt.figure()
    plt.plot(labels_green, C_np_uc, marker='o', linewidth=1.5)
    plt.title('Reflectances for green tile')
    plt.xlabel('Exposure Time (x10^3 useconds)')
    plt.ylabel('Unnormalized reflectance')

    out_path = os.path.join(BASE_RESULTS_FOLDER, EXPERIMENT)
    commons.create_folder_if_necessary(out_path)
    out_path = os.path.join(out_path, iSINGLE_TILE_LOC + "_Linearity.png")
    plt.savefig(out_path, dpi=250, mode="png")
