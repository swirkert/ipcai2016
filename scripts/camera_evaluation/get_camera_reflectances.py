
"""

Surgical Spectral Imaging Library 2016

Copyright (c) German Cancer Research Center,
Computer Assisted Interventions.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

See LICENSE for details

"""
from helpers.file_operations import *
from msi.io.envireader import EnviReader
import msi.msimanipulations as mani
import numpy as np


def get_camera_reflectances(folder, prefix='', suffix='.tiff',
                            pixel_location=(124,256), size=(10, 10)):
    file_names = get_image_files_from_folder(folder=folder, prefix=prefix,
                                             suffix=suffix, fullpath=False)

    file_reader = EnviReader()

    dark_image = file_reader.read(os.path.join(folder, 'reference', 'dark'),
                                  header_ext='.hdr', file_ext=suffix)
    white_image = file_reader.read(os.path.join(folder, 'reference', 'white'),
                                   header_ext='.hdr', file_ext=suffix)
    reflectance_values = []

    for i in xrange(0,len(file_names)):
        msi = file_reader.read(fileToRead=os.path.join(folder,
                                                       file_names[i][:-len(suffix)]),
                              header_ext='.hdr', file_ext=suffix)

        msi_selection_mani = mani.image_correction(msi, white_image, dark_image)
        image_roi = msi_selection_mani.get_image()[pixel_location[0]:pixel_location[0]+size[0],
                    pixel_location[1]:pixel_location[1]+size[1], :]
        image_mean = np.mean(image_roi, axis=(0, 1))

        reflectance_values.append(image_mean)

    return np.array(reflectance_values)


def get_camera_reflectances_uncorrected(folder, prefix='',suffix='.tiff',
                                        pixel_location=(114,256), size=(10, 10)):

    file_names = get_image_files_from_folder(folder=folder, prefix=prefix,
                                             suffix=suffix, fullpath=False)

    file_reader = EnviReader()
    reflectance_values_uncorrected = []

    for i in xrange(0,len(file_names)):
        msi = file_reader.read(fileToRead=os.path.join(folder,
                                                       file_names[i][:-len(suffix)]),
                              header_ext='.hdr', file_ext=suffix)
        image_roi = msi.get_image()[pixel_location[0]:pixel_location[0]+size[0],
                    pixel_location[1]:pixel_location[1]+size[1], :]
        image_mean = np.mean(image_roi, axis=(0, 1))
        reflectance_values_uncorrected.append(image_mean)

    return np.array(reflectance_values_uncorrected)

#### USAGE
# reflectance_corrected = get_camera_reflectances('/media/avemuri/E130-Projekte/Biophotonics/'
#                         'Data/2016_09_08_Ximea/data/Ximea_recordings/'
#                         'Color_tiles_exposure_adapted/', suffix='.bsq',
#                         pixel_location=(100,200))
#
# print reflectance_corrected, '\n\n'
#
# reflectance_uncorrected = get_camera_reflectances_uncorrected('/media/avemuri/E130-Projekte/Biophotonics/'
#                         'Data/2016_09_08_Ximea/data/Ximea_recordings/'
#                         'Color_tiles_exposure_adapted/', suffix='.bsq',
#                         pixel_location=(100,200))
#
# print reflectance_uncorrected, '\n\n'
