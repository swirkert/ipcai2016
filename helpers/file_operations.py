
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


import os


def get_image_files_from_folder(folder, prefix="", suffix=".tiff", fullpath=False):
    # small helper function to get all the image files in a folder
    # it will only return files which end with suffix.
    # if fullpath==True it will return the full path of the file, otherwise
    # only the filename
    # get all filenames
    image_files = [f for f in os.listdir(folder) if
                   os.path.isfile(os.path.join(folder, f))]
    image_files.sort()
    image_files = [f for f in image_files if f.endswith(suffix)]
    image_files = [f for f in image_files if f.startswith(prefix)]
    if fullpath:  # expand to full path if wished
        image_files = [os.path.join(folder, f) for f in image_files]
    return image_files


