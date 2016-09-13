
import numpy as np
import pandas as pd

from msi.io.spectrometerreader import SpectrometerReader
from helpers.file_operations import *


def get_all_spectrometer_measurements(folder):
    spectro_meas = get_image_files_from_folder(folder, suffix=".txt", fullpath=True)
    w = get_spectrometer_wavelengths(spectro_meas[0])
    measurements_array = np.zeros((len(spectro_meas), len(w)))

    for i, s in enumerate(spectro_meas):
        measurements_array[i] = get_spectrometer_measurement(s)
    return measurements_array


def get_all_spectrometer_measurements_as_df(folder):
    spectro_meas = get_image_files_from_folder(folder, suffix=".txt", fullpath=True)
    w = get_spectrometer_wavelengths(spectro_meas[0])
    measurements_array = get_all_spectrometer_measurements(folder)

    df = pd.DataFrame(data=measurements_array, columns=w)
    return df


def get_spectrometer_measurement(filename):
    spectro_reader = SpectrometerReader()
    return spectro_reader.read(filename).get_image()


def get_spectrometer_measurement_df(filename):
    meas = get_spectrometer_measurement(filename)
    w = get_spectrometer_wavelengths(filename)
    df = pd.DataFrame(data=meas, index=w)
    return df


def get_spectrometer_wavelengths(filename):
    spectro_reader = SpectrometerReader()
    return spectro_reader.read(filename).get_wavelengths()


