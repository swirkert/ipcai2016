import copy
import os

import numpy as np
import pandas as pd
import scipy

from mc.camera import transform_color, ImagingSystem
import get_camera_calibration_info as cc
import get_spectrometer_measurements as sm
import get_camera_reflectances as cr


def get_principle_components_info_df(filename):
    filter_responses = cc.get_camera_calibration_info_df(filename)
    w = filter_responses.columns
    fr_matrix = filter_responses.values
    fr_shape = fr_matrix.shape
    principal_components = np.zeros((fr_shape[0], 2, fr_shape[1]))
    for i in range(fr_matrix.shape[0]):
        if i < 8:
            mask = w*10**9 < 575
        else:
            mask = w*10**9 < 525
        principal_components[i, 0, :] = fr_matrix[i, :]
        principal_components[i, 1, :] = fr_matrix[i, :]
        principal_components[i, 0, ~mask] = 0.
        principal_components[i, 1,  mask] = 0.

    return principal_components


def get_corrected_filter_responses_df(calibration_file, S, C, w, d, s_wav):
    filters = cc.get_camera_calibration_info_df(calibration_file)
    F_wav = filters.columns
    F = filters.values
    pc = get_principle_components_info_df(calibration_file)
    F_opt = np.zeros_like(F)

    # transform spectrometer measurements to F wavelengths:
    S = _to_F_wav(s_wav, S, F_wav)
    w = _to_F_wav(s_wav, w, F_wav)
    d = _to_F_wav(s_wav, d, F_wav)

    C = C / np.sum(C, axis=1)[:, np.newaxis]

    def optimization_function(x0, C_opt, S_opt, imaging_system, pc):
        # first take the current guess and set the imaging system
        # to this updated value:
        modified_imaging_system = copy.deepcopy(imaging_system)
        modified_imaging_system.F = _eval_filter_basis(pc, x0)
        # now use the new imaging system to estimate C from S
        C_estimated = transform_color(modified_imaging_system, S_opt,
                                      normalize_color=True)
        # return quadratic error
        return np.sum((C_opt-C_estimated)**2)

    imaging_system = ImagingSystem(F_wav, F, q=None, w=w, d=d)
    # find factors for principle components
    x0 = np.ones(pc.shape[0] * pc.shape[1])
    x0_opt = x0
    # start optimization
    x0_opt = scipy.optimize.least_squares(optimization_function, x0,
                                          bounds=(0., np.inf),
                                          args=(C, S, imaging_system, pc))["x"]
    # x0_opt = scipy.optimize.minimize(optimization_function, x0,
    #                                  args=(C, S, imaging_system, pc),
    #                                  method='Nelder-Mead',
    #                                  options={"maxiter":10000}
    #                                  )["x"]
    F_opt = _eval_filter_basis(pc, x0_opt)
    df = pd.DataFrame(data=F_opt, columns=filters.columns)
    return df


def _eval_filter_basis(basis, factors):
    # basis as column vectors, factors as 1-d
    result = np.zeros((basis.shape[0], basis.shape[2]))
    pairwise_factors = np.reshape(factors, (-1, basis.shape[1]))
    for i, b in enumerate(basis):
        result[i] = np.dot(b.T, pairwise_factors[i, :])
    return result


def _to_F_wav(spectro_wav, spectro, f_wav):
    # interpolate the spectrometer values to fit the wavelengths recorded
    # by the filter calibration
    f = scipy.interpolate.interp1d(spectro_wav, spectro, bounds_error=False, fill_value=0.)
    s_new = f(f_wav)
    return s_new


def hack_get_integration_times():
    return np.array([8., 14.6, 23.2, 40.7, 86.8, 191.2,
                     89.6, 71.6, 19.1, 14.1, 18.1, 50.6,
                     18.1, 54.6, 16.6, 68.6, 27.1, 15.1,
                     38.9, 15.1, 38.1, 68.6, 28.1, 26.6])

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    TOP = "/media/wirkert/data/Data/2016_09_08_Ximea"
    color_tiles_loc = "Color_tiles_exposure_adapted"
    pixel_location = (80, 228)
    window_size = (50, 65)

    calib_file = os.path.join(TOP, "Ximea_software/xiSpec-calibration-data/CMV2K-SSM4x4-470_620-9.2.4.11.xml")
    S_folder = os.path.join(TOP, "data", "spectrometer", color_tiles_loc)
    C_folder = os.path.join(TOP, "data", "Ximea_recordings", color_tiles_loc)

    S = sm.get_all_spectrometer_measurements_as_df(S_folder)
    C_np = cr.get_camera_reflectances(C_folder, suffix='.bsq', pixel_location=pixel_location, size=window_size)
    C = pd.DataFrame(C_np)
    d = sm.get_spectrometer_measurement(os.path.join(TOP, "data", "spectrometer", "dark.txt"))
    w = sm.get_spectrometer_measurement(os.path.join(TOP, "data", "spectrometer", "white.txt"))

    # hack_it = hack_get_integration_times()
    # C_np = C_np / hack_it[:, np.newaxis]

    get_corrected_filter_responses_df(calib_file, S.values, C_np, w, d, S.columns)

