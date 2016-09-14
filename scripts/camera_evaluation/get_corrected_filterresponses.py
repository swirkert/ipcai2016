import copy

import numpy as np
import pandas as pd
import scipy

from mc.camera import transform_color, ImagingSystem
import get_camera_calibration_info as cc


def get_principle_components(filename):
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


def get_corrected_filter_responses_df(S, C, F, w, d, pc):
    # transform input which is in pandas dataframes to np arrays
    F_wav = F.columns
    F = F.values
    # transform spectrometer measurements to F wavelengths:
    spectrometer_wavelengths = S.columns
    S = to_wav_df(spectrometer_wavelengths, S, F_wav)
    w = to_wav_df(spectrometer_wavelengths, w, F_wav)
    d = to_wav_df(spectrometer_wavelengths, d, F_wav)

    C = C.values / np.sum(C.values, axis=1)[:, np.newaxis]

    def optimization_function(x0, C_opt, S_opt, imaging_system, pc):
        # first take the current guess and set the imaging system
        # to this updated value:
        modified_imaging_system = copy.deepcopy(imaging_system)
        modified_imaging_system.F = _eval_filter_basis(pc, x0)
        # now use the new imaging system to estimate C from S
        C_estimated = transform_color(modified_imaging_system, S_opt,
                                      normalize_color=True)
        # return residuals
        residuals = (C_opt - C_estimated).flatten()
        return residuals

    imaging_system = ImagingSystem(F_wav, F, q=None, w=w, d=d)
    # find factors for principle components
    x0 = np.ones(pc.shape[0] * pc.shape[1])
    # start optimization
    x0_opt = scipy.optimize.least_squares(optimization_function, x0,
                                          bounds=(0., np.inf),
                                          args=(C, S, imaging_system, pc))["x"]
    F_opt = _eval_filter_basis(pc, x0_opt)
    df = pd.DataFrame(data=F_opt, columns=F_wav)
    return df


def _eval_filter_basis(basis, factors):
    # basis as column vectors, factors as 1-d
    result = np.zeros((basis.shape[0], basis.shape[2]))
    pairwise_factors = np.reshape(factors, (-1, basis.shape[1]))
    for i, b in enumerate(basis):
        result[i] = np.dot(b.T, pairwise_factors[i, :])
    return result


def to_wav_df(spectro_wav, spectro, f_wav):
    return to_wav(spectro_wav, spectro.values, f_wav)


def to_wav(spectro_wav, spectro, f_wav):
    # interpolate the spectrometer values to fit the wavelengths recorded
    # by the filter calibration
    f = scipy.interpolate.interp1d(spectro_wav, np.squeeze(spectro),
                                   bounds_error=False, fill_value=0.)
    s_new = f(f_wav)
    return s_new


def hack_get_integration_times():
    return np.array([8., 14.6, 23.2, 40.7, 86.8, 191.2,
                     89.6, 71.6, 19.1, 14.1, 18.1, 50.6,
                     18.1, 54.6, 16.6, 68.6, 27.1, 15.1,
                     38.9, 15.1, 38.1, 68.6, 28.1, 26.6])


