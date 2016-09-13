
import copy

import numpy as np
import scipy.optimize

from msi.msi import Msi
from msi.normalize import standard_normalizer


class ImagingSystem:
    """
    Encapsulates all the information needed to link spectrometer/reflectance
    measurements to camera measurements which are dependent on the cameras
    imaging system (the filters used, quantum efficiency, ...)

    More concretely the following equation is assumed to hold:

    \begin{eqnarray}
    C_{j,k} & \propto & \frac{\int_{l=0}^m q_l f_{k,l} i_l r_{j,l}}
                           {\int_{l=0}^m q_l f_{k,l} i_l}\\
    & = &  \frac{\int_{l=0}^m  q_l f_{k,l} i_l \frac{s_{j,l} - d_l}{w_l - d_l}}
                {\int_{l=0}^m q_l f_{k,l} i_l}\\
    & = &  \frac{\int_{l=0}^m  q_l f_{k,l} i_l \frac{s_{j,l} - d_l}{i_l}}
                {\int_{l=0}^m q_l f_{k,l} i_l}\\
    & = &  \frac{\int_{l=0}^m  q_l f_{k,l} (s_{j,l} - d_l)}
                {\int_{l=0}^m q_l f_{k,l} i_l}\\
    & = &  \frac{\int_{l=0}^m  q_l f_{k,l} (s_{j,l} - d_l)}
                {\int_{l=0}^m q_l f_{k,l} (w_{l} - d_l)}
    \end{eqnarray}

    ## Dimensions

    dimension | rolling index | means
    -- | -- | --
    n | j | number of measurements
    v | k | number of wavelengths measured by imaging system
    m | l | number of wavelengths measured by spectrometer

    ## Parameters
    parameter | dimension | means
    -- | -- | --
    wavelengths | m |  wavelengths measured by the spectrometer
    F | vxm | Initial estimate on the filter sensitivities
    q | m | Initial estimate on the quantum efficiency of the camera. For each wavelength measured by the spectrometer gives a value on how big the cameras quantum efficiency  is.
    w | m |  white measurement
    d | m |  dark measurement
    S | nxm |  spectrometer measurements
    C | nxv | measurements made by camera.
    """

    def __init__(self, wavelengths, F, q=None, w=None, d=None):
        self.wavelengths = wavelengths
        m = len(self.wavelengths)
        v, m2 = F.shape
        if m != m2:
            raise ValueError("number of wavelengths in the filter specification " +
                             str(m2) +
                             " does not match number of wavelenths " + str(m))
        if q is None:
            q = np.ones(m)
        if w is None:
            w = np.ones(m)
        if d is None:
            d = np.zeros(m)

        self.q = np.squeeze(q)
        self.F = F
        self.w = np.squeeze(w)
        self.d = np.squeeze(d)

    def get_v(self):
        return self.F.shape[0]

    def get_nr_bands(self):
        """
        Alias to self.get_v()

        Returns
        -------
        number of bands of the imaging system (v)
        """
        return self.get_v()


def calibrate(C, S, start_imaging_system ):
    """
    Relates measurements S made by a spectrometer with
    measurements of the same point made by a camera C.
    Optionally, prior knowledge about your image system can be included.

    Parameters
    ---------
    C   measurements made by camera (nxv).
    S   spectrometer measurements (nxm)
    start_imaging_system     best guess for the imaging system as a starting
                                point for the calibration

    Returns
    -------
    F_new   an estimate for F so that transform_reflectance
    on S matches C more closely
    """
    n, m = S.shape
    n2, v = C.shape

    if n != n2:
        raise ValueError("number of measurements made with the camera " + str(n) +
                         "does not match number of measurements made with " +
                         "the spectrometer" + str(n2))

    def optimization_function(F_flattend, C, S, imaging_system):
        # first take the current guess and set the imaging system
        # to this updated value:
        imaging_system.F = np.reshape(F_flattend.copy(),
                                      imaging_system.F.shape)
        # now use the new imaging system to estimate C from S
        C_estimated = transform_color(imaging_system, S)
        # return quadratic error
        return np.sum((C-C_estimated)**2)

    # use available knowledge to start
    imaging_system_optimized = copy.deepcopy(start_imaging_system)
    # flatten because optimize works on 1d arrays
    x0 = imaging_system_optimized.F.flatten()
    # start optimization
    F_final = scipy.optimize.fmin(optimization_function, x0,
                                  (C, S, imaging_system_optimized),
                                  maxiter=10000) # args

    return F_final.reshape(start_imaging_system.F.shape)


def transform_reflectance(imaging_system, R):
    """
    Given a set of reflectances (nxm), transform them to what the imaging
    system measures (no noise added).

    Parameters
    ----------
    R   set of reflectance measurements (nxm). These can e.g. be the output
        of a Monte Carlo simulation.

    Returns
    -------
    C   the measurement transformed to show what the imaging system would
        measure (nxv)
    """
    i = imaging_system  # short alias for imaging system
    C = np.zeros((_nr_samples_to_transform(R), i.get_nr_bands()))

    # iterate over bands of imaging system
    for k in range(i.get_v()):
        combined_imaging_system = i.q * i.F[k, :] * (i.w - i.d)
        vectorized_response = combined_imaging_system * R
        # camera response for band k:
        C_k = np.trapz(vectorized_response, i.wavelengths) / \
              np.trapz(combined_imaging_system, i.wavelengths)
        C[:, k] = C_k

    return np.squeeze(normalize(C))


def transform_color(imaging_system, S, normalize_color=True):
    """
    Given a set of spectrometer measurements (nxm),
    transform them to what the imaging system measures (no noise added).
    The difference to transform_reflectance is that S is not dark and white
    light corrected.

    Parameters
    ----------
    S   set of spectrometer measurements (nxm)

    Returns
    -------
    C   the measurement transformed to show what the imaging system would
        measure (nxv)
    """
    i = imaging_system  # short alias for imaging system
    C = np.zeros((_nr_samples_to_transform(S), i.get_v()))

    # iterate over bands of imaging system
    for k in range(i.get_nr_bands()):
        q_times_Fk = i.q * i.F[k, :]
        vectorized_response = q_times_Fk * (S - i.d)
        # camera response for band k:
        C_k = np.trapz(vectorized_response, i.wavelengths) / \
              np.trapz(q_times_Fk * (i.w - i.d), i.wavelengths)
        C[:, k] = C_k

    if normalize_color:
        C = normalize(C)

    return np.squeeze(C)


def _nr_samples_to_transform(X):
    if len(X.shape) == 1:
        n = 1
    else:
        n = X.shape[0]
    return n


def normalize(C):
    # briefly transform to msi to be able to apply standard normalizer
    msi = Msi()
    msi.set_image(C.copy())
    standard_normalizer.normalize(msi)
    # back to array format used here:
    return msi.get_image()
